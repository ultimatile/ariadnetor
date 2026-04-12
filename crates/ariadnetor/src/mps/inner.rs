//! Inner product, norm, and expectation value for MPS

use arnet_core::backend::ComputeBackend;
use arnet_core::scalar::Scalar;
use arnet_linalg::{BlockSparseContractResult, contract, contract_block_sparse};
use arnet_tensor::block_sparse::{BlockCoord, BlockSparse, Direction, QNIndex};
use arnet_tensor::sector::Sector;
use arnet_tensor::{ComputeBackendTensorExt, Dense};
use num_traits::{Float, One};

use super::chain::TensorChain;
use super::types::{CanonicalForm, Mpo, Mps};

/// Compute the inner product ⟨ψ|φ⟩ of two MPS via the transfer matrix method.
///
/// Contracts left-to-right, accumulating a (χ_ψ × χ_φ) environment tensor.
///
/// # Panics
///
/// Panics if the MPS lengths differ or either is empty.
pub fn inner<T, B>(psi: &Mps<Dense<T>, B>, phi: &Mps<Dense<T>, B>) -> T
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert!(n > 0, "MPS must have at least one site");

    let backend = psi.backend();

    // Environment: (χ_ψ, χ_φ), starts as 1×1 identity
    let mut env = backend.make_tensor(vec![T::one()], vec![1, 1]);

    for j in 0..n {
        let psi_j = psi.storage(j).conj();
        let phi_j = phi.storage(j);

        // env(a,b) × conj(ψ)(a,d,c) → temp(b,d,c)
        let temp = contract(backend, &env, &psi_j, "ab,adc->bdc")
            .expect("inner product contraction failed");

        // temp(b,d,c) × φ(b,d,e) → new_env(c,e)
        env = contract(backend, &temp, phi_j, "bdc,bde->ce")
            .expect("inner product contraction failed");
    }

    env.get(&[0, 0])
}

/// Compute the norm ‖ψ‖ = √⟨ψ|ψ⟩.
///
/// If the MPS is in canonical form, exploits the structure for O(1) computation:
/// - `Left`/`Right`: returns 1.0 (normalized by construction).
/// - `Mixed`: returns Frobenius norm of the orthogonality center tensor.
///
/// Otherwise computes the full inner product.
pub fn norm<T, B>(psi: &Mps<Dense<T>, B>) -> T::Real
where
    T: Scalar,
    B: ComputeBackend,
{
    match psi.canonical_form() {
        CanonicalForm::Left | CanonicalForm::Right => T::Real::one(),
        CanonicalForm::Mixed { center } => psi.storage(*center).norm(),
        _ => {
            let overlap = inner(psi, psi);
            overlap.re().sqrt()
        }
    }
}

/// Compute ⟨ψ|A|φ⟩ — the MPO-inserted inner product (bra-ket with operator).
///
/// Contracts left-to-right with a (χ_ψ × χ_A × χ_φ) environment tensor.
/// When `psi == phi`, this is the expectation value of `A`.
///
/// # Panics
///
/// Panics if the MPS/MPO lengths differ or any is empty.
pub fn braket<T, B>(psi: &Mps<Dense<T>, B>, op: &Mpo<Dense<T>, B>, phi: &Mps<Dense<T>, B>) -> T
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert_eq!(n, op.len(), "MPO length must match MPS length");
    assert!(n > 0, "must have at least one site");

    let backend = psi.backend();

    // Environment: (χ_ψ, χ_A, χ_φ), starts as 1×1×1
    let mut env = backend.make_tensor(vec![T::one()], vec![1, 1, 1]);

    for j in 0..n {
        let psi_j = psi.storage(j).conj(); // bra: (ψ_L, d_bra, ψ_R)
        let a_j = op.storage(j); // operator: (A_L, d_ket, d_bra, A_R)
        let phi_j = phi.storage(j); // ket: (φ_L, d_ket, φ_R)

        // Step 1: env(a,b,c) × conj(ψ)(a,d,e) → temp1(b,c,d,e)
        let temp1 = contract(backend, &env, &psi_j, "abc,ade->bcde")
            .expect("expect contraction step 1 failed");

        // Step 2: temp1(b,c,d,e) × A(b,f,d,g) → temp2(c,e,f,g)
        let temp2 = contract(backend, &temp1, a_j, "bcde,bfdg->cefg")
            .expect("expect contraction step 2 failed");

        // Step 3: temp2(c,e,f,g) × φ(c,f,h) → env_new(e,g,h)
        env = contract(backend, &temp2, phi_j, "cefg,cfh->egh")
            .expect("expect contraction step 3 failed");
    }

    env.get(&[0, 0, 0])
}

// ============================================================================
// BlockSparse inner product and norm
// ============================================================================

/// Compute the inner product ⟨ψ|φ⟩ of two block-sparse MPS via the transfer
/// matrix method.
///
/// Uses [`BlockSparse::dagger`] to create the bra tensor with flipped
/// directions, then contracts left-to-right with two
/// [`contract_block_sparse`] steps per site:
///
/// 1. `contract(env, dagger(ψ_j), [0], [0])` — absorb bra's left bond
/// 2. `contract(result, φ_j, [0,1], [0,1])` — absorb ket's left bond + physical
///
/// Returns `T::zero()` when the MPS states have incompatible total flux
/// (the final environment has no allowed blocks).
///
/// # Panics
///
/// Panics if the MPS lengths differ or either is empty.
pub fn inner_block_sparse<T, S, B>(
    psi: &Mps<BlockSparse<T, S>, B>,
    phi: &Mps<BlockSparse<T, S>, B>,
) -> T
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert!(n > 0, "MPS must have at least one site");

    let backend = psi.backend();

    // Initial environment: rank-2 identity tensor matching the left boundaries.
    // Leg 0 contracts with dagger(psi)'s left bond (which has flipped direction),
    //   so env[0] keeps psi's original direction.
    // Leg 1 contracts with phi's left bond (via step2 result),
    //   so env[1] has the opposite direction.
    let mut env = {
        let psi_left = &psi.storage(0).indices()[0];
        let phi_left = &phi.storage(0).indices()[0];
        let env_leg0 = QNIndex::new(psi_left.blocks().to_vec(), psi_left.direction());
        let phi_dir_flipped = match phi_left.direction() {
            Direction::Out => Direction::In,
            Direction::In => Direction::Out,
        };
        let env_leg1 = QNIndex::new(phi_left.blocks().to_vec(), phi_dir_flipped);
        let mut e = BlockSparse::<T, S>::zeros(vec![env_leg0, env_leg1], S::identity());
        if let Some(d) = e.block_data_mut(&BlockCoord(vec![0, 0])) {
            d[0] = T::one();
        }
        e
    };

    for j in 0..n {
        let bra_j = psi.storage(j).dagger();
        let phi_j = phi.storage(j);

        // Step 1: env(a,b) × bra(a,d,c) → result(b,d,c)
        let step1 = match contract_block_sparse(backend, &env, &bra_j, &[0], &[0])
            .expect("inner product step 1 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 1 always produces a tensor (rank >= 2)")
            }
        };

        // Step 2: result(b,d,c) × phi(b,d,e) → new_env(c,e)
        env = match contract_block_sparse(backend, &step1, phi_j, &[0, 1], &[0, 1])
            .expect("inner product step 2 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 2 always produces a tensor (output rank is 2)")
            }
        };
    }

    // Extract scalar from the final rank-2 env (shape [1, 1]).
    // Returns zero when flux mismatch leaves no allowed blocks.
    match env.block_data(&BlockCoord(vec![0, 0])) {
        None => T::zero(),
        Some(d) => {
            assert_eq!(
                d.len(),
                1,
                "final environment must be 1×1 (MPS boundary bonds must be dim 1)"
            );
            d[0]
        }
    }
}

/// Compute ⟨ψ|A|φ⟩ for block-sparse MPS/MPO via the transfer matrix method.
///
/// Uses [`BlockSparse::dagger`] for the bra and three
/// [`contract_block_sparse`] steps per site:
///
/// 1. `contract(env, dagger(ψ_j), [0], [0])` — absorb bra's left bond
/// 2. `contract(step1, A_j, [0,2], [0,2])` — absorb MPO's left bond + bra physical
/// 3. `contract(step2, φ_j, [0,2], [0,1])` — absorb ket's left bond + ket physical
///
/// The BlockSparse MPO leg direction convention is `(Out, In, Out, In)`
/// for `(χ_L, d_ket, d_bra, χ_R)`.
///
/// Returns `T::zero()` when the states/operator have incompatible flux.
///
/// # Panics
///
/// Panics if the MPS/MPO lengths differ or any is empty.
pub fn braket_block_sparse<T, S, B>(
    psi: &Mps<BlockSparse<T, S>, B>,
    op: &Mpo<BlockSparse<T, S>, B>,
    phi: &Mps<BlockSparse<T, S>, B>,
) -> T
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert_eq!(n, op.len(), "MPO length must match MPS length");
    assert!(n > 0, "must have at least one site");

    let backend = psi.backend();

    // Initial environment: rank-3 identity tensor.
    // Leg 0 contracts with dagger(psi)'s left → same direction as psi's left.
    // Leg 1 contracts with A's left → opposite direction.
    // Leg 2 contracts with phi's left (via step3 result) → opposite direction.
    let mut env = {
        let psi_left = &psi.storage(0).indices()[0];
        let a_left = &op.storage(0).indices()[0];
        let phi_left = &phi.storage(0).indices()[0];
        let flip = |d: Direction| match d {
            Direction::Out => Direction::In,
            Direction::In => Direction::Out,
        };
        let env_leg0 = QNIndex::new(psi_left.blocks().to_vec(), psi_left.direction());
        let env_leg1 = QNIndex::new(a_left.blocks().to_vec(), flip(a_left.direction()));
        let env_leg2 = QNIndex::new(phi_left.blocks().to_vec(), flip(phi_left.direction()));
        let mut e = BlockSparse::<T, S>::zeros(vec![env_leg0, env_leg1, env_leg2], S::identity());
        if let Some(d) = e.block_data_mut(&BlockCoord(vec![0, 0, 0])) {
            d[0] = T::one();
        }
        e
    };

    for j in 0..n {
        let bra_j = psi.storage(j).dagger();
        let a_j = op.storage(j);
        let phi_j = phi.storage(j);

        // Step 1: env(a,b,c) × bra(a,d,e) → step1(b,c,d,e)
        let step1 = match contract_block_sparse(backend, &env, &bra_j, &[0], &[0])
            .expect("braket step 1 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 1 always produces a tensor (rank >= 3)")
            }
        };

        // Step 2: step1(b,c,d,e) × A(b,f,d,g) → step2(c,e,f,g)
        // Contract step1[0] with A[0] (left bonds), step1[2] with A[2] (bra physical)
        let step2 = match contract_block_sparse(backend, &step1, a_j, &[0, 2], &[0, 2])
            .expect("braket step 2 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 2 always produces a tensor (rank >= 2)")
            }
        };

        // Step 3: step2(c,e,f,g) × phi(c,f,h) → new_env(e,g,h)
        // Contract step2[0] with phi[0] (phi left bond), step2[2] with phi[1] (ket physical)
        env = match contract_block_sparse(backend, &step2, phi_j, &[0, 2], &[0, 1])
            .expect("braket step 3 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 3 always produces a tensor (rank >= 1)")
            }
        };
    }

    // Extract scalar from the final rank-3 env (shape [1, 1, 1]).
    match env.block_data(&BlockCoord(vec![0, 0, 0])) {
        None => T::zero(),
        Some(d) => {
            assert_eq!(
                d.len(),
                1,
                "final braket environment must be 1×1×1 (boundary bonds must be dim 1)"
            );
            d[0]
        }
    }
}

/// Compute the norm ‖ψ‖ = √⟨ψ|ψ⟩ for a block-sparse MPS.
///
/// Exploits canonical form when available:
/// - `Left` / `Right`: normalized by construction → 1.0.
/// - `Mixed { center }`: Frobenius norm of the center tensor.
/// - Otherwise: full inner product via [`inner_block_sparse`].
pub fn norm_block_sparse<T, S, B>(psi: &Mps<BlockSparse<T, S>, B>) -> T::Real
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    match psi.canonical_form() {
        CanonicalForm::Left | CanonicalForm::Right => T::Real::one(),
        CanonicalForm::Mixed { center } => psi.storage(*center).norm(),
        _ => {
            let overlap = inner_block_sparse(psi, psi);
            overlap.re().sqrt()
        }
    }
}
