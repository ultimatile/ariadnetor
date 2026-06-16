//! Inner product, norm, and expectation value for MPS.

use arnet_core::Scalar;
use arnet_linalg::{
    BlockSparseContractResult, contract_block_sparse_with_backend, contract_with_backend,
};
use arnet_tensor::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, DenseLayout,
    DenseStorage, DenseTensor, Direction, OpsFor, QNIndex, Sector,
};
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
pub(super) fn inner_dense<T, B>(
    backend: &B,
    psi: &Mps<DenseStorage<T>, DenseLayout>,
    phi: &Mps<DenseStorage<T>, DenseLayout>,
) -> T
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert!(n > 0, "MPS must have at least one site");

    // Environment: (χ_ψ, χ_φ), starts as 1×1 identity.
    let mut env: DenseTensor<T> = DenseTensor::ones(vec![1, 1]);

    for j in 0..n {
        let psi_j = psi.site(j).conj();
        let phi_j = phi.site(j);

        // env(a,b) × conj(ψ)(a,d,c) → temp(b,d,c)
        let temp = contract_with_backend(backend, &env, &psi_j, "ab,adc->bdc")
            .expect("inner product contraction failed");

        // temp(b,d,c) × φ(b,d,e) → new_env(c,e)
        env = contract_with_backend(backend, &temp, phi_j, "bdc,bde->ce")
            .expect("inner product contraction failed");
    }

    // Final environment is 1×1; extract the single element.
    env.data_slice()[0]
}

/// Compute the norm ‖ψ‖ = √⟨ψ|ψ⟩.
///
/// The kernel exploits canonical form when available:
/// - `Left` / `Right`: returns 1.0 (normalized by construction).
/// - `Mixed`: returns Frobenius norm of the orthogonality center
///   tensor (O(d) where d is the center site size).
/// - Otherwise: full inner-product evaluation, O(n × χ³).
pub(super) fn norm_dense<T, B>(backend: &B, psi: &Mps<DenseStorage<T>, DenseLayout>) -> T::Real
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    match psi.canonical_form() {
        CanonicalForm::Left | CanonicalForm::Right => T::Real::one(),
        CanonicalForm::Mixed { center } => psi.site(*center).norm(),
        _ => {
            let overlap = inner_dense(backend, psi, psi);
            overlap.re().sqrt()
        }
    }
}

/// Compute ⟨ψ|A|φ⟩ — the MPO-inserted inner product (bra-ket with operator).
pub(super) fn braket_dense<T, B>(
    backend: &B,
    psi: &Mps<DenseStorage<T>, DenseLayout>,
    op: &Mpo<DenseStorage<T>, DenseLayout>,
    phi: &Mps<DenseStorage<T>, DenseLayout>,
) -> T
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert_eq!(n, op.len(), "MPO length must match MPS length");
    assert!(n > 0, "must have at least one site");

    // Environment: (χ_ψ, χ_A, χ_φ), starts as 1×1×1.
    let mut env: DenseTensor<T> = DenseTensor::ones(vec![1, 1, 1]);

    for j in 0..n {
        let psi_j = psi.site(j).conj(); // bra: (ψ_L, d_bra, ψ_R)
        let a_j = op.site(j); // operator: (A_L, d_ket, d_bra, A_R)
        let phi_j = phi.site(j); // ket: (φ_L, d_ket, φ_R)

        // env(a,b,c) × conj(ψ)(a,d,e) → temp1(b,c,d,e)
        let temp1 = contract_with_backend(backend, &env, &psi_j, "abc,ade->bcde")
            .expect("braket contraction step 1 failed");

        // temp1(b,c,d,e) × A(b,f,d,g) → temp2(c,e,f,g)
        let temp2 = contract_with_backend(backend, &temp1, a_j, "bcde,bfdg->cefg")
            .expect("braket contraction step 2 failed");

        // temp2(c,e,f,g) × φ(c,f,h) → env_new(e,g,h)
        env = contract_with_backend(backend, &temp2, phi_j, "cefg,cfh->egh")
            .expect("braket contraction step 3 failed");
    }

    // Final environment is 1×1×1; extract the single element.
    env.data_slice()[0]
}

// ============================================================================
// BlockSparse inner product, norm, and braket
// ============================================================================

/// Compute the inner product ⟨ψ|φ⟩ for block-sparse MPS via the transfer
/// matrix method.
pub(super) fn inner_bsp<T, S, B>(
    backend: &B,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    phi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
) -> T
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert!(n > 0, "MPS must have at least one site");

    let mut env = {
        let psi_left = &psi.site(0).data().layout().indices()[0];
        let phi_left = &phi.site(0).data().layout().indices()[0];
        let env_leg0 = QNIndex::new(psi_left.blocks().to_vec(), psi_left.direction());
        let phi_dir_flipped = match phi_left.direction() {
            Direction::Out => Direction::In,
            Direction::In => Direction::Out,
        };
        let env_leg1 = QNIndex::new(phi_left.blocks().to_vec(), phi_dir_flipped);
        let mut e = BlockSparseTensor::<T, S>::zeros(vec![env_leg0, env_leg1], S::identity());
        if let Some(d) = e.data_mut().block_data_mut(&BlockCoord(vec![0, 0])) {
            d[0] = T::one();
        }
        e
    };

    for j in 0..n {
        let bra_j = psi.site(j).dagger();
        let phi_j = phi.site(j);

        let step1 = match contract_block_sparse_with_backend(backend, &env, &bra_j, &[0], &[0])
            .expect("inner product step 1 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 1 always produces a tensor (rank >= 2)")
            }
        };

        env = match contract_block_sparse_with_backend(backend, &step1, phi_j, &[0, 1], &[0, 1])
            .expect("inner product step 2 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 2 always produces a tensor (output rank is 2)")
            }
        };
    }

    // Extract scalar from the final rank-2 env (shape [1, 1]).
    match env.data().block_data(&BlockCoord(vec![0, 0])) {
        None => T::zero(),
        Some(d) => {
            assert_eq!(
                d.len(),
                1,
                "final environment must be 1×1 (MPS boundary bonds must be dim 1)",
            );
            d[0]
        }
    }
}

/// Compute ⟨ψ|A|φ⟩ for block-sparse MPS/MPO via the transfer matrix method.
pub(super) fn braket_bsp<T, S, B>(
    backend: &B,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    phi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
) -> T
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert_eq!(n, op.len(), "MPO length must match MPS length");
    assert!(n > 0, "must have at least one site");

    let mut env = {
        let psi_left = &psi.site(0).data().layout().indices()[0];
        let a_left = &op.site(0).data().layout().indices()[0];
        let phi_left = &phi.site(0).data().layout().indices()[0];
        let flip = |d: Direction| match d {
            Direction::Out => Direction::In,
            Direction::In => Direction::Out,
        };
        let env_leg0 = QNIndex::new(psi_left.blocks().to_vec(), psi_left.direction());
        let env_leg1 = QNIndex::new(a_left.blocks().to_vec(), flip(a_left.direction()));
        let env_leg2 = QNIndex::new(phi_left.blocks().to_vec(), flip(phi_left.direction()));
        let mut e =
            BlockSparseTensor::<T, S>::zeros(vec![env_leg0, env_leg1, env_leg2], S::identity());
        if let Some(d) = e.data_mut().block_data_mut(&BlockCoord(vec![0, 0, 0])) {
            d[0] = T::one();
        }
        e
    };

    for j in 0..n {
        let bra_j = psi.site(j).dagger();
        let a_j = op.site(j);
        let phi_j = phi.site(j);

        let step1 = match contract_block_sparse_with_backend(backend, &env, &bra_j, &[0], &[0])
            .expect("braket step 1 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 1 always produces a tensor (rank >= 3)")
            }
        };

        let step2 = match contract_block_sparse_with_backend(backend, &step1, a_j, &[0, 2], &[0, 2])
            .expect("braket step 2 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 2 always produces a tensor (rank >= 2)")
            }
        };

        env = match contract_block_sparse_with_backend(backend, &step2, phi_j, &[0, 2], &[0, 1])
            .expect("braket step 3 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 3 always produces a tensor (rank >= 1)")
            }
        };
    }

    match env.data().block_data(&BlockCoord(vec![0, 0, 0])) {
        None => T::zero(),
        Some(d) => {
            assert_eq!(
                d.len(),
                1,
                "final braket environment must be 1×1×1 (boundary bonds must be dim 1)",
            );
            d[0]
        }
    }
}

/// Compute the norm ‖ψ‖ = √⟨ψ|ψ⟩ for a block-sparse MPS.
pub(super) fn norm_bsp<T, S, B>(
    backend: &B,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
) -> T::Real
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    match psi.canonical_form() {
        CanonicalForm::Left | CanonicalForm::Right => T::Real::one(),
        CanonicalForm::Mixed { center } => psi.site(*center).norm(),
        _ => {
            let overlap = inner_bsp(backend, psi, psi);
            overlap.re().sqrt()
        }
    }
}
