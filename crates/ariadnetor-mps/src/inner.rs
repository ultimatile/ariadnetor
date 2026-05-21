//! Inner product, norm, and expectation value for MPS.

use std::sync::Arc;

use arnet::{
    BlockCoord, BlockSparseContractResult, BlockSparseLayout, BlockSparseStorage,
    BlockSparseTensor, ComputeBackend, DenseLayout, DenseStorage, DenseTensor, DenseTensorData,
    Direction, QNIndex, Scalar, Sector, Tensor, contract, contract_block_sparse,
};
use num_traits::{Float, One, Zero};

use super::chain::TensorChain;
use super::internal_helpers::{bsp_dagger, dense_conj};
use super::types::{CanonicalForm, Mpo, Mps};

/// Compute the inner product ⟨ψ|φ⟩ of two MPS via the transfer matrix method.
///
/// Contracts left-to-right, accumulating a (χ_ψ × χ_φ) environment tensor.
///
/// # Panics
///
/// Panics if the MPS lengths differ or either is empty.
pub(super) fn inner_dense<T, B>(
    psi: &Mps<DenseStorage<T>, DenseLayout, B>,
    phi: &Mps<DenseStorage<T>, DenseLayout, B>,
) -> T
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert!(n > 0, "MPS must have at least one site");

    let backend_arc = Arc::clone(psi.backend_arc());
    let order = backend_arc.preferred_order();

    // Environment: (χ_ψ, χ_φ), starts as 1×1 identity.
    let env_td = DenseTensorData::from_raw_parts(vec![T::one()], vec![1, 1], order);
    let mut env: DenseTensor<T, B> =
        Tensor::<DenseStorage<T>, DenseLayout, B>::with_backend(env_td, Arc::clone(&backend_arc));

    for j in 0..n {
        let psi_j = dense_conj(psi.site(j));
        let phi_j = phi.site(j);

        // env(a,b) × conj(ψ)(a,d,c) → temp(b,d,c)
        let temp = contract(&env, &psi_j, "ab,adc->bdc").expect("inner product contraction failed");

        // temp(b,d,c) × φ(b,d,e) → new_env(c,e)
        env = contract(&temp, phi_j, "bdc,bde->ce").expect("inner product contraction failed");
    }

    // Final environment is 1×1; extract the single element.
    env.data_slice()[0]
}

/// Compute the norm ‖ψ‖ = √⟨ψ|ψ⟩.
///
/// If the MPS is in canonical form, exploits the structure for O(1)
/// computation:
/// - `Left` / `Right`: returns 1.0 (normalized by construction).
/// - `Mixed`: returns Frobenius norm of the orthogonality center tensor.
///
/// Otherwise computes the full inner product.
pub(super) fn norm_dense<T, B>(psi: &Mps<DenseStorage<T>, DenseLayout, B>) -> T::Real
where
    T: Scalar,
    B: ComputeBackend,
{
    match psi.canonical_form() {
        CanonicalForm::Left | CanonicalForm::Right => T::Real::one(),
        CanonicalForm::Mixed { center } => psi.site(*center).norm(),
        _ => {
            let overlap = inner_dense(psi, psi);
            overlap.re().sqrt()
        }
    }
}

/// Compute ⟨ψ|A|φ⟩ — the MPO-inserted inner product (bra-ket with operator).
pub(super) fn braket_dense<T, B>(
    psi: &Mps<DenseStorage<T>, DenseLayout, B>,
    op: &Mpo<DenseStorage<T>, DenseLayout, B>,
    phi: &Mps<DenseStorage<T>, DenseLayout, B>,
) -> T
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert_eq!(n, op.len(), "MPO length must match MPS length");
    assert!(n > 0, "must have at least one site");

    let backend_arc = Arc::clone(psi.backend_arc());
    let order = backend_arc.preferred_order();

    // Environment: (χ_ψ, χ_A, χ_φ), starts as 1×1×1.
    let env_td = DenseTensorData::from_raw_parts(vec![T::one()], vec![1, 1, 1], order);
    let mut env: DenseTensor<T, B> =
        Tensor::<DenseStorage<T>, DenseLayout, B>::with_backend(env_td, Arc::clone(&backend_arc));

    for j in 0..n {
        let psi_j = dense_conj(psi.site(j)); // bra: (ψ_L, d_bra, ψ_R)
        let a_j = op.site(j); // operator: (A_L, d_ket, d_bra, A_R)
        let phi_j = phi.site(j); // ket: (φ_L, d_ket, φ_R)

        // env(a,b,c) × conj(ψ)(a,d,e) → temp1(b,c,d,e)
        let temp1 =
            contract(&env, &psi_j, "abc,ade->bcde").expect("braket contraction step 1 failed");

        // temp1(b,c,d,e) × A(b,f,d,g) → temp2(c,e,f,g)
        let temp2 =
            contract(&temp1, a_j, "bcde,bfdg->cefg").expect("braket contraction step 2 failed");

        // temp2(c,e,f,g) × φ(c,f,h) → env_new(e,g,h)
        env = contract(&temp2, phi_j, "cefg,cfh->egh").expect("braket contraction step 3 failed");
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
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    phi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
) -> T
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert!(n > 0, "MPS must have at least one site");

    let backend_arc = Arc::clone(psi.backend_arc());

    let mut env = {
        let psi_left = &psi.site(0).data().layout().indices()[0];
        let phi_left = &phi.site(0).data().layout().indices()[0];
        let env_leg0 = QNIndex::new(psi_left.blocks().to_vec(), psi_left.direction());
        let phi_dir_flipped = match phi_left.direction() {
            Direction::Out => Direction::In,
            Direction::In => Direction::Out,
        };
        let env_leg1 = QNIndex::new(phi_left.blocks().to_vec(), phi_dir_flipped);
        let mut e = BlockSparseTensor::<T, S, B>::zeros_with_backend(
            vec![env_leg0, env_leg1],
            S::identity(),
            Arc::clone(&backend_arc),
        );
        if let Some(d) = e.data_mut().block_data_mut(&BlockCoord(vec![0, 0])) {
            d[0] = T::one();
        }
        e
    };

    for j in 0..n {
        let bra_j = bsp_dagger(psi.site(j));
        let phi_j = phi.site(j);

        let step1 = match contract_block_sparse(&env, &bra_j, &[0], &[0])
            .expect("inner product step 1 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 1 always produces a tensor (rank >= 2)")
            }
        };

        env = match contract_block_sparse(&step1, phi_j, &[0, 1], &[0, 1])
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
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    phi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
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

    let backend_arc = Arc::clone(psi.backend_arc());

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
        let mut e = BlockSparseTensor::<T, S, B>::zeros_with_backend(
            vec![env_leg0, env_leg1, env_leg2],
            S::identity(),
            Arc::clone(&backend_arc),
        );
        if let Some(d) = e.data_mut().block_data_mut(&BlockCoord(vec![0, 0, 0])) {
            d[0] = T::one();
        }
        e
    };

    for j in 0..n {
        let bra_j = bsp_dagger(psi.site(j));
        let a_j = op.site(j);
        let phi_j = phi.site(j);

        let step1 = match contract_block_sparse(&env, &bra_j, &[0], &[0])
            .expect("braket step 1 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 1 always produces a tensor (rank >= 3)")
            }
        };

        let step2 = match contract_block_sparse(&step1, a_j, &[0, 2], &[0, 2])
            .expect("braket step 2 contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("step 2 always produces a tensor (rank >= 2)")
            }
        };

        env = match contract_block_sparse(&step2, phi_j, &[0, 2], &[0, 1])
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
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
) -> T::Real
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    match psi.canonical_form() {
        CanonicalForm::Left | CanonicalForm::Right => T::Real::one(),
        CanonicalForm::Mixed { center } => bsp_tensor_norm(psi.site(*center)),
        _ => {
            let overlap = inner_bsp(psi, psi);
            overlap.re().sqrt()
        }
    }
}

/// Frobenius norm of a block-sparse tensor (sum of squared abs values).
fn bsp_tensor_norm<T, S, B>(t: &BlockSparseTensor<T, S, B>) -> T::Real
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let mut sq = T::Real::zero();
    for &x in t.data().storage().data() {
        let a = x.abs();
        sq = sq + a * a;
    }
    <T::Real as Float>::sqrt(sq)
}
