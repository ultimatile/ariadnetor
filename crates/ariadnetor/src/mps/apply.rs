//! MPO-MPS application: apply an MPO to an MPS

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::{
    BlockSparseContractResult, contract, contract_block_sparse, fuse_legs_block_sparse,
    permute_block_sparse,
};
use arnet_tensor::{BlockSparse, Dense, Direction, Sector, reorder};

use super::chain::TensorChain;
use super::types::{Mpo, Mps, TruncateParams};

/// Apply an MPO to an MPS, producing a new MPS.
///
/// For each site, contracts the MPO tensor (rank-4) with the MPS tensor
/// (rank-3) over the physical index, then fuses the bond dimensions:
///
/// ```text
/// W[w_L, d_ket, d_bra, w_R] × A[χ_L, d_ket, χ_R]
///   → result[w_L*χ_L, d_bra, w_R*χ_R]
/// ```
///
/// If `params` is `Some`, the result is canonicalized and truncated.
/// If `None`, the exact (lossless) result is returned with `Unknown`
/// canonical form.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths, either is empty,
/// or `params.center` is `Some(c)` with `c >= psi.len()`.
pub(super) fn apply_dense<T, B>(
    op: &Mpo<Dense<T>, B>,
    psi: &Mps<Dense<T>, B>,
    params: Option<&TruncateParams>,
) -> Mps<Dense<T>, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let backend = psi.backend();
    let mut storages = Vec::with_capacity(n);

    for j in 0..n {
        let w = op.storage(j); // (w_L, d_ket, d_bra, w_R)
        let a = psi.storage(j); // (χ_L, d, χ_R)

        // Contract over physical index (b = d_ket = d):
        // W(a,b,c,d) × A(e,b,f) → result(a,e,c,d,f)
        // = (w_L, χ_L, d_bra, w_R, χ_R)
        let result =
            contract(backend, w, a, "abcd,ebf->aecdf").expect("MPO-MPS contraction failed");

        let shape = result.shape();
        let w_l = shape[0];
        let chi_l = shape[1];
        let d_bra = shape[2];
        let w_r = shape[3];
        let chi_r = shape[4];

        // Fuse bond dimensions: (w_L*χ_L, d_bra, w_R*χ_R)
        // Reorder to RM for correct axis-merge semantics, reshape, then back.
        let order = backend.preferred_order();
        let rm = arnet_core::MemoryOrder::RowMajor;
        let result_rm = reorder(&result, order, rm);
        let fused_rm = result_rm.reshape(vec![w_l * chi_l, d_bra, w_r * chi_r]);
        let fused = reorder(&fused_rm, rm, order);

        storages.push(fused);
    }

    let mut result_mps = Mps::with_backend(storages, Arc::clone(psi.backend_arc()));

    if let Some(trunc_params) = params {
        let center = trunc_params.center.unwrap_or(0);
        super::canonicalize::canonicalize_dense(&mut result_mps, center);
        super::truncate::truncate_dense(&mut result_mps, trunc_params);
    }

    result_mps
}

/// Apply a BlockSparse MPO to a BlockSparse MPS, producing a new MPS.
///
/// For each site, contracts the MPO tensor (rank-4) with the MPS tensor
/// (rank-3) over the physical index, then fuses bond dimensions:
///
/// ```text
/// W[w_L, d_ket, d_bra, w_R] × A[χ_L, d_ket, χ_R]
///   → contract → [w_L, d_bra, w_R, χ_L, χ_R]
///   → permute  → [w_L, χ_L, d_bra, w_R, χ_R]
///   → fuse     → [w_L⊗χ_L, d_bra, w_R⊗χ_R]
/// ```
///
/// If `params` is `Some`, the result is canonicalized and truncated.
/// If `None`, the exact (lossless) result is returned with `Unknown`
/// canonical form.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths, either is empty,
/// or `params.center` is `Some(c)` with `c >= psi.len()`.
pub(super) fn apply_bsp<T, S, B>(
    op: &Mpo<BlockSparse<T, S>, B>,
    psi: &Mps<BlockSparse<T, S>, B>,
    params: Option<&TruncateParams>,
) -> Mps<BlockSparse<T, S>, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let backend = psi.backend();
    let mut storages = Vec::with_capacity(n);

    for j in 0..n {
        let w = op.storage(j); // (w_L, d_ket, d_bra, w_R)
        let a = psi.storage(j); // (χ_L, d, χ_R)

        // Contract over physical index (d_ket = W axis 1, d = A axis 1):
        // Output: [w_L, d_bra, w_R, χ_L, χ_R] (lhs_free then rhs_free)
        let contracted = match contract_block_sparse(backend, w, a, &[1], &[1])
            .expect("MPO-MPS contraction failed")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("MPO-MPS contraction always produces a tensor")
            }
        };

        // Permute to [w_L, χ_L, d_bra, w_R, χ_R]
        let permuted =
            permute_block_sparse(backend, &contracted, &[0, 3, 1, 2, 4]).expect("permute failed");

        // Fuse left bond: (w_L, χ_L) → single Out leg
        let fused_left = fuse_legs_block_sparse(backend, &permuted, 0, 2, Direction::Out)
            .expect("left bond fusion failed");

        // Fuse right bond: (w_R, χ_R) → single In leg
        let fused = fuse_legs_block_sparse(backend, &fused_left, 2, 2, Direction::In)
            .expect("right bond fusion failed");

        storages.push(fused);
    }

    let mut result_mps = Mps::with_backend(storages, Arc::clone(psi.backend_arc()));

    if let Some(trunc_params) = params {
        let center = trunc_params.center.unwrap_or(0);
        super::canonicalize::canonicalize_bsp(&mut result_mps, center);
        super::truncate::truncate_bsp(&mut result_mps, trunc_params);
    }

    result_mps
}
