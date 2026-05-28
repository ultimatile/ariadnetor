//! MPO-MPS application: apply an MPO to an MPS via the streaming-naive
//! algorithm.

use std::num::NonZeroUsize;
use std::sync::Arc;

use arnet::{
    BlockSparseContractResult, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor,
    ComputeBackend, DenseLayout, DenseStorage, DenseTensor, Direction, MemoryOrder, Scalar, Sector,
    TruncSvdParams, contract, contract_block_sparse, diagonal_scale, diagonal_scale_block_sparse,
    fuse_legs_block_sparse, permute_block_sparse, qr, qr_block_sparse, trunc_svd,
    trunc_svd_block_sparse,
};

use super::chain::TensorChain;
use super::types::{CanonicalForm, Mpo, Mps, TruncateParams};

/// Apply a Dense MPO to a Dense MPS via the streaming-naive algorithm.
///
/// For each site `j`, contracts the MPO tensor (rank-4) with the MPS tensor
/// (rank-3) over the physical index, fuses the resulting bonds, absorbs the
/// carry factor from site `j-1`, then emits the new left factor of site `j`
/// via QR (lossless) or truncated SVD (when `forward_cap` is `Some(k)` and
/// the natural rank exceeds `k * chi_max`). The carry passes to site `j+1`.
///
/// After the forward sweep, if `params` is `Some`, the result is finished by
/// `canonicalize_dense` + `truncate_dense`. The forward sweep keeps the peak
/// per-site bond bounded by the QR ranks rather than the fully inflated
/// `w_R * chi_R` product, while delegating canonicalization and `chi_max`
/// truncation to the standard pipeline.
///
/// `forward_cap = None` is lossless streaming naive: the forward branch is
/// always QR, and the final state matches a materialize-then-compress
/// baseline modulo QR sign and floating-point roundoff.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths, if either is empty,
/// or via the underlying `canonicalize_dense` / `truncate_dense` for
/// out-of-range `params.center`.
pub(super) fn apply_streaming_naive_dense<T, B>(
    op: &Mpo<DenseStorage<T>, DenseLayout, B>,
    psi: &Mps<DenseStorage<T>, DenseLayout, B>,
    params: Option<&TruncateParams>,
    forward_cap: Option<NonZeroUsize>,
) -> Mps<DenseStorage<T>, DenseLayout, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let backend_arc = Arc::clone(psi.backend_arc());
    let order = backend_arc.preferred_order();
    let rm = MemoryOrder::RowMajor;

    // The forward branch falls to truncated SVD only when both a user
    // chi_max and a forward_cap factor are set; otherwise the per-site QR
    // branch runs unconditionally (lossless streaming naive).
    let chi_max_forward: Option<usize> = match (params.and_then(|p| p.svd.chi_max), forward_cap) {
        (Some(chi), Some(factor)) => Some(chi.saturating_mul(factor.get())),
        _ => None,
    };
    let cutoff = params.and_then(|p| p.svd.target_trunc_err);

    let mut tensors: Vec<DenseTensor<T, B>> = Vec::with_capacity(n);
    let mut carry: Option<DenseTensor<T, B>> = None;

    for j in 0..n {
        let w = op.site(j);
        let a = psi.site(j);
        let local = contract(w, a, "abcd,ebf->aecdf")
            .expect("MPO-MPS contraction: validated by entry point");
        let s = local.shape();
        let (w_l, chi_l, d_bra, w_r, chi_r) = (s[0], s[1], s[2], s[3], s[4]);
        let local_rm = local.reordered(rm);
        let fused_rm = local_rm.reshape(vec![w_l * chi_l, d_bra, w_r * chi_r]);
        let mut p = fused_rm.reordered(order);

        if let Some(c) = carry.as_ref() {
            p = contract(c, &p, "ab,bcd->acd").expect("carry absorption: validated by entry point");
        }

        if j < n - 1 {
            let p_shape = p.shape().to_vec();
            let (left, d, right) = (p_shape[0], p_shape[1], p_shape[2]);
            let rank = (left * d).min(right);

            let use_svd = match chi_max_forward {
                Some(cap) => rank > cap,
                None => false,
            };

            if !use_svd {
                let (q, r) = qr(&p, 2).expect("QR: validated by entry point");
                let k = q.shape()[1];
                let q_rm = q.reordered(rm);
                let q_multi_rm = q_rm.reshape(vec![left, d, k]);
                let q_site = q_multi_rm.reordered(order);
                tensors.push(q_site);
                carry = Some(r);
            } else {
                let svd_params = TruncSvdParams {
                    chi_max: chi_max_forward,
                    target_trunc_err: cutoff,
                };
                let (u, s_vec, vt, _err) =
                    trunc_svd(&p, 2, &svd_params).expect("trunc_svd: validated by entry point");
                let k = u.shape()[1];
                let u_rm = u.reordered(rm);
                let u_multi_rm = u_rm.reshape(vec![left, d, k]);
                let u_site = u_multi_rm.reordered(order);
                tensors.push(u_site);

                let svt = diagonal_scale(&vt, s_vec.data_slice(), 0)
                    .expect("S·Vt scaling: validated by entry point");
                carry = Some(svt);
            }
        } else {
            tensors.push(p);
        }
    }

    let mut result_mps: Mps<DenseStorage<T>, DenseLayout, B> =
        Mps::<DenseStorage<T>, DenseLayout, B>::with_backend(tensors, backend_arc);
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });

    // Delegate final canonicalization + truncation to the standard pipeline.
    // This reuses the three-sweep gauge pattern in `truncate_dense` and
    // honors every `SvdAbsorb` variant and any in-range `params.center`.
    if let Some(trunc_params) = params {
        let center = trunc_params.center.unwrap_or(0);
        super::canonicalize::canonicalize_dense(&mut result_mps, center);
        super::truncate::truncate_dense(&mut result_mps, trunc_params);
    }

    result_mps
}

/// Local single-site MPO·MPS product for the BlockSparse path.
fn local_product_bsp<T, S, B>(
    w: &BlockSparseTensor<T, S, B>,
    a: &BlockSparseTensor<T, S, B>,
) -> BlockSparseTensor<T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    // Contract over physical index (d_ket = W axis 1, d = A axis 1):
    // Output: [w_L, d_bra, w_R, χ_L, χ_R]
    let contracted = match contract_block_sparse(w, a, &[1], &[1])
        .expect("MPO-MPS contraction: validated by entry point")
    {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => {
            unreachable!("MPO-MPS contraction always produces a tensor")
        }
    };

    // Permute to [w_L, χ_L, d_bra, w_R, χ_R].
    let permuted = permute_block_sparse(&contracted, &[0, 3, 1, 2, 4])
        .expect("permute: validated by entry point");

    // Fuse left bond: (w_L, χ_L) → single Out leg.
    let fused_left = fuse_legs_block_sparse(&permuted, 0, 2, Direction::Out)
        .expect("left bond fusion: validated by entry point");

    // Fuse right bond: (w_R, χ_R) → single In leg.
    fuse_legs_block_sparse(&fused_left, 2, 2, Direction::In)
        .expect("right bond fusion: validated by entry point")
}

/// Apply a BlockSparse MPO to a BlockSparse MPS via the streaming-naive
/// algorithm.
///
/// See [`apply_streaming_naive_dense`] for the algorithm description; the
/// BlockSparse variant mirrors it via [`local_product_bsp`],
/// `qr_block_sparse`, and `trunc_svd_block_sparse`, then delegates the
/// final canonicalization + truncation to `canonicalize_bsp` +
/// `truncate_bsp`.
pub(super) fn apply_streaming_naive_bsp<T, S, B>(
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    params: Option<&TruncateParams>,
    forward_cap: Option<NonZeroUsize>,
) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let backend_arc = Arc::clone(psi.backend_arc());

    let chi_max_forward: Option<usize> = match (params.and_then(|p| p.svd.chi_max), forward_cap) {
        (Some(chi), Some(factor)) => Some(chi.saturating_mul(factor.get())),
        _ => None,
    };
    let cutoff = params.and_then(|p| p.svd.target_trunc_err);

    let mut tensors: Vec<BlockSparseTensor<T, S, B>> = Vec::with_capacity(n);
    let mut carry: Option<BlockSparseTensor<T, S, B>> = None;

    for j in 0..n {
        let mut p = local_product_bsp(op.site(j), psi.site(j));

        if let Some(c) = carry.as_ref() {
            p = match contract_block_sparse(c, &p, &[1], &[0])
                .expect("carry absorption: validated by entry point")
            {
                BlockSparseContractResult::Tensor(t) => t,
                BlockSparseContractResult::Scalar(_) => {
                    unreachable!("carry absorption is a rank-3 result")
                }
            };
        }

        if j < n - 1 {
            let rank = forward_rank_estimate_bsp(&p);
            let use_svd = match chi_max_forward {
                Some(cap) => rank > cap,
                None => false,
            };

            if !use_svd {
                let (q, r) = qr_block_sparse(&p, 2).expect("QR: validated by entry point");
                tensors.push(q);
                carry = Some(r);
            } else {
                let svd_params = TruncSvdParams {
                    chi_max: chi_max_forward,
                    target_trunc_err: cutoff,
                };
                let (u, s_vec, vt, _err) = trunc_svd_block_sparse(&p, 2, &svd_params)
                    .expect("trunc_svd: validated by entry point");
                tensors.push(u);

                let svt = diagonal_scale_block_sparse(&vt, &s_vec, 0)
                    .expect("S·Vt scaling: validated by entry point");
                carry = Some(svt);
            }
        } else {
            tensors.push(p);
        }
    }

    let mut result_mps: Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B> =
        Mps::<BlockSparseStorage<T>, BlockSparseLayout<S>, B>::with_backend(tensors, backend_arc);
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });

    if let Some(trunc_params) = params {
        let center = trunc_params.center.unwrap_or(0);
        super::canonicalize::canonicalize_bsp(&mut result_mps, center);
        super::truncate::truncate_bsp(&mut result_mps, trunc_params);
    }

    result_mps
}

/// Conservative natural-rank estimate for the QR / SVD switch.
fn forward_rank_estimate_bsp<T, S, B>(p: &BlockSparseTensor<T, S, B>) -> usize
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let shape = p.shape();
    let left_d = shape[0].saturating_mul(shape[1]);
    let right = shape[2];
    left_d.min(right)
}
