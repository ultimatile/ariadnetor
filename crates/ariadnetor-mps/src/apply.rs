//! MPO-MPS application: apply an MPO to an MPS.

use std::sync::Arc;

use arnet::{
    BlockSparseContractResult, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor,
    ComputeBackend, DenseLayout, DenseStorage, DenseTensor, Direction, MemoryOrder, Scalar, Sector,
    TruncSvdParams, contract, contract_block_sparse, diagonal_scale, diagonal_scale_block_sparse,
    fuse_legs_block_sparse, permute_block_sparse, qr, qr_block_sparse, trunc_svd,
    trunc_svd_block_sparse,
};

use super::chain::TensorChain;
use super::internal_helpers::{dense_reshape, reorder_dense_tensor};
use super::types::{CanonicalForm, Mpo, Mps, SvdAbsorb, TruncateParams};

// Forward pass switches from QR (lossless, fast) to truncated SVD when the
// natural rank exceeds this multiple of the user-supplied chi_max.
const ZIPUP_SVD_RATIO: usize = 4;

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
/// If `None`, the exact result is returned with `Unknown` canonical form.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths, either is empty,
/// or `params.center` is `Some(c)` with `c >= psi.len()`.
pub(super) fn apply_dense<T, B>(
    op: &Mpo<DenseStorage<T>, DenseLayout, B>,
    psi: &Mps<DenseStorage<T>, DenseLayout, B>,
    params: Option<&TruncateParams>,
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

    let mut sites: Vec<DenseTensor<T, B>> = Vec::with_capacity(n);

    for j in 0..n {
        let w = op.site(j); // (w_L, d_ket, d_bra, w_R)
        let a = psi.site(j); // (χ_L, d, χ_R)

        // W(a,b,c,d) × A(e,b,f) → result(a,e,c,d,f)
        // = (w_L, χ_L, d_bra, w_R, χ_R)
        let result = contract(w, a, "abcd,ebf->aecdf").expect("MPO-MPS contraction failed");

        let shape = result.shape();
        let w_l = shape[0];
        let chi_l = shape[1];
        let d_bra = shape[2];
        let w_r = shape[3];
        let chi_r = shape[4];

        let result_rm = reorder_dense_tensor(&result, order, rm);
        let fused_rm = dense_reshape(&result_rm, vec![w_l * chi_l, d_bra, w_r * chi_r]);
        let fused = reorder_dense_tensor(&fused_rm, rm, order);

        sites.push(fused);
    }

    let mut result_mps: Mps<DenseStorage<T>, DenseLayout, B> =
        Mps::<DenseStorage<T>, DenseLayout, B>::with_backend(sites, backend_arc);

    if let Some(trunc_params) = params {
        let center = trunc_params.center.unwrap_or(0);
        super::canonicalize::canonicalize_dense(&mut result_mps, center);
        super::truncate::truncate_dense(&mut result_mps, trunc_params);
    }

    result_mps
}

/// Apply an MPO to an MPS via the zip-up algorithm (interleaved
/// contraction + compression).
///
/// # Limitations
///
/// The backward sweep currently honors only [`SvdAbsorb::Right`].
/// `params.center` must be `None` or `Some(0)`. Other values panic
/// up front.
pub(super) fn apply_zipup_dense<T, B>(
    op: &Mpo<DenseStorage<T>, DenseLayout, B>,
    psi: &Mps<DenseStorage<T>, DenseLayout, B>,
    params: Option<&TruncateParams>,
) -> Mps<DenseStorage<T>, DenseLayout, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    if let Some(p) = params {
        assert!(
            matches!(p.absorb, SvdAbsorb::Right),
            "apply_zipup currently supports only SvdAbsorb::Right; got {:?}. \
             Use the naive apply path for Left/Both, or canonicalize the result \
             after the fact.",
            p.absorb,
        );
        assert!(
            matches!(p.center, None | Some(0)),
            "apply_zipup currently parks the orthogonality center at site 0; \
             requested center = {:?}. Call canonicalize on the result to shift \
             the center, or use the naive apply path.",
            p.center,
        );
    }

    let backend_arc = Arc::clone(psi.backend_arc());
    let order = backend_arc.preferred_order();
    let rm = MemoryOrder::RowMajor;

    let chi_max_forward = params
        .and_then(|p| p.svd.chi_max)
        .map(|c| c.saturating_mul(ZIPUP_SVD_RATIO));
    let cutoff = params.and_then(|p| p.svd.target_trunc_err);

    let mut tensors: Vec<DenseTensor<T, B>> = Vec::with_capacity(n);
    let mut carry: Option<DenseTensor<T, B>> = None;

    for j in 0..n {
        let w = op.site(j);
        let a = psi.site(j);
        let local =
            contract(w, a, "abcd,ebf->aecdf").expect("MPO-MPS contraction failed in apply_zipup");
        let s = local.shape();
        let (w_l, chi_l, d_bra, w_r, chi_r) = (s[0], s[1], s[2], s[3], s[4]);
        let local_rm = reorder_dense_tensor(&local, order, rm);
        let fused_rm = dense_reshape(&local_rm, vec![w_l * chi_l, d_bra, w_r * chi_r]);
        let mut p = reorder_dense_tensor(&fused_rm, rm, order);

        if let Some(c) = carry.as_ref() {
            p = contract(c, &p, "ab,bcd->acd")
                .expect("carry absorption failed in apply_zipup forward");
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
                let (q, r) = qr(&p, 2).expect("QR failed in apply_zipup forward");
                let k = q.shape()[1];
                let q_rm = reorder_dense_tensor(&q, order, rm);
                let q_multi_rm = dense_reshape(&q_rm, vec![left, d, k]);
                let q_site = reorder_dense_tensor(&q_multi_rm, rm, order);
                tensors.push(q_site);
                carry = Some(r);
            } else {
                let svd_params = TruncSvdParams {
                    chi_max: chi_max_forward,
                    target_trunc_err: cutoff,
                };
                let (u, s_vec, vt, _err) =
                    trunc_svd(&p, 2, &svd_params).expect("trunc_svd failed in apply_zipup forward");
                let k = u.shape()[1];
                let u_rm = reorder_dense_tensor(&u, order, rm);
                let u_multi_rm = dense_reshape(&u_rm, vec![left, d, k]);
                let u_site = reorder_dense_tensor(&u_multi_rm, rm, order);
                tensors.push(u_site);

                let svt = diagonal_scale(&vt, s_vec.data_slice(), 0)
                    .expect("S·Vt scaling failed in apply_zipup forward");
                carry = Some(svt);
            }
        } else {
            tensors.push(p);
        }
    }

    let Some(trunc_params) = params else {
        let mut result_mps: Mps<DenseStorage<T>, DenseLayout, B> =
            Mps::<DenseStorage<T>, DenseLayout, B>::with_backend(tensors, backend_arc);
        result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });
        return result_mps;
    };

    // Backward pass: right-to-left truncated SVD sweep, parking S leftward.
    let svd_params = trunc_params.svd.clone();
    for j in (1..n).rev() {
        let p_shape = tensors[j].shape().to_vec();
        let (_left, d, right) = (p_shape[0], p_shape[1], p_shape[2]);

        let (u, s_vec, vt, _err) = trunc_svd(&tensors[j], 1, &svd_params)
            .expect("trunc_svd failed in apply_zipup backward");
        let chi = vt.shape()[0];

        let vt_rm = reorder_dense_tensor(&vt, order, rm);
        let vt_multi_rm = dense_reshape(&vt_rm, vec![chi, d, right]);
        tensors[j] = reorder_dense_tensor(&vt_multi_rm, rm, order);

        let us = diagonal_scale(&u, s_vec.data_slice(), 1)
            .expect("U·S scaling failed in apply_zipup backward");
        let new_prev = contract(&tensors[j - 1], &us, "abc,cd->abd")
            .expect("US absorption failed in apply_zipup backward");
        tensors[j - 1] = new_prev;
    }

    let mut result_mps: Mps<DenseStorage<T>, DenseLayout, B> =
        Mps::<DenseStorage<T>, DenseLayout, B>::with_backend(tensors, backend_arc);
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: 0 });
    result_mps
}

/// Apply a BlockSparse MPO to a BlockSparse MPS, producing a new MPS.
pub(super) fn apply_bsp<T, S, B>(
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    params: Option<&TruncateParams>,
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
    let mut sites: Vec<BlockSparseTensor<T, S, B>> = Vec::with_capacity(n);

    for j in 0..n {
        sites.push(local_product_bsp(op.site(j), psi.site(j)));
    }

    let mut result_mps: Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B> =
        Mps::<BlockSparseStorage<T>, BlockSparseLayout<S>, B>::with_backend(sites, backend_arc);

    if let Some(trunc_params) = params {
        let center = trunc_params.center.unwrap_or(0);
        super::canonicalize::canonicalize_bsp(&mut result_mps, center);
        super::truncate::truncate_bsp(&mut result_mps, trunc_params);
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
    let contracted =
        match contract_block_sparse(w, a, &[1], &[1]).expect("MPO-MPS contraction failed") {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("MPO-MPS contraction always produces a tensor")
            }
        };

    // Permute to [w_L, χ_L, d_bra, w_R, χ_R].
    let permuted = permute_block_sparse(&contracted, &[0, 3, 1, 2, 4]).expect("permute failed");

    // Fuse left bond: (w_L, χ_L) → single Out leg.
    let fused_left =
        fuse_legs_block_sparse(&permuted, 0, 2, Direction::Out).expect("left bond fusion failed");

    // Fuse right bond: (w_R, χ_R) → single In leg.
    fuse_legs_block_sparse(&fused_left, 2, 2, Direction::In).expect("right bond fusion failed")
}

/// Apply a BlockSparse MPO to a BlockSparse MPS via the zip-up algorithm.
pub(super) fn apply_zipup_bsp<T, S, B>(
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    params: Option<&TruncateParams>,
) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    if let Some(p) = params {
        assert!(
            matches!(p.absorb, SvdAbsorb::Right),
            "apply_zipup currently supports only SvdAbsorb::Right; got {:?}.",
            p.absorb,
        );
        assert!(
            matches!(p.center, None | Some(0)),
            "apply_zipup currently parks the orthogonality center at site 0; \
             requested center = {:?}.",
            p.center,
        );
    }

    let backend_arc = Arc::clone(psi.backend_arc());

    let chi_max_forward = params
        .and_then(|p| p.svd.chi_max)
        .map(|c| c.saturating_mul(ZIPUP_SVD_RATIO));
    let cutoff = params.and_then(|p| p.svd.target_trunc_err);

    let mut tensors: Vec<BlockSparseTensor<T, S, B>> = Vec::with_capacity(n);
    let mut carry: Option<BlockSparseTensor<T, S, B>> = None;

    for j in 0..n {
        let mut p = local_product_bsp(op.site(j), psi.site(j));

        if let Some(c) = carry.as_ref() {
            p = match contract_block_sparse(c, &p, &[1], &[0])
                .expect("carry absorption failed in apply_zipup_bsp forward")
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
                let (q, r) = qr_block_sparse(&p, 2).expect("QR failed in apply_zipup_bsp forward");
                tensors.push(q);
                carry = Some(r);
            } else {
                let svd_params = TruncSvdParams {
                    chi_max: chi_max_forward,
                    target_trunc_err: cutoff,
                };
                let (u, s_vec, vt, _err) = trunc_svd_block_sparse(&p, 2, &svd_params)
                    .expect("trunc_svd failed in apply_zipup_bsp forward");
                tensors.push(u);

                let svt = diagonal_scale_block_sparse(&vt, &s_vec, 0)
                    .expect("S·Vt scaling failed in apply_zipup_bsp forward");
                carry = Some(svt);
            }
        } else {
            tensors.push(p);
        }
    }

    let Some(trunc_params) = params else {
        let mut result_mps: Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B> =
            Mps::<BlockSparseStorage<T>, BlockSparseLayout<S>, B>::with_backend(
                tensors,
                backend_arc,
            );
        result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });
        return result_mps;
    };

    let svd_params = trunc_params.svd.clone();
    for j in (1..n).rev() {
        let (u, s_vec, vt, _err) = trunc_svd_block_sparse(&tensors[j], 1, &svd_params)
            .expect("trunc_svd failed in apply_zipup_bsp backward");
        tensors[j] = vt;

        let us = diagonal_scale_block_sparse(&u, &s_vec, 1)
            .expect("U·S scaling failed in apply_zipup_bsp backward");

        let prev_last_axis = tensors[j - 1].rank() - 1;
        let new_prev = match contract_block_sparse(&tensors[j - 1], &us, &[prev_last_axis], &[0])
            .expect("US absorption failed in apply_zipup_bsp backward")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("US absorption keeps at least the physical legs free")
            }
        };
        tensors[j - 1] = new_prev;
    }

    let mut result_mps: Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B> =
        Mps::<BlockSparseStorage<T>, BlockSparseLayout<S>, B>::with_backend(tensors, backend_arc);
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: 0 });
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
