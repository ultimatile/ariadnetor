//! MPO-MPS application: apply an MPO to an MPS

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::{
    BlockSparseContractResult, TruncSvdParams, contract, contract_block_sparse, diagonal_scale,
    fuse_legs_block_sparse, permute_block_sparse, qr, trunc_svd,
};
use arnet_tensor::{BlockSparse, Dense, Direction, Sector, reorder};

use super::chain::TensorChain;
use super::types::{CanonicalForm, Mpo, Mps, TruncateParams};

// Forward pass switches from QR (lossless, fast) to truncated SVD when the
// natural rank exceeds this multiple of the user-supplied chi_max. The
// expanded rank cap absorbs the bulk of the singular weight forward while
// leaving the final, exact truncation to the backward sweep.
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

/// Apply an MPO to an MPS via the zip-up algorithm (interleaved
/// contraction + compression).
///
/// Two-pass scheme:
/// - Forward (left → right): contract each `W[j]` and `A[j]` with the carry
///   from site `j-1`, then split off a left-isometric site tensor by QR.
///   When a `chi_max` budget is given and the natural rank would exceed
///   `ZIPUP_SVD_RATIO * chi_max`, the QR is replaced by a truncated SVD
///   with that expanded cap to bound the intermediate bond dimension while
///   leaving most of the truncation to the final backward sweep.
/// - Backward (right → left): if `params` is `Some`, run a truncated SVD
///   sweep that distributes singular values leftward, producing right-
///   isometric tensors at sites `1..N` and the orthogonality center at site
///   `0` (`Mixed { center: 0 }`).
///
/// When `params` is `None` the forward QR pass alone is run; the result
/// already has left-isometric tensors at sites `0..N-1` and the
/// orthogonality center at site `N-1` (`Mixed { center: N-1 }`).
///
/// Unlike [`apply_dense`], zip-up never materializes the inflated bond
/// dimension `w * χ` simultaneously across all sites; each per-site
/// reduction is local. The cost is that per-cut SVDs are taken before the
/// right environment is fully resolved, so for the same `chi_max` the
/// truncation error is generally larger than the naive
/// "exact-product-then-truncate" path.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths or either is empty.
pub(super) fn apply_zipup_dense<T, B>(
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
    let order = backend.preferred_order();
    let rm = arnet_core::MemoryOrder::RowMajor;

    let chi_max_forward = params
        .and_then(|p| p.svd.chi_max)
        .map(|c| c.saturating_mul(ZIPUP_SVD_RATIO));
    let cutoff = params.and_then(|p| p.svd.target_trunc_err);

    let mut tensors: Vec<Dense<T>> = Vec::with_capacity(n);
    let mut carry: Option<Dense<T>> = None;

    for j in 0..n {
        // Local MPO-MPS product, fused to (w_L * χ_L, d_bra, w_R * χ_R).
        let w = op.storage(j);
        let a = psi.storage(j);
        let local = contract(backend, w, a, "abcd,ebf->aecdf")
            .expect("MPO-MPS contraction failed in apply_zipup");
        let s = local.shape();
        let (w_l, chi_l, d_bra, w_r, chi_r) = (s[0], s[1], s[2], s[3], s[4]);
        let local_rm = reorder(&local, order, rm);
        let fused_rm = local_rm.reshape(vec![w_l * chi_l, d_bra, w_r * chi_r]);
        let mut p = reorder(&fused_rm, rm, order);

        if let Some(c) = carry.as_ref() {
            // carry: (k_prev, w_L * χ_L); p: (w_L * χ_L, d_bra, w_R * χ_R).
            p = contract(backend, c, &p, "ab,bcd->acd")
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
                let (q, r) = qr(backend, &p, 2).expect("QR failed in apply_zipup forward");
                let k = q.shape()[1];
                let q_rm = reorder(&q, order, rm);
                let q_multi_rm = q_rm.reshape(vec![left, d, k]);
                let q_site = reorder(&q_multi_rm, rm, order);
                tensors.push(q_site);
                carry = Some(r);
            } else {
                let svd_params = TruncSvdParams {
                    chi_max: chi_max_forward,
                    target_trunc_err: cutoff,
                };
                let (u, s_vec, vt, _err) = trunc_svd(backend, &p, 2, &svd_params)
                    .expect("trunc_svd failed in apply_zipup forward");
                let k = u.shape()[1];
                let u_rm = reorder(&u, order, rm);
                let u_multi_rm = u_rm.reshape(vec![left, d, k]);
                let u_site = reorder(&u_multi_rm, rm, order);
                tensors.push(u_site);

                let svt = diagonal_scale(backend, &vt, s_vec.data(), 0)
                    .expect("S·Vt scaling failed in apply_zipup forward");
                carry = Some(svt);
            }
        } else {
            tensors.push(p);
        }
    }

    let Some(trunc_params) = params else {
        let mut result_mps = Mps::with_backend(tensors, Arc::clone(psi.backend_arc()));
        result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });
        return result_mps;
    };

    // Backward pass: right-to-left truncated SVD sweep, parking S leftward.
    let svd_params = trunc_params.svd.clone();
    for j in (1..n).rev() {
        let p_shape = tensors[j].shape().to_vec();
        let (_left, d, right) = (p_shape[0], p_shape[1], p_shape[2]);

        let (u, s_vec, vt, _err) = trunc_svd(backend, &tensors[j], 1, &svd_params)
            .expect("trunc_svd failed in apply_zipup backward");
        let chi = vt.shape()[0];

        let vt_rm = reorder(&vt, order, rm);
        let vt_multi_rm = vt_rm.reshape(vec![chi, d, right]);
        tensors[j] = reorder(&vt_multi_rm, rm, order);

        let us = diagonal_scale(backend, &u, s_vec.data(), 1)
            .expect("U·S scaling failed in apply_zipup backward");
        let new_prev = contract(backend, &tensors[j - 1], &us, "abc,cd->abd")
            .expect("US absorption failed in apply_zipup backward");
        tensors[j - 1] = new_prev;
    }

    let mut result_mps = Mps::with_backend(tensors, Arc::clone(psi.backend_arc()));
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: 0 });
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
