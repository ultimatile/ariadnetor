//! MPO-MPS application: apply an MPO to an MPS via the streaming-naive,
//! Stoudenmire-White zip-up, density-matrix, variational-fitting, or
//! successive-randomized-compression algorithm. The density-matrix kernels
//! live in the [`density_matrix`] submodule and reuse this module's
//! per-site `local_product_*` helpers; the variational and randomized
//! kernels live in their own submodules.

mod density_matrix;
mod successive_randomized;
mod variational;

pub(crate) use density_matrix::{apply_density_matrix_bsp, apply_density_matrix_dense};
pub(crate) use successive_randomized::apply_successive_randomized_dense;
pub(crate) use variational::{apply_variational_bsp, apply_variational_dense};

use std::num::NonZeroUsize;

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{
    TruncSvdParams, contract, diagonal_scale, fuse_legs_block_sparse_with_backend,
    permute_block_sparse_with_backend, qr, tensordot, trunc_svd,
};
use ariadnetor_tensor::{
    BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, DenseLayout, DenseStorage,
    DenseTensor, Direction, OpsFor, Sector,
};

use super::chain::TensorChain;
use super::types::{ApplyError, CanonicalForm, Mpo, Mps, TruncateParams};

/// Return the result-boundary error when `t` contains a non-finite
/// element (NaN/inf in either component), attributing it to `site`.
///
/// The detector is an elementwise scan, not a norm fold: a norm
/// conflates genuine poison with a finite tensor whose Frobenius norm
/// merely overflows the scalar's real type, and the latter must not be
/// rejected. The error's diagnostic still carries the (necessarily
/// non-finite) norm, computed only on the failure path and cast lossily
/// so the error type stays non-generic. Shared across apply methods so
/// a future fallible method reuses the same boundary predicate.
pub(crate) fn check_finite<T: Scalar>(site: usize, t: &DenseTensor<T>) -> Result<(), ApplyError> {
    use num_traits::{Float, ToPrimitive};
    if t.data_slice()
        .iter()
        .all(|x| x.re().is_finite() && x.im().is_finite())
    {
        Ok(())
    } else {
        Err(ApplyError::NonFinite {
            site,
            norm: t.norm().to_f64().unwrap_or(f64::NAN),
        })
    }
}

/// Local single-site MPO·MPS product for the Dense path: contract the MPO
/// (rank-4) with the MPS (rank-3) over the physical index and fuse the two
/// boundary pairs, yielding `(w_l*chi_l, d_bra, w_r*chi_r)`. Shared by the
/// streaming-naive and zip-up forward sweeps (mirrors [`local_product_bsp`]).
fn local_product_dense<T, B>(w: &DenseTensor<T>, a: &DenseTensor<T>, backend: &B) -> DenseTensor<T>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    let local = contract(backend, w, a, "abcd,ebf->aecdf")
        .expect("MPO-MPS contraction: validated by entry point");
    // Fuse the (w_l, chi_l) and (w_r, chi_r) boundary pairs, keeping the
    // physical bra leg: (w_l*chi_l, d_bra, w_r*chi_r). A two-group regrouping
    // goes through reshape_logical in a single round-trip; the single-leg
    // fuse_legs / split_leg cannot express it in one op.
    let s = local.shape();
    let (w_l, chi_l, d_bra, w_r, chi_r) = (s[0], s[1], s[2], s[3], s[4]);
    local.reshape_logical(vec![w_l * chi_l, d_bra, w_r * chi_r])
}

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
    backend: &B,
    op: &Mpo<DenseStorage<T>, DenseLayout>,
    psi: &Mps<DenseStorage<T>, DenseLayout>,
    params: Option<&TruncateParams>,
    forward_cap: Option<NonZeroUsize>,
) -> Mps<DenseStorage<T>, DenseLayout>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    // The forward branch falls to truncated SVD only when both a user
    // chi_max and a forward_cap factor are set; otherwise the per-site QR
    // branch runs unconditionally (lossless streaming naive).
    let chi_max_forward: Option<usize> = match (params.and_then(|p| p.svd.chi_max), forward_cap) {
        (Some(chi), Some(factor)) => Some(chi.saturating_mul(factor.get())),
        _ => None,
    };
    let cutoff = params.and_then(|p| p.svd.target_trunc_err);

    let mut tensors: Vec<DenseTensor<T>> = Vec::with_capacity(n);
    let mut carry: Option<DenseTensor<T>> = None;

    for j in 0..n {
        let mut p = local_product_dense(op.site(j), psi.site(j), backend);

        if let Some(c) = carry.as_ref() {
            p = contract(backend, c, &p, "ab,bcd->acd")
                .expect("carry absorption: validated by entry point");
        }

        if j < n - 1 {
            let s = p.shape();
            let (left, d, right) = (s[0], s[1], s[2]);
            let rank = (left * d).min(right);

            let use_svd = match chi_max_forward {
                Some(cap) => rank > cap,
                None => false,
            };

            if !use_svd {
                let (q, r) = qr(backend, &p, 2).expect("QR: validated by entry point");
                // Split Q's fused row leg back into (left, d, k).
                let q_site = q.split_leg(0, &[left, d]);
                tensors.push(q_site);
                carry = Some(r);
            } else {
                let svd_params = TruncSvdParams {
                    chi_max: chi_max_forward,
                    target_trunc_err: cutoff,
                };
                let (u, s_vec, vt, _err) = trunc_svd(backend, &p, 2, &svd_params)
                    .expect("trunc_svd: validated by entry point");
                // Split U's fused row leg back into (left, d, k).
                let u_site = u.split_leg(0, &[left, d]);
                tensors.push(u_site);

                let svt = diagonal_scale(backend, &vt, s_vec.data_slice(), 0)
                    .expect("S·Vt scaling: validated by entry point");
                carry = Some(svt);
            }
        } else {
            tensors.push(p);
        }
    }

    let mut result_mps: Mps<DenseStorage<T>, DenseLayout> = Mps::from_sites(tensors);
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });
    finish_dense(backend, &mut result_mps, params);
    result_mps
}

/// Shared optional finishing pass: delegate final canonicalization +
/// truncation to the standard pipeline. This reuses the three-sweep gauge
/// pattern in `truncate_dense` and honors every `SvdAbsorb` variant and any
/// in-range `params.center`. Used by the streaming-naive and
/// successive-randomized methods; zip-up and density-matrix carry their
/// truncation inside the sweep instead.
fn finish_dense<T, B>(
    backend: &B,
    result: &mut Mps<DenseStorage<T>, DenseLayout>,
    params: Option<&TruncateParams>,
) where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    if let Some(trunc_params) = params {
        let center = trunc_params.center.unwrap_or(0);
        super::canonicalize::canonicalize_dense(backend, result, center);
        super::truncate::truncate_dense(backend, result, trunc_params);
    }
}

/// Apply a Dense MPO to a Dense MPS via the Stoudenmire-White zip-up
/// algorithm.
///
/// Right-canonicalizes `psi` (orthogonality center at site 0), then runs a
/// single left-to-right sweep. For each non-final site `j`, contracts the
/// MPO tensor with the MPS tensor over the physical index, fuses the
/// boundary bonds, absorbs the carry from site `j-1`, and emits the left
/// factor via a truncated SVD capped directly at `chi_max` (from
/// `params.svd`). The `S·Vt` factor carries to site `j+1`. Because `psi` is
/// right-canonical, each per-site truncation is near-optimal — heuristic
/// rather than variationally optimal, since applying the MPO breaks exact
/// right-isometry; there is no separate backward pass.
///
/// `params = None` (or both `chi_max` and `target_trunc_err` set to `None`)
/// keeps the full SVD rank at every bond, making the sweep lossless: the
/// result then matches the exact MPO-MPS product up to gauge and
/// floating-point roundoff. The result is left in `Mixed { center: n - 1 }`.
/// `params.absorb` and `params.center` are not consulted: zip-up
/// intrinsically carries `S·Vt` rightward and ends with the orthogonality
/// center at the last site.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths, or if either is empty.
pub(super) fn apply_zipup_dense<T, B>(
    backend: &B,
    op: &Mpo<DenseStorage<T>, DenseLayout>,
    psi: &Mps<DenseStorage<T>, DenseLayout>,
    params: Option<&TruncateParams>,
) -> Mps<DenseStorage<T>, DenseLayout>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    // Zip-up requires psi right-canonical so each forward-sweep truncation
    // is near-optimal against the right-canonical remainder.
    let mut psi_rc = psi.clone();
    super::canonicalize::canonicalize_dense(backend, &mut psi_rc, 0);

    let svd_params = TruncSvdParams {
        chi_max: params.and_then(|p| p.svd.chi_max),
        target_trunc_err: params.and_then(|p| p.svd.target_trunc_err),
    };

    let mut tensors: Vec<DenseTensor<T>> = Vec::with_capacity(n);
    let mut carry: Option<DenseTensor<T>> = None;

    for j in 0..n {
        let mut p = local_product_dense(op.site(j), psi_rc.site(j), backend);

        if let Some(c) = carry.as_ref() {
            p = contract(backend, c, &p, "ab,bcd->acd")
                .expect("carry absorption: validated by entry point");
        }

        if j < n - 1 {
            let s = p.shape();
            let (left, d) = (s[0], s[1]);
            let (u, s_vec, vt, _err) = trunc_svd(backend, &p, 2, &svd_params)
                .expect("trunc_svd: validated by entry point");
            // Split U's fused row leg back into (left, d, k).
            let u_site = u.split_leg(0, &[left, d]);
            tensors.push(u_site);

            let svt = diagonal_scale(backend, &vt, s_vec.data_slice(), 0)
                .expect("S·Vt scaling: validated by entry point");
            carry = Some(svt);
        } else {
            tensors.push(p);
        }
    }

    let mut result_mps: Mps<DenseStorage<T>, DenseLayout> = Mps::from_sites(tensors);
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });
    result_mps
}

/// Local single-site MPO·MPS product for the BlockSparse path.
fn local_product_bsp<T, S, B>(
    w: &BlockSparseTensor<T, S>,
    a: &BlockSparseTensor<T, S>,
    backend: &B,
) -> BlockSparseTensor<T, S>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    // Contract over physical index (d_ket = W axis 1, d = A axis 1):
    // Output: [w_L, d_bra, w_R, χ_L, χ_R]
    let contracted = tensordot(backend, w, a, &[1], &[1])
        .expect("MPO-MPS contraction: validated by entry point");

    // Permute to [w_L, χ_L, d_bra, w_R, χ_R].
    let permuted = permute_block_sparse_with_backend(backend, &contracted, &[0, 3, 1, 2, 4])
        .expect("permute: validated by entry point");

    // Fuse left bond: (w_L, χ_L) → single Out leg.
    let fused_left = fuse_legs_block_sparse_with_backend(backend, &permuted, 0, 2, Direction::Out)
        .expect("left bond fusion: validated by entry point");

    // Fuse right bond: (w_R, χ_R) → single In leg.
    fuse_legs_block_sparse_with_backend(backend, &fused_left, 2, 2, Direction::In)
        .expect("right bond fusion: validated by entry point")
}

/// Apply a BlockSparse MPO to a BlockSparse MPS via the streaming-naive
/// algorithm.
///
/// See [`apply_streaming_naive_dense`] for the algorithm description; the
/// BlockSparse variant mirrors it via [`local_product_bsp`],
/// `qr`, and `trunc_svd`,
/// then delegates the final canonicalization + truncation to
/// `canonicalize_bsp` + `truncate_bsp`.
pub(super) fn apply_streaming_naive_bsp<T, S, B>(
    backend: &B,
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    params: Option<&TruncateParams>,
    forward_cap: Option<NonZeroUsize>,
) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let chi_max_forward: Option<usize> = match (params.and_then(|p| p.svd.chi_max), forward_cap) {
        (Some(chi), Some(factor)) => Some(chi.saturating_mul(factor.get())),
        _ => None,
    };
    let cutoff = params.and_then(|p| p.svd.target_trunc_err);

    let mut tensors: Vec<BlockSparseTensor<T, S>> = Vec::with_capacity(n);
    let mut carry: Option<BlockSparseTensor<T, S>> = None;

    for j in 0..n {
        let mut p = local_product_bsp(op.site(j), psi.site(j), backend);

        if let Some(c) = carry.as_ref() {
            p = tensordot(backend, c, &p, &[1], &[0])
                .expect("carry absorption: validated by entry point");
        }

        if j < n - 1 {
            let rank = forward_rank_estimate_bsp(&p);
            let use_svd = match chi_max_forward {
                Some(cap) => rank > cap,
                None => false,
            };

            if !use_svd {
                let (q, r) = qr(backend, &p, 2).expect("QR: validated by entry point");
                tensors.push(q);
                carry = Some(r);
            } else {
                let svd_params = TruncSvdParams {
                    chi_max: chi_max_forward,
                    target_trunc_err: cutoff,
                };
                let (u, s_vec, vt, _err) = trunc_svd(backend, &p, 2, &svd_params)
                    .expect("trunc_svd: validated by entry point");
                tensors.push(u);

                let svt = diagonal_scale(backend, &vt, &s_vec, 0)
                    .expect("S·Vt scaling: validated by entry point");
                carry = Some(svt);
            }
        } else {
            tensors.push(p);
        }
    }

    let mut result_mps: Mps<BlockSparseStorage<T>, BlockSparseLayout<S>> = Mps::from_sites(tensors);
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });

    if let Some(trunc_params) = params {
        let center = trunc_params.center.unwrap_or(0);
        super::canonicalize::canonicalize_bsp(backend, &mut result_mps, center);
        super::truncate::truncate_bsp(backend, &mut result_mps, trunc_params);
    }

    result_mps
}

/// Conservative natural-rank estimate for the QR / SVD switch.
fn forward_rank_estimate_bsp<T, S>(p: &BlockSparseTensor<T, S>) -> usize
where
    T: Scalar,
    S: Sector,
{
    let shape = p.shape();
    let left_d = shape[0].saturating_mul(shape[1]);
    let right = shape[2];
    left_d.min(right)
}

/// Apply a BlockSparse MPO to a BlockSparse MPS via the Stoudenmire-White
/// zip-up algorithm.
///
/// See [`apply_zipup_dense`] for the algorithm description; the BlockSparse
/// variant mirrors it via [`local_product_bsp`], `canonicalize_bsp`,
/// `trunc_svd`, and `diagonal_scale`.
pub(super) fn apply_zipup_bsp<T, S, B>(
    backend: &B,
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    params: Option<&TruncateParams>,
) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let mut psi_rc = psi.clone();
    super::canonicalize::canonicalize_bsp(backend, &mut psi_rc, 0);

    let svd_params = TruncSvdParams {
        chi_max: params.and_then(|p| p.svd.chi_max),
        target_trunc_err: params.and_then(|p| p.svd.target_trunc_err),
    };

    let mut tensors: Vec<BlockSparseTensor<T, S>> = Vec::with_capacity(n);
    let mut carry: Option<BlockSparseTensor<T, S>> = None;

    for j in 0..n {
        let mut p = local_product_bsp(op.site(j), psi_rc.site(j), backend);

        if let Some(c) = carry.as_ref() {
            p = tensordot(backend, c, &p, &[1], &[0])
                .expect("carry absorption: validated by entry point");
        }

        if j < n - 1 {
            let (u, s_vec, vt, _err) = trunc_svd(backend, &p, 2, &svd_params)
                .expect("trunc_svd: validated by entry point");
            tensors.push(u);

            let svt = diagonal_scale(backend, &vt, &s_vec, 0)
                .expect("S·Vt scaling: validated by entry point");
            carry = Some(svt);
        } else {
            tensors.push(p);
        }
    }

    let mut result_mps: Mps<BlockSparseStorage<T>, BlockSparseLayout<S>> = Mps::from_sites(tensors);
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });
    result_mps
}
