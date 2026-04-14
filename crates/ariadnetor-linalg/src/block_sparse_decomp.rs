//! Block-sparse tensor decompositions via fused sector method.
//!
//! Decomposes [`BlockSparse<T, S>`] tensors using SVD, QR, or LQ by:
//! 1. Fusing left/right leg groups into fused sectors
//! 2. Assembling dense matrices per fused sector pair
//! 3. Running per-sector dense decomposition
//! 4. Reconstructing BlockSparse output tensors
//!
//! All decompositions produce a left tensor with `flux = identity()` and a
//! right tensor with `flux = original_flux`. The new bond uses fused left
//! sectors with direction `In` on the left side and `Out` on the right side.

mod fused_sector;

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_tensor::BlockSparse;
use arnet_tensor::Dense;
use arnet_tensor::Sector;
use num_traits::{Float, ToPrimitive, Zero};

use crate::decomposition::TruncSvdParams;
use crate::error::LinalgError;
use arnet_tensor::reorder;
use fused_sector::*;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// SVD singular values stored per fused sector.
///
/// Each sector's values are sorted in descending order. Sectors with no
/// retained values are omitted.
#[derive(Debug)]
pub struct BlockSingularValues<R, S: Sector> {
    /// (fused sector, singular values) pairs, sorted by sector.
    pub values: Vec<(S, Vec<R>)>,
}

impl<R, S: Sector> BlockSingularValues<R, S> {
    /// Transform each singular value, preserving the sector structure.
    pub fn map<U, F>(&self, mut f: F) -> BlockSingularValues<U, S>
    where
        F: FnMut(&R) -> U,
    {
        BlockSingularValues {
            values: self
                .values
                .iter()
                .map(|(s, vs)| (s.clone(), vs.iter().map(&mut f).collect()))
                .collect(),
        }
    }
}

/// Result of a block-sparse SVD: `(U, S, Vt)`.
pub type BlockSparseSvdResult<T, S> = (
    BlockSparse<T, S>,
    BlockSingularValues<<T as Scalar>::Real, S>,
    BlockSparse<T, S>,
);

/// Result of a truncated block-sparse SVD: `(U, S, Vt, trunc_err)`.
pub type BlockSparseTruncSvdResult<T, S> = (
    BlockSparse<T, S>,
    BlockSingularValues<<T as Scalar>::Real, S>,
    BlockSparse<T, S>,
    <T as Scalar>::Real,
);

/// Result of a block-sparse QR or LQ decomposition.
pub type BlockSparseQrResult<T, S> = (BlockSparse<T, S>, BlockSparse<T, S>);

// ---------------------------------------------------------------------------
// Public API -- SVD
// ---------------------------------------------------------------------------

/// Thin SVD of a block-sparse tensor via fused sector method.
///
/// Returns `(U, S, Vt)` where:
/// - `U`: legs = `[left_legs..., bond(In)]`, `flux = identity()`
/// - `S`: singular values per fused sector (descending within each sector)
/// - `Vt`: legs = `[bond(Out), right_legs...]`, `flux = original_flux`
pub fn svd_block_sparse<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparse<T, S>,
    nrow: usize,
) -> Result<BlockSparseSvdResult<T, S>, LinalgError> {
    validate_nrow(tensor.rank(), nrow)?;
    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);

    let mut u_matrices = Vec::with_capacity(groups.len());
    let mut s_values = Vec::with_capacity(groups.len());
    let mut vt_matrices = Vec::with_capacity(groups.len());
    let mut k_per_sector = Vec::with_capacity(groups.len());

    for group in &groups {
        let matrix = assemble_sector_matrix(tensor, group, order);
        let dense = Dense::new(matrix, vec![group.m, group.n]);
        let (u, s, vt) = crate::decomposition::svd(backend, &dense, 1)?;
        k_per_sector.push(group.m.min(group.n));
        u_matrices.push(to_vec_in_order(&u, order));
        s_values.push((group.sector.clone(), s.data().to_vec()));
        vt_matrices.push(to_vec_in_order(&vt, order));
    }

    let u_tensor = build_left_tensor(
        &groups,
        &u_matrices,
        &k_per_sector,
        tensor.indices(),
        nrow,
        order,
    );
    let vt_tensor = build_right_tensor(
        &groups,
        &vt_matrices,
        &k_per_sector,
        tensor.indices(),
        nrow,
        tensor.flux().clone(),
        order,
    );

    Ok((
        u_tensor,
        BlockSingularValues { values: s_values },
        vt_tensor,
    ))
}

/// Truncated SVD of a block-sparse tensor via fused sector method.
///
/// Performs full per-sector SVD, then applies cross-sector truncation using
/// `chi_max` and/or `target_trunc_err` from `params`. When both are set,
/// the stricter (smaller) bound applies.
///
/// Returns `(U, S, Vt, trunc_err)` where `trunc_err` is the Frobenius norm
/// of discarded singular values.
pub fn trunc_svd_block_sparse<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparse<T, S>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<BlockSparseTruncSvdResult<T, S>, LinalgError> {
    validate_nrow(tensor.rank(), nrow)?;
    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);

    // Per-sector full SVD
    let mut u_matrices = Vec::with_capacity(groups.len());
    let mut all_s: Vec<Vec<T::Real>> = Vec::with_capacity(groups.len());
    let mut vt_matrices = Vec::with_capacity(groups.len());
    let mut k_full = Vec::with_capacity(groups.len());

    for group in &groups {
        let matrix = assemble_sector_matrix(tensor, group, order);
        let dense = Dense::new(matrix, vec![group.m, group.n]);
        let (u, s, vt) = crate::decomposition::svd(backend, &dense, 1)?;
        k_full.push(group.m.min(group.n));
        u_matrices.push(to_vec_in_order(&u, order));
        all_s.push(s.data().to_vec());
        vt_matrices.push(to_vec_in_order(&vt, order));
    }

    // Global cross-sector truncation
    let (k_per_sector, trunc_err) = cross_sector_truncate::<T>(&all_s, groups.len(), params)?;

    // Truncate per-sector data
    let mut u_trunc = Vec::with_capacity(groups.len());
    let mut s_trunc_values = Vec::with_capacity(groups.len());
    let mut vt_trunc = Vec::with_capacity(groups.len());

    for (gi, group) in groups.iter().enumerate() {
        let k_t = k_per_sector[gi];
        let k_f = k_full[gi];
        if k_t == 0 {
            u_trunc.push(Vec::new());
            s_trunc_values.push(Vec::new());
            vt_trunc.push(Vec::new());
        } else if k_t == k_f {
            u_trunc.push(u_matrices[gi].clone());
            s_trunc_values.push(all_s[gi].clone());
            vt_trunc.push(vt_matrices[gi].clone());
        } else {
            let m = group.m;
            let n = group.n;
            // Truncate U (m x k_f -> m x k_t): keep first k_t columns
            let u_t = match order {
                MemoryOrder::RowMajor => {
                    let mut buf = vec![T::zero(); m * k_t];
                    for r in 0..m {
                        buf[r * k_t..(r + 1) * k_t]
                            .copy_from_slice(&u_matrices[gi][r * k_f..r * k_f + k_t]);
                    }
                    buf
                }
                MemoryOrder::ColumnMajor => {
                    // Columns are contiguous; take first k_t columns (k_t * m elements)
                    u_matrices[gi][..k_t * m].to_vec()
                }
            };
            u_trunc.push(u_t);
            s_trunc_values.push(all_s[gi][..k_t].to_vec());
            // Truncate Vt (k_f x n -> k_t x n): keep first k_t rows
            let vt_t = match order {
                MemoryOrder::RowMajor => {
                    // Rows are contiguous; take first k_t rows (k_t * n elements)
                    vt_matrices[gi][..k_t * n].to_vec()
                }
                MemoryOrder::ColumnMajor => {
                    let mut buf = vec![T::zero(); k_t * n];
                    for c in 0..n {
                        buf[c * k_t..(c + 1) * k_t]
                            .copy_from_slice(&vt_matrices[gi][c * k_f..c * k_f + k_t]);
                    }
                    buf
                }
            };
            vt_trunc.push(vt_t);
        }
    }

    let u_tensor = build_left_tensor(
        &groups,
        &u_trunc,
        &k_per_sector,
        tensor.indices(),
        nrow,
        order,
    );
    let vt_tensor = build_right_tensor(
        &groups,
        &vt_trunc,
        &k_per_sector,
        tensor.indices(),
        nrow,
        tensor.flux().clone(),
        order,
    );

    let sv_pairs: Vec<(S, Vec<T::Real>)> = groups
        .iter()
        .zip(s_trunc_values)
        .filter(|(_, sv)| !sv.is_empty())
        .map(|(g, sv)| (g.sector.clone(), sv))
        .collect();

    Ok((
        u_tensor,
        BlockSingularValues { values: sv_pairs },
        vt_tensor,
        trunc_err,
    ))
}

// ---------------------------------------------------------------------------
// Public API -- QR / LQ
// ---------------------------------------------------------------------------

/// QR decomposition of a block-sparse tensor via fused sector method.
///
/// Returns `(Q, R)` where:
/// - `Q`: legs = `[left_legs..., bond(In)]`, `flux = identity()`
/// - `R`: legs = `[bond(Out), right_legs...]`, `flux = original_flux`
pub fn qr_block_sparse<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparse<T, S>,
    nrow: usize,
) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
    validate_nrow(tensor.rank(), nrow)?;
    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);
    let (q_mats, r_mats, k_per) =
        decompose_per_sector(&groups, tensor, nrow, backend, order, |b, d| {
            crate::decomposition::qr(b, d, 1)
                .map(|(q, r)| (to_vec_in_order(&q, order), to_vec_in_order(&r, order)))
        })?;
    let q = build_left_tensor(&groups, &q_mats, &k_per, tensor.indices(), nrow, order);
    let r = build_right_tensor(
        &groups,
        &r_mats,
        &k_per,
        tensor.indices(),
        nrow,
        tensor.flux().clone(),
        order,
    );
    Ok((q, r))
}

/// LQ decomposition of a block-sparse tensor via fused sector method.
///
/// Returns `(L, Q)` where:
/// - `L`: legs = `[left_legs..., bond(In)]`, `flux = identity()`
/// - `Q`: legs = `[bond(Out), right_legs...]`, `flux = original_flux`
pub fn lq_block_sparse<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparse<T, S>,
    nrow: usize,
) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
    validate_nrow(tensor.rank(), nrow)?;
    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);
    let (l_mats, q_mats, k_per) =
        decompose_per_sector(&groups, tensor, nrow, backend, order, |b, d| {
            crate::decomposition::lq(b, d, 1)
                .map(|(l, q)| (to_vec_in_order(&l, order), to_vec_in_order(&q, order)))
        })?;
    let l = build_left_tensor(&groups, &l_mats, &k_per, tensor.indices(), nrow, order);
    let q = build_right_tensor(
        &groups,
        &q_mats,
        &k_per,
        tensor.indices(),
        nrow,
        tensor.flux().clone(),
        order,
    );
    Ok((l, q))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_nrow(rank: usize, nrow: usize) -> Result<(), LinalgError> {
    if nrow == 0 || nrow >= rank {
        return Err(LinalgError::InvalidArgument(format!(
            "nrow must be in 1..rank, got nrow={nrow} for rank={rank}"
        )));
    }
    Ok(())
}

/// Run a two-output decomposition (QR or LQ) per fused sector.
#[allow(clippy::type_complexity)]
fn decompose_per_sector<T, S, B, F>(
    groups: &[FusedSectorGroup<S>],
    tensor: &BlockSparse<T, S>,
    _nrow: usize,
    backend: &B,
    order: MemoryOrder,
    decompose: F,
) -> Result<(Vec<Vec<T>>, Vec<Vec<T>>, Vec<usize>), LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    F: Fn(&B, &Dense<T>) -> Result<(Vec<T>, Vec<T>), LinalgError>,
{
    let mut left_mats = Vec::with_capacity(groups.len());
    let mut right_mats = Vec::with_capacity(groups.len());
    let mut k_per = Vec::with_capacity(groups.len());
    for group in groups {
        let matrix = assemble_sector_matrix(tensor, group, order);
        let dense = Dense::new(matrix, vec![group.m, group.n]);
        let (l, r) = decompose(backend, &dense)?;
        k_per.push(group.m.min(group.n));
        left_mats.push(l);
        right_mats.push(r);
    }
    Ok((left_mats, right_mats, k_per))
}

/// Global cross-sector truncation of singular values.
///
/// Returns `(k_per_sector, trunc_err)`.
fn cross_sector_truncate<T: Scalar>(
    all_s: &[Vec<T::Real>],
    num_sectors: usize,
    params: &TruncSvdParams,
) -> Result<(Vec<usize>, T::Real), LinalgError> {
    let mut sorted_sv: Vec<(T::Real, usize)> = Vec::new();
    for (si, s_data) in all_s.iter().enumerate() {
        for &sv in s_data {
            sorted_sv.push((sv, si));
        }
    }
    // Deterministic tie-breaking by sector index for equal singular values
    sorted_sv.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let total_sv = sorted_sv.len();
    if total_sv == 0 {
        return Ok((vec![0; num_sectors], T::Real::zero()));
    }
    let mut chi = total_sv;

    if let Some(chi_max) = params.chi_max {
        if chi_max == 0 {
            return Err(LinalgError::InvalidArgument(
                "chi_max must be at least 1".into(),
            ));
        }
        chi = chi.min(chi_max);
    }

    if let Some(target_err) = params.target_trunc_err {
        let target_sq = target_err * target_err;
        let mut discarded_sq = 0.0_f64;
        let mut chi_err = total_sv;
        for i in (0..total_sv).rev() {
            let si_sq = sorted_sv[i].0.to_f64().unwrap().powi(2);
            if discarded_sq + si_sq > target_sq {
                break;
            }
            discarded_sq += si_sq;
            chi_err = i;
        }
        chi = chi.min(chi_err).max(1);
    }

    let mut err_sq = T::Real::zero();
    for &(sv, _) in &sorted_sv[chi..] {
        err_sq = err_sq + sv * sv;
    }

    let mut k_per_sector = vec![0usize; num_sectors];
    for &(_, si) in &sorted_sv[..chi] {
        k_per_sector[si] += 1;
    }

    Ok((k_per_sector, err_sq.sqrt()))
}

/// Extract tensor data in the specified memory order.
fn to_vec_in_order<T: Scalar>(tensor: &Dense<T>, order: MemoryOrder) -> Vec<T> {
    // Dense data is already in the backend's preferred order (which is `order`
    // since callers pass backend.preferred_order()). No reorder needed for
    // the same order; reorder is a no-op (clone) when from == to.
    reorder(tensor, order, order).data().to_vec()
}

#[cfg(test)]
mod tests;
