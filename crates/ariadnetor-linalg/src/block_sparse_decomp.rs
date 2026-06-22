//! Block-sparse tensor decompositions via fused sector method.
//!
//! Decomposes `BlockSparseTensor<T, S>` tensors using SVD, QR, or LQ by:
//! 1. Fusing left/right leg groups into fused sectors
//! 2. Assembling dense matrices per fused sector pair
//! 3. Running per-sector dense decomposition
//! 4. Reconstructing block-sparse output tensors
//!
//! All decompositions produce a left tensor with `flux = identity()` and a
//! right tensor with `flux = original_flux`. The new bond uses fused left
//! sectors with direction `In` on the left side and `Out` on the right side.

mod eigh;
pub(crate) mod fused_sector;

pub(crate) use eigh::eigh_block_sparse_with_policy_dense;

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder};
use arnet_tensor::{
    BlockSparseTensor, BlockSparseTensorData, DenseTensorData, Sector, reorder_data,
};
use num_traits::{Float, ToPrimitive, Zero};

use crate::decomposition::TruncSvdParams;
use crate::error::LinalgError;
use fused_sector::*;

// Public types ============================================================

/// Per-sector real scalar values keyed by fused sector.
///
/// The block-sparse analog of the dense flat `DenseTensor<T::Real>`: the same
/// container serves any per-sector real spectrum, so both SVD singular values
/// (non-negative, descending) and `eigh` eigenvalues (signed, ascending) use
/// it. The ordering within each sector is fixed by the producing operation,
/// not by this type. Sectors with no values are omitted.
#[derive(Debug)]
pub struct BlockScalars<R, S: Sector> {
    /// (fused sector, value list) pairs, sorted by sector.
    pub values: Vec<(S, Vec<R>)>,
}

impl<R, S: Sector> BlockScalars<R, S> {
    /// Transform each value, preserving the sector structure.
    pub fn map<U, F>(&self, mut f: F) -> BlockScalars<U, S>
    where
        F: FnMut(&R) -> U,
    {
        BlockScalars {
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
    BlockSparseTensor<T, S>,
    BlockScalars<<T as Scalar>::Real, S>,
    BlockSparseTensor<T, S>,
);

/// Internal kernel form of [`BlockSparseSvdResult`] on joined-form [`BlockSparseTensorData<T, S>`].
pub(crate) type BlockSparseSvdResultBsp<T, S> = (
    BlockSparseTensorData<T, S>,
    BlockScalars<<T as Scalar>::Real, S>,
    BlockSparseTensorData<T, S>,
);

/// Result of a truncated block-sparse SVD: `(U, S, Vt, trunc_err)`.
pub type BlockSparseTruncSvdResult<T, S> = (
    BlockSparseTensor<T, S>,
    BlockScalars<<T as Scalar>::Real, S>,
    BlockSparseTensor<T, S>,
    <T as Scalar>::Real,
);

/// Internal kernel form of [`BlockSparseTruncSvdResult`] on joined-form [`BlockSparseTensorData<T, S>`].
pub(crate) type BlockSparseTruncSvdResultBsp<T, S> = (
    BlockSparseTensorData<T, S>,
    BlockScalars<<T as Scalar>::Real, S>,
    BlockSparseTensorData<T, S>,
    <T as Scalar>::Real,
);

/// Result of a block-sparse QR or LQ decomposition.
pub type BlockSparseQrResult<T, S> = (BlockSparseTensor<T, S>, BlockSparseTensor<T, S>);

/// Internal kernel form of [`BlockSparseQrResult`] on joined-form [`BlockSparseTensorData<T, S>`].
pub(crate) type BlockSparseQrResultBsp<T, S> =
    (BlockSparseTensorData<T, S>, BlockSparseTensorData<T, S>);

/// Result of a block-sparse self-adjoint eigenvalue decomposition:
/// `(eigenvalues, eigenvectors)`.
///
/// - eigenvalues: [`BlockScalars`] of real values, ascending within each sector
/// - eigenvectors: [`BlockSparseTensor`] with legs `[original_row_legs..., bond(In)]`
///   and identity flux, the per-sector eigenvector columns (built exactly like
///   the SVD `U` factor)
pub type BlockSparseEighResult<T, S> = (
    BlockScalars<<T as Scalar>::Real, S>,
    BlockSparseTensor<T, S>,
);

/// Internal kernel form of [`BlockSparseEighResult`] on joined-form [`BlockSparseTensorData<T, S>`].
pub(crate) type BlockSparseEighResultBsp<T, S> = (
    BlockScalars<<T as Scalar>::Real, S>,
    BlockSparseTensorData<T, S>,
);

// Public API -- SVD =======================================================

/// Internal kernel for the block-sparse SVD on joined-form
/// [`BlockSparseTensorData<T, S>`]. The public entries dispatch over layout:
/// the auto-policy [`svd`](crate::svd) pins `ExecPolicy::Sequential`, while
/// [`expert::svd`](crate::expert::svd) forwards a caller-specified `policy`.
pub(crate) fn svd_block_sparse_with_policy_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<BlockSparseSvdResultBsp<T, S>, LinalgError> {
    validate_nrow(tensor.layout().rank(), nrow)?;
    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);

    let mut u_matrices = Vec::with_capacity(groups.len());
    let mut s_values = Vec::with_capacity(groups.len());
    let mut vt_matrices = Vec::with_capacity(groups.len());
    let mut k_per_sector = Vec::with_capacity(groups.len());

    for group in &groups {
        let matrix = assemble_sector_matrix(tensor, group, order);
        let dense = DenseTensorData::from_raw_parts(matrix, vec![group.m, group.n], order);
        let (u, s, vt) = crate::decomposition::svd_with_policy_dense(backend, &dense, 1, policy)?;
        k_per_sector.push(group.m.min(group.n));
        u_matrices.push(to_vec_in_order(&u, order));
        s_values.push((group.sector.clone(), s.storage().data().to_vec()));
        vt_matrices.push(to_vec_in_order(&vt, order));
    }

    let u_tensor = build_left_tensor(
        &groups,
        &u_matrices,
        &k_per_sector,
        tensor.layout().indices(),
        nrow,
        order,
    );
    let vt_tensor = build_right_tensor(
        &groups,
        &vt_matrices,
        &k_per_sector,
        tensor.layout().indices(),
        nrow,
        tensor.layout().flux().clone(),
        order,
    );

    Ok((u_tensor, BlockScalars { values: s_values }, vt_tensor))
}

/// Internal kernel for the truncated block-sparse SVD on joined-form
/// [`BlockSparseTensorData<T, S>`]. The public entries dispatch over layout:
/// the auto-policy [`trunc_svd`](crate::trunc_svd) pins `ExecPolicy::Sequential`,
/// while [`expert::trunc_svd`](crate::expert::trunc_svd) forwards a
/// caller-specified `policy`.
pub(crate) fn trunc_svd_block_sparse_with_policy_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    params: &TruncSvdParams,
    policy: ExecPolicy,
) -> Result<BlockSparseTruncSvdResultBsp<T, S>, LinalgError> {
    validate_nrow(tensor.layout().rank(), nrow)?;
    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);

    // Per-sector full SVD
    let mut u_matrices = Vec::with_capacity(groups.len());
    let mut all_s: Vec<Vec<T::Real>> = Vec::with_capacity(groups.len());
    let mut vt_matrices = Vec::with_capacity(groups.len());
    let mut k_full = Vec::with_capacity(groups.len());

    for group in &groups {
        let matrix = assemble_sector_matrix(tensor, group, order);
        let dense = DenseTensorData::from_raw_parts(matrix, vec![group.m, group.n], order);
        let (u, s, vt) = crate::decomposition::svd_with_policy_dense(backend, &dense, 1, policy)?;
        k_full.push(group.m.min(group.n));
        u_matrices.push(to_vec_in_order(&u, order));
        all_s.push(s.storage().data().to_vec());
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
            u_trunc.push(truncate_cols(&u_matrices[gi], m, k_f, k_t, order));
            s_trunc_values.push(all_s[gi][..k_t].to_vec());
            vt_trunc.push(truncate_rows(&vt_matrices[gi], k_f, n, k_t, order));
        }
    }

    let u_tensor = build_left_tensor(
        &groups,
        &u_trunc,
        &k_per_sector,
        tensor.layout().indices(),
        nrow,
        order,
    );
    let vt_tensor = build_right_tensor(
        &groups,
        &vt_trunc,
        &k_per_sector,
        tensor.layout().indices(),
        nrow,
        tensor.layout().flux().clone(),
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
        BlockScalars { values: sv_pairs },
        vt_tensor,
        trunc_err,
    ))
}

// Public API -- QR / LQ ===================================================

/// Internal kernel for the block-sparse QR on joined-form
/// [`BlockSparseTensorData<T, S>`]. The public entries dispatch over layout:
/// the auto-policy [`qr`](crate::qr) pins `ExecPolicy::Sequential`, while
/// [`expert::qr`](crate::expert::qr) forwards a caller-specified `policy`.
pub(crate) fn qr_block_sparse_with_policy_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<BlockSparseQrResultBsp<T, S>, LinalgError> {
    validate_nrow(tensor.layout().rank(), nrow)?;
    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);
    let (q_mats, r_mats, k_per) =
        decompose_per_sector(&groups, tensor, nrow, backend, order, |b, d| {
            crate::decomposition::qr_with_policy_dense(b, d, 1, policy)
                .map(|(q, r)| (to_vec_in_order(&q, order), to_vec_in_order(&r, order)))
        })?;
    let q = build_left_tensor(
        &groups,
        &q_mats,
        &k_per,
        tensor.layout().indices(),
        nrow,
        order,
    );
    let r = build_right_tensor(
        &groups,
        &r_mats,
        &k_per,
        tensor.layout().indices(),
        nrow,
        tensor.layout().flux().clone(),
        order,
    );
    Ok((q, r))
}

/// Internal kernel for the block-sparse LQ on joined-form
/// [`BlockSparseTensorData<T, S>`]. The public entries dispatch over layout:
/// the auto-policy [`lq`](crate::lq) pins `ExecPolicy::Sequential`, while
/// [`expert::lq`](crate::expert::lq) forwards a caller-specified `policy`.
pub(crate) fn lq_block_sparse_with_policy_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<BlockSparseQrResultBsp<T, S>, LinalgError> {
    validate_nrow(tensor.layout().rank(), nrow)?;
    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);
    let (l_mats, q_mats, k_per) =
        decompose_per_sector(&groups, tensor, nrow, backend, order, |b, d| {
            crate::decomposition::lq_with_policy_dense(b, d, 1, policy)
                .map(|(l, q)| (to_vec_in_order(&l, order), to_vec_in_order(&q, order)))
        })?;
    let l = build_left_tensor(
        &groups,
        &l_mats,
        &k_per,
        tensor.layout().indices(),
        nrow,
        order,
    );
    let q = build_right_tensor(
        &groups,
        &q_mats,
        &k_per,
        tensor.layout().indices(),
        nrow,
        tensor.layout().flux().clone(),
        order,
    );
    Ok((l, q))
}

// Internal helpers ========================================================

/// Keep the first `k_t` columns of an `m × k_f` matrix.
fn truncate_cols<T: Scalar>(
    data: &[T],
    m: usize,
    k_f: usize,
    k_t: usize,
    order: MemoryOrder,
) -> Vec<T> {
    match order {
        MemoryOrder::RowMajor => {
            let mut buf = vec![T::zero(); m * k_t];
            for r in 0..m {
                buf[r * k_t..(r + 1) * k_t].copy_from_slice(&data[r * k_f..r * k_f + k_t]);
            }
            buf
        }
        MemoryOrder::ColumnMajor => data[..k_t * m].to_vec(),
    }
}

/// Keep the first `k_t` rows of a `k_f × n` matrix.
fn truncate_rows<T: Scalar>(
    data: &[T],
    k_f: usize,
    n: usize,
    k_t: usize,
    order: MemoryOrder,
) -> Vec<T> {
    match order {
        MemoryOrder::RowMajor => data[..k_t * n].to_vec(),
        MemoryOrder::ColumnMajor => {
            let mut buf = vec![T::zero(); k_t * n];
            for c in 0..n {
                buf[c * k_t..(c + 1) * k_t].copy_from_slice(&data[c * k_f..c * k_f + k_t]);
            }
            buf
        }
    }
}

fn validate_nrow(rank: usize, nrow: usize) -> Result<(), LinalgError> {
    if nrow == 0 || nrow >= rank {
        return Err(LinalgError::InvalidArgument(format!(
            "nrow must satisfy 1 <= nrow < rank, got nrow={nrow} for rank={rank}"
        )));
    }
    Ok(())
}

/// Run a two-output decomposition (QR or LQ) per fused sector.
#[allow(clippy::type_complexity)]
fn decompose_per_sector<T, S, B, F>(
    groups: &[FusedSectorGroup<S>],
    tensor: &BlockSparseTensorData<T, S>,
    _nrow: usize,
    backend: &B,
    order: MemoryOrder,
    decompose: F,
) -> Result<(Vec<Vec<T>>, Vec<Vec<T>>, Vec<usize>), LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    F: Fn(&B, &DenseTensorData<T>) -> Result<(Vec<T>, Vec<T>), LinalgError>,
{
    let mut left_mats = Vec::with_capacity(groups.len());
    let mut right_mats = Vec::with_capacity(groups.len());
    let mut k_per = Vec::with_capacity(groups.len());
    for group in groups {
        let matrix = assemble_sector_matrix(tensor, group, order);
        let dense = DenseTensorData::from_raw_parts(matrix, vec![group.m, group.n], order);
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

/// Extract tensor data in `order`. `reorder_data` is a no-op clone
/// when `tensor.order() == order`, so passing the backend's preferred
/// order short-circuits to the existing buffer.
fn to_vec_in_order<T: Scalar>(tensor: &DenseTensorData<T>, order: MemoryOrder) -> Vec<T> {
    reorder_data(tensor, order).storage().data().to_vec()
}

#[cfg(test)]
mod tests;
