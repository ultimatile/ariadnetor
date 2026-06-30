//! Block-sparse self-adjoint eigenvalue decomposition via the fused sector method.
//!
//! A quantum-number-conserving Hermitian operator is block-diagonal across
//! fused sectors — one square Hermitian dense block per sector. This runs a
//! dense `eigh` per fused sector and reassembles the eigenvectors into a
//! block-sparse tensor, structurally identical to how SVD builds its `U` factor.
//!
//! # Preconditions
//!
//! - `flux == identity`. A non-identity flux pairs each left fused sector with
//!   a *different* right sector (`s_l.dual().fuse(flux)`), so every per-sector
//!   block is off-diagonal and generally rectangular — there is no
//!   eigendecomposition.
//! - The fused-sector universe is symmetric: every left fused sector has its
//!   dual among the right fused sectors with equal total dimension, and vice
//!   versa. Without this an unmatched sector is silently absent from the matched
//!   groups, so the result would carry a spectrum for only part of the sector
//!   space. This is the block-sparse analog of dense squareness; dense `eigh`
//!   cannot drop rows/columns because it operates on a single matrix.
//!
//! Per-sector Hermiticity itself is trusted, matching dense `eigh` (which checks
//! squareness and trusts the caller's Hermiticity). The structural form of that
//! trust here is that the row legs are the duals of the column legs: when they
//! mirror, the row and column bases enumerate in correspondence and each
//! assembled per-sector block is Hermitian. A non-mirrored operator is not
//! self-adjoint and is trusted away, exactly as dense `eigh` trusts a square
//! matrix to be Hermitian.

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{ComputeBackend, ExecPolicy};
use ariadnetor_tensor::{BlockSparseTensorData, DenseTensorData, Sector};

use crate::eigen::eigh_with_policy_dense;
use crate::error::LinalgError;

use super::fused_sector::{
    assemble_sector_matrix, build_left_tensor, compute_fused_sector_groups,
    validate_square_universe,
};
use super::{BlockScalars, BlockSparseEighResultBsp, to_vec_in_order, validate_nrow};

/// Internal kernel for the block-sparse self-adjoint eigenvalue decomposition
/// on joined-form [`BlockSparseTensorData<T, S>`]. The public entry point is
/// [`crate::eigh_block_sparse_with_backend`].
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if `nrow` is out of range, the flux
/// is non-identity, or the fused-sector universe is asymmetric (an unmatched or
/// dimension-mismatched sector).
pub(crate) fn eigh_block_sparse_with_policy_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<BlockSparseEighResultBsp<T, S>, LinalgError> {
    let rank = tensor.layout().rank();
    validate_nrow(rank, nrow)?;

    let flux = tensor.layout().flux();
    if *flux != S::identity() {
        return Err(LinalgError::InvalidArgument(format!(
            "eigh requires identity flux (a self-adjoint operator is flux-neutral), got {flux:?}"
        )));
    }

    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);
    validate_square_universe(tensor, &groups, nrow, "eigh")?;

    let mut v_matrices = Vec::with_capacity(groups.len());
    let mut w_values = Vec::with_capacity(groups.len());
    let mut k_per_sector = Vec::with_capacity(groups.len());

    for group in &groups {
        let matrix = assemble_sector_matrix(tensor, group, order);
        let dense = DenseTensorData::from_raw_parts(matrix, vec![group.m, group.n], order);
        // Per-sector dense eigh: eigenvalues real and ascending, eigenvectors
        // as columns in `order`.
        let (w, v) = eigh_with_policy_dense(backend, &dense, 1, policy)?;
        k_per_sector.push(group.n);
        v_matrices.push(to_vec_in_order(&v, order));
        w_values.push((group.sector.clone(), w.data().to_vec()));
    }

    // The eigenvector tensor mirrors SVD's U: bond direction `In`, identity
    // flux, full per-sector rank `k = n`.
    let v_tensor = build_left_tensor(
        &groups,
        &v_matrices,
        &k_per_sector,
        tensor.layout().indices(),
        nrow,
        order,
    );

    Ok((BlockScalars { values: w_values }, v_tensor))
}

#[cfg(test)]
mod tests;
