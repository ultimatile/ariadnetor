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

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy};
use arnet_tensor::{BlockSparseTensorData, DenseTensorData, QNIndex, Sector};

use crate::eigen::eigh_with_policy_dense;
use crate::error::LinalgError;

use super::fused_sector::{
    FusedSectorGroup, assemble_sector_matrix, build_left_tensor, compute_fused_sector_groups,
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
    validate_square_universe(tensor, &groups, nrow)?;

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

/// Verify the bipartition forms a QN-square operator: every matched fused
/// sector is square, and the matched sectors together cover the entire row and
/// column space so none is silently dropped.
///
/// [`compute_fused_sector_groups`] keys groups by left fused sector and
/// silently omits any left sector lacking a right partner. Summing each side's
/// matched dimensions and comparing against the full leg-dimension products
/// detects both an unmatched sector (the sum falls short) and a per-sector
/// rectangular block (`m != n`) — directly from the already-computed `groups`,
/// without re-enumerating the fused-sector universe. The hard `m == n` check
/// is what the per-sector dense `eigh` relies on (so no release-stripped
/// `debug_assert` is needed in the loop).
fn validate_square_universe<T: Scalar, S: Sector>(
    tensor: &BlockSparseTensorData<T, S>,
    groups: &[FusedSectorGroup<S>],
    nrow: usize,
) -> Result<(), LinalgError> {
    let indices = tensor.layout().indices();
    let total_left = leg_dim_product(&indices[..nrow]);
    let total_right = leg_dim_product(&indices[nrow..]);

    let mut matched_left = 0usize;
    let mut matched_right = 0usize;
    for group in groups {
        if group.m != group.n {
            return Err(LinalgError::InvalidArgument(format!(
                "eigh requires a square operator: fused sector {:?} has a {}x{} block",
                group.sector, group.m, group.n
            )));
        }
        matched_left += group.m;
        matched_right += group.n;
    }

    if matched_left != total_left || matched_right != total_right {
        return Err(LinalgError::InvalidArgument(format!(
            "eigh requires a square operator: matched fused sectors cover {matched_left}/{total_left} row and {matched_right}/{total_right} column dimensions, so a sector has no matching partner"
        )));
    }

    Ok(())
}

/// Total dimension spanned by a set of legs: the product over legs of each
/// leg's summed block dimensions. Equals the sum of every fused sector's
/// dimension on that side, without enumerating the fused sectors.
fn leg_dim_product<S: Sector>(indices: &[QNIndex<S>]) -> usize {
    indices
        .iter()
        .map(|idx| {
            (0..idx.num_blocks())
                .map(|b| idx.block_dim(b))
                .sum::<usize>()
        })
        .product()
}

#[cfg(test)]
mod tests;
