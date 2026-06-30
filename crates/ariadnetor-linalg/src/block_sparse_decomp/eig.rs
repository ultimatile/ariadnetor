//! Block-sparse general (non-Hermitian) eigenvalue decomposition via the fused
//! sector method.
//!
//! A quantum-number-conserving operator is block-diagonal across fused sectors —
//! one square dense block per sector. This runs a dense `eig` per fused sector
//! and reassembles the right eigenvectors into a block-sparse tensor,
//! structurally identical to how SVD builds its `U` factor and to the
//! self-adjoint [`eigh`](super::eigh).
//!
//! # Preconditions
//!
//! The structural preconditions match `eigh`: identity flux and a symmetric
//! (QN-square) fused-sector universe. A non-identity flux pairs each left fused
//! sector with a *different* right sector (`s_l.dual().fuse(flux)`), so every
//! per-sector block is off-diagonal and generally rectangular — there is no
//! eigendecomposition; and the universe must be square so no sector is silently
//! dropped from the spectrum.
//!
//! Unlike `eigh`, a general eigendecomposition makes no Hermiticity assumption:
//! it trusts only per-sector squareness, mirroring dense `eig`, which requires a
//! square matrix and is otherwise general. The eigenvalues and eigenvectors are
//! complex even for a real operand (`T::Complex`), and carry no canonical order:
//! within each sector, eigenvector column `i` corresponds to eigenvalue `i` in
//! the order the dense kernel returns.

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{ComputeBackend, ExecPolicy};
use ariadnetor_tensor::{BlockSparseTensorData, DenseTensorData, Sector};

use crate::eigen::eig_with_policy_dense;
use crate::error::LinalgError;

use super::fused_sector::{
    assemble_sector_matrix, build_left_tensor, compute_fused_sector_groups,
    validate_square_universe,
};
use super::{BlockScalars, BlockSparseEigResultBsp, to_vec_in_order, validate_nrow};

/// Internal kernel for the block-sparse general eigenvalue decomposition on
/// joined-form [`BlockSparseTensorData<T, S>`]. The public entry point is
/// [`crate::eig_block_sparse_with_backend`].
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if `nrow` is out of range, the flux
/// is non-identity, or the fused-sector universe is asymmetric (an unmatched or
/// dimension-mismatched sector).
pub(crate) fn eig_block_sparse_with_policy_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<BlockSparseEigResultBsp<T, S>, LinalgError> {
    let rank = tensor.layout().rank();
    validate_nrow(rank, nrow)?;

    let flux = tensor.layout().flux();
    if *flux != S::identity() {
        return Err(LinalgError::InvalidArgument(format!(
            "eig requires identity flux (a QN-conserving operator is flux-neutral), got {flux:?}"
        )));
    }

    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);
    validate_square_universe(tensor, &groups, nrow, "eig")?;

    let mut v_matrices = Vec::with_capacity(groups.len());
    let mut w_values = Vec::with_capacity(groups.len());
    let mut k_per_sector = Vec::with_capacity(groups.len());

    for group in &groups {
        let matrix = assemble_sector_matrix(tensor, group, order);
        let dense = DenseTensorData::from_raw_parts(matrix, vec![group.m, group.n], order);
        // Per-sector dense eig: complex eigenvalues in the kernel's order, right
        // eigenvectors as columns in `order`.
        let (w, v) = eig_with_policy_dense(backend, &dense, 1, policy)?;
        k_per_sector.push(group.n);
        v_matrices.push(to_vec_in_order(&v, order));
        w_values.push((group.sector.clone(), w.data().to_vec()));
    }

    // The eigenvector tensor mirrors SVD's U: bond direction `In`, identity
    // flux, full per-sector rank `k = n`. It is built at `T::Complex`, since the
    // per-sector eigenvectors are complex; `build_left_tensor` is generic over
    // the scalar type and its other inputs are scalar-type-independent.
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
