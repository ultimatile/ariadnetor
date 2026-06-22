//! Block-sparse matrix exponential via the fused sector method.
//!
//! A quantum-number-conserving operator is block-diagonal across fused sectors —
//! one square dense block per sector. With identity flux, the allowed blocks of
//! the operand satisfy `fused(left) == fused(right)`, and each block tuple
//! belongs to a single fused sector, so distinct sectors occupy disjoint row and
//! column ranges: the operator is exactly the block diagonal `A = diag(A_q)`.
//! Since a block-diagonal matrix's powers stay block-diagonal per sector
//! (`A^k = diag(A_q^k)`), the exponential is `exp(A) = diag(exp(A_q))`, so a
//! dense per-sector exponential reassembled per sector is exact, not an
//! approximation.
//!
//! Unlike the decompositions (SVD / QR / LQ / eigh / eig), which introduce a new
//! bond, the exponential returns a tensor with the **same legs and identity
//! flux** as the operand: each power, linear combination, and Padé solve
//! preserves identity flux, so `exp(A)` does too. The per-sector results are
//! scattered back into the operand's block coordinates by
//! [`build_square_tensor`]. The output may be denser than the operand, since a
//! per-sector exponential is generally full even where the operand had absent
//! blocks.
//!
//! # Preconditions
//!
//! The structural preconditions match `eigh` / `eig`: identity flux and a
//! QN-square fused-sector universe. A non-identity flux pairs each left fused
//! sector with a *different* right sector (`s_l.dual().fuse(flux)`), so every
//! per-sector block is off-diagonal and generally rectangular — there is no
//! square operator to exponentiate; and the universe must be square so each
//! per-sector block is square. Per-sector Hermiticity / anti-Hermiticity (for
//! the specialized variants) is trusted exactly as dense `expm_hermitian` and
//! block-sparse `eigh` trust it.

use std::any::TypeId;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_tensor::{BlockSparseTensorData, DenseTensorData, Sector};

use crate::block_sparse_decomp::fused_sector::{
    assemble_sector_matrix, build_square_tensor, compute_fused_sector_groups,
    validate_square_universe,
};
use crate::block_sparse_decomp::{to_vec_in_order, validate_nrow};
use crate::error::LinalgError;
use crate::expm::{expm_antihermitian_dense, expm_dense, expm_hermitian_dense};

/// Run a per-sector dense exponential and reassemble into a same-shape tensor.
///
/// Validates the square-operator preconditions (`nrow` range, identity flux,
/// QN-square universe), then for each fused sector assembles the dense block,
/// applies `exp`, and scatters the result back. `exp` is the dense kernel for
/// the chosen variant; `op` names the operation in rejection messages.
fn expm_per_sector<T, S, B, F>(
    backend: &B,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    op: &str,
    exp: F,
) -> Result<BlockSparseTensorData<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    F: Fn(&DenseTensorData<T>) -> Result<DenseTensorData<T>, LinalgError>,
{
    let rank = tensor.layout().rank();
    validate_nrow(rank, nrow)?;

    let flux = tensor.layout().flux();
    if *flux != S::identity() {
        return Err(LinalgError::InvalidArgument(format!(
            "{op} requires identity flux (a QN-conserving operator is flux-neutral), got {flux:?}"
        )));
    }

    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);
    validate_square_universe(tensor, &groups, nrow, op)?;

    let mut result_matrices = Vec::with_capacity(groups.len());
    for group in &groups {
        let matrix = assemble_sector_matrix(tensor, group, order);
        let dense = DenseTensorData::from_raw_parts(matrix, vec![group.m, group.n], order);
        let result = exp(&dense)?;
        result_matrices.push(to_vec_in_order(&result, order));
    }

    Ok(build_square_tensor(
        &groups,
        &result_matrices,
        tensor.layout().indices(),
        S::identity(),
        order,
    ))
}

/// Internal kernel for the block-sparse general matrix exponential on
/// joined-form [`BlockSparseTensorData<T, S>`]. The public entry point is
/// [`crate::expm_block_sparse_with_backend`].
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if `nrow` is out of range, the flux
/// is non-identity, or the fused-sector universe is asymmetric.
pub(crate) fn expm_block_sparse_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
) -> Result<BlockSparseTensorData<T, S>, LinalgError> {
    expm_per_sector(backend, tensor, nrow, "expm", |dense| {
        expm_dense(backend, dense, 1)
    })
}

/// Internal kernel for the block-sparse Hermitian matrix exponential on
/// joined-form [`BlockSparseTensorData<T, S>`]. The public entry point is
/// [`crate::expm_hermitian_block_sparse_with_backend`].
///
/// Per-sector Hermiticity is trusted, mirroring dense `expm_hermitian`.
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if `nrow` is out of range, the flux
/// is non-identity, or the fused-sector universe is asymmetric.
pub(crate) fn expm_hermitian_block_sparse_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
) -> Result<BlockSparseTensorData<T, S>, LinalgError> {
    expm_per_sector(backend, tensor, nrow, "expm_hermitian", |dense| {
        expm_hermitian_dense(backend, dense, 1)
    })
}

/// Internal kernel for the block-sparse anti-Hermitian matrix exponential on
/// joined-form [`BlockSparseTensorData<T, S>`]. The public entry point is
/// [`crate::expm_antihermitian_block_sparse_with_backend`].
///
/// Per-sector anti-Hermiticity is trusted, mirroring dense
/// `expm_antihermitian`. A real element type is rejected up front — before the
/// sector loop — so an empty real operand errors consistently rather than
/// succeeding by vacuously skipping every per-sector check.
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if the element type is real, `nrow`
/// is out of range, the flux is non-identity, or the fused-sector universe is
/// asymmetric.
pub(crate) fn expm_antihermitian_block_sparse_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
) -> Result<BlockSparseTensorData<T, S>, LinalgError> {
    let tid = TypeId::of::<T>();
    if tid == TypeId::of::<f64>() || tid == TypeId::of::<f32>() {
        return Err(LinalgError::InvalidArgument(
            "expm_antihermitian requires complex input type (Complex<f64> or Complex<f32>)".into(),
        ));
    }

    expm_per_sector(backend, tensor, nrow, "expm_antihermitian", |dense| {
        expm_antihermitian_dense(backend, dense, 1)
    })
}

#[cfg(test)]
mod tests;
