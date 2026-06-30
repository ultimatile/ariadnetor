//! Block-sparse linear solve and matrix inverse via the fused sector method.
//!
//! A quantum-number-conserving operator with identity flux is block-diagonal
//! across fused sectors — one square dense block per sector. With identity
//! flux the allowed blocks of the operand satisfy `fused(left) == fused(right)`,
//! so distinct sectors occupy disjoint row and column ranges and the operator
//! is the block diagonal `A = diag(A_q)`. The linear system `A X = B` then
//! decouples per fused row-sector into `A_q X_q = B_q`, and the inverse is
//! `A^{-1} = diag(A_q^{-1})`: a dense per-sector solve / inverse reassembled
//! per sector is exact, not an approximation. `B`'s flux confines, for each
//! row-sector `q`, the non-zero columns of `B_q` to a single right-sector, so
//! solving over the compressed columns equals the full solve restricted to
//! that sector.
//!
//! # Leg-mirroring precondition
//!
//! Unlike the matrix exponential — which returns a tensor with the operand's
//! own legs regardless of their pairing — `solve` returns `X` with `B`'s index
//! structure and `inverse` returns the operand's index structure (matching the
//! dense convention, where `solve` returns `b.shape()` and `inverse` the input
//! shape). For that output to be re-contractable — `A` contracted with `X`
//! reproduces `B` — `A`'s column legs must be the duals of its row legs. So
//! both ops require a **leg-mirrored square operator**: `rank == 2 * nrow` and
//! each column leg the dual (same blocks, opposite direction) of the
//! corresponding row leg. This is strictly stronger than the
//! `validate_square_universe` check the `expm` / `eigh` / `eig` paths use
//! (which permits an asymmetric leg split); mirroring is what makes `A`'s
//! column tuples (fusing to `q.dual()`) the same block-index tuples, in the
//! same order, as `A`'s / `B`'s row tuples (fusing to `q`), so scattering `X_q`
//! (whose rows index `A`'s column space) back through `B`'s row coordinates is
//! correct. Mirroring is validated, not trusted: a non-mirrored operator would
//! silently mislabel the output's legs and break the downstream contraction.

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::ComputeBackend;
use ariadnetor_tensor::{BlockSparseTensorData, DenseTensorData, QNIndex, Sector};

use crate::block_sparse_decomp::fused_sector::{
    assemble_sector_matrix, build_square_tensor, compute_fused_sector_groups,
};
use crate::block_sparse_decomp::{to_vec_in_order, validate_nrow};
use crate::error::LinalgError;
use crate::solve::{inverse_dense, solve_dense};

/// True when `right` is the dual of `left`: the same `(sector, dim)` blocks but
/// the opposite leg direction. A dual right leg makes a row tuple fusing to `q`
/// share its block indices with the mirror column tuple fusing to `q.dual()`.
fn is_dual_leg<S: Sector>(left: &QNIndex<S>, right: &QNIndex<S>) -> bool {
    left.direction() != right.direction() && left.blocks() == right.blocks()
}

/// True when two legs are identical: the same blocks and the same direction.
/// Used for the shared free row legs of `A` and `B` in `solve` (unlike a
/// contracted pair, which requires opposite directions).
fn is_same_leg<S: Sector>(a: &QNIndex<S>, b: &QNIndex<S>) -> bool {
    a.direction() == b.direction() && a.blocks() == b.blocks()
}

/// Validate that `tensor` is a leg-mirrored square operator at `nrow`: the leg
/// count is symmetric (`rank == 2 * nrow`) and each column leg is the dual of
/// the corresponding row leg. `op` names the operation in rejection messages.
fn validate_leg_mirrored<T, S: Sector>(
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    op: &str,
) -> Result<(), LinalgError> {
    let indices = tensor.layout().indices();
    let rank = indices.len();
    if rank != 2 * nrow {
        return Err(LinalgError::InvalidArgument(format!(
            "{op} requires a leg-mirrored square operator: rank {rank} must equal 2*nrow={}",
            2 * nrow
        )));
    }
    for i in 0..nrow {
        if !is_dual_leg(&indices[i], &indices[nrow + i]) {
            return Err(LinalgError::InvalidArgument(format!(
                "{op} requires a leg-mirrored square operator: column leg {} is not the dual of row leg {i}",
                nrow + i
            )));
        }
    }
    Ok(())
}

/// Reject a non-identity flux on the operator.
fn validate_identity_flux<T, S: Sector>(
    tensor: &BlockSparseTensorData<T, S>,
    op: &str,
) -> Result<(), LinalgError> {
    let flux = tensor.layout().flux();
    if *flux != S::identity() {
        return Err(LinalgError::InvalidArgument(format!(
            "{op} requires identity flux (a QN-conserving operator is flux-neutral), got {flux:?}"
        )));
    }
    Ok(())
}

/// Internal kernel for the block-sparse linear solve on joined-form
/// [`BlockSparseTensorData<T, S>`]. The public entry point is
/// [`crate::solve_block_sparse_with_backend`].
///
/// Solves `A X = B` per fused sector and returns `X` with `B`'s index
/// structure and flux. `A` is the operator (receiver), `B` the right-hand side;
/// `nrow_a` splits `A`'s legs into row / column groups.
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if `nrow_a` is out of range for `A`
/// or `B`, `A`'s flux is non-identity, `A` is not a leg-mirrored square
/// operator, or `B`'s row legs do not match `A`'s.
pub(crate) fn solve_block_sparse_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    a: &BlockSparseTensorData<T, S>,
    b: &BlockSparseTensorData<T, S>,
    nrow_a: usize,
) -> Result<BlockSparseTensorData<T, S>, LinalgError> {
    let a_indices = a.layout().indices();
    validate_nrow(a_indices.len(), nrow_a)?;
    validate_identity_flux(a, "solve")?;
    validate_leg_mirrored(a, nrow_a, "solve")?;

    // Validate B's rank before slicing its legs, so a too-low-rank B errors
    // rather than panicking on the slice. B must have at least one column leg
    // (vector RHS is unsupported), which `validate_nrow`'s `nrow < rank` gives.
    let b_indices = b.layout().indices();
    validate_nrow(b_indices.len(), nrow_a)?;
    for i in 0..nrow_a {
        if !is_same_leg(&a_indices[i], &b_indices[i]) {
            return Err(LinalgError::InvalidArgument(format!(
                "solve requires B's row legs to match A's row legs (shared free legs): leg {i} differs"
            )));
        }
    }

    let order = backend.preferred_order();
    let groups_a = compute_fused_sector_groups(a, nrow_a);
    let groups_b = compute_fused_sector_groups(b, nrow_a);

    // Iterate B's groups (they define the result's sector structure) and pair
    // each with A's group of the same fused sector. Under the validated
    // preconditions every B sector has an A partner: A is mirrored (covers all
    // its left fused sectors) and B's row legs equal A's, so B's left fused
    // sectors are a subset of A's.
    let mut x_matrices = Vec::with_capacity(groups_b.len());
    for b_group in &groups_b {
        let a_group = groups_a
            .iter()
            .find(|g| g.sector() == b_group.sector())
            .expect("internal error: B fused sector has no matching A sector");
        let m = a_group.m;
        let nrhs = b_group.n;
        let a_q = assemble_sector_matrix(a, a_group, order);
        let b_q = assemble_sector_matrix(b, b_group, order);
        let a_dense = DenseTensorData::from_raw_parts(a_q, vec![m, m], order);
        let b_dense = DenseTensorData::from_raw_parts(b_q, vec![m, nrhs], order);
        let x_q = solve_dense(backend, &a_dense, &b_dense, 1)?;
        x_matrices.push(to_vec_in_order(&x_q, order));
    }

    Ok(build_square_tensor(
        &groups_b,
        &x_matrices,
        b_indices,
        b.layout().flux().clone(),
        order,
    ))
}

/// Internal kernel for the block-sparse matrix inverse on joined-form
/// [`BlockSparseTensorData<T, S>`]. The public entry point is
/// [`crate::inverse_block_sparse_with_backend`].
///
/// Returns `A^{-1}` with the operand's legs and identity flux: a per-sector
/// dense inverse reassembled into a same-shape operator, the same skeleton as
/// the matrix exponential but with the leg-mirroring precondition that makes
/// the result a re-contractable inverse.
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if `nrow` is out of range, the flux
/// is non-identity, or the operand is not a leg-mirrored square operator.
pub(crate) fn inverse_block_sparse_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
) -> Result<BlockSparseTensorData<T, S>, LinalgError> {
    let rank = tensor.layout().rank();
    validate_nrow(rank, nrow)?;
    validate_identity_flux(tensor, "inverse")?;
    validate_leg_mirrored(tensor, nrow, "inverse")?;

    let order = backend.preferred_order();
    let groups = compute_fused_sector_groups(tensor, nrow);

    let mut result_matrices = Vec::with_capacity(groups.len());
    for group in &groups {
        let matrix = assemble_sector_matrix(tensor, group, order);
        let dense = DenseTensorData::from_raw_parts(matrix, vec![group.m, group.n], order);
        let inv = inverse_dense(backend, &dense, 1)?;
        result_matrices.push(to_vec_in_order(&inv, order));
    }

    Ok(build_square_tensor(
        &groups,
        &result_matrices,
        tensor.layout().indices(),
        S::identity(),
        order,
    ))
}

#[cfg(test)]
mod tests;
