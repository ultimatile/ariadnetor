//! Explicit-backend operation paths for block-sparse tensors.
//!
//! Call-site-backend counterparts for block-sparse operations: the backend is
//! supplied as `&B` and the tensor's own backend is never consulted (tensors
//! no longer carry one).
//!
//! Block-sparse kernels read the per-sector packed buffer under the backend's
//! preferred order and have no internal reorder step. The layout-order
//! invariant is therefore enforced with the release-active
//! [`check_bsp_data_layout_order_matches`] against the supplied backend,
//! returning [`LinalgError`] on mismatch.
//!
//! The backend is bound by
//! [`OpsFor<BlockSparseStorage<T>>`](arnet_tensor::OpsFor): this public surface
//! is the capability gate, so a backend that has not declared it operates on
//! block-sparse storage cannot be passed here. Internal kernels stay
//! `ComputeBackend`-bound; they are reachable only through this gate.

use arnet_core::Scalar;
use arnet_core::backend::ExecPolicy;
use arnet_tensor::{BlockSparseStorage, BlockSparseTensor, Direction, OpsFor, Sector};

use crate::block_sparse_contract::{
    BlockSparseContractResult, BlockSparseContractResultBsp,
    contract_block_sparse_with_policy_dense,
};
use crate::block_sparse_decomp::{
    BlockScalars, BlockSparseEigResult, BlockSparseEighResult, eig_block_sparse_with_policy_dense,
    eigh_block_sparse_with_policy_dense,
};
use crate::block_sparse_fuse::fuse_legs_block_sparse_dense;
use crate::block_sparse_permute::permute_block_sparse_dense;
use crate::block_sparse_scale::diagonal_scale_block_sparse_dense;
use crate::block_sparse_trace::trace_block_sparse_dense;
use crate::error::LinalgError;
use crate::tensor_bridge::check_bsp_data_layout_order_matches;

#[cfg(test)]
mod tests;

/// Block-sparse contraction over the given axis pairs, using the supplied backend.
///
/// The layout-order invariant is checked against the supplied backend for both
/// operands before the per-sector GEMMs.
pub fn contract_block_sparse_with_backend<
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
>(
    backend: &B,
    lhs: &BlockSparseTensor<T, S>,
    rhs: &BlockSparseTensor<T, S>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
) -> Result<BlockSparseContractResult<T, S>, LinalgError> {
    check_bsp_data_layout_order_matches(lhs.data(), backend, "contract_block_sparse: lhs")?;
    check_bsp_data_layout_order_matches(rhs.data(), backend, "contract_block_sparse: rhs")?;
    let result = contract_block_sparse_with_policy_dense(
        backend,
        lhs.data(),
        rhs.data(),
        axes_lhs,
        axes_rhs,
        ExecPolicy::Sequential,
    )?;
    match result {
        BlockSparseContractResultBsp::Tensor(t) => Ok(BlockSparseContractResult::Tensor(
            BlockSparseTensor::from_data(t),
        )),
        BlockSparseContractResultBsp::Scalar(s) => Ok(BlockSparseContractResult::Scalar(s)),
    }
}

/// Block-sparse axis permutation, using the supplied backend.
pub fn permute_block_sparse_with_backend<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    perm: &[usize],
) -> Result<BlockSparseTensor<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    check_bsp_data_layout_order_matches(tensor.data(), backend, "permute_block_sparse")?;
    let result = permute_block_sparse_dense(backend, tensor.data(), perm)?;
    Ok(BlockSparseTensor::from_data(result))
}

/// Block-sparse consecutive leg fusion, using the supplied backend.
pub fn fuse_legs_block_sparse_with_backend<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    start: usize,
    count: usize,
    fused_direction: Direction,
) -> Result<BlockSparseTensor<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    check_bsp_data_layout_order_matches(tensor.data(), backend, "fuse_legs_block_sparse")?;
    let result =
        fuse_legs_block_sparse_dense(backend, tensor.data(), start, count, fused_direction)?;
    Ok(BlockSparseTensor::from_data(result))
}

/// Block-sparse partial trace over the given bond index pairs, using the
/// supplied backend.
///
/// Each pair ties two mutually-dual legs (identical sector blocks, opposite
/// directions) on their diagonal; the result keeps the non-paired legs in
/// their original order and the input flux. The layout-order invariant is
/// checked against the supplied backend before the per-block traces.
pub fn trace_block_sparse_with_backend<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    pairs: &[(usize, usize)],
) -> Result<BlockSparseTensor<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    check_bsp_data_layout_order_matches(tensor.data(), backend, "trace_block_sparse")?;
    let result = trace_block_sparse_dense(backend, tensor.data(), pairs)?;
    Ok(BlockSparseTensor::from_data(result))
}

/// Block-sparse self-adjoint eigenvalue decomposition, using the supplied
/// backend.
///
/// The operand must be a QN-square Hermitian-structured operator: identity flux
/// and a symmetric fused-sector universe (every fused sector paired with its
/// dual at equal dimension). Returns per-sector real eigenvalues, ascending
/// within each sector, and the eigenvector tensor (legs
/// `[row_legs..., bond(In)]`, identity flux). The layout-order invariant is
/// checked against the supplied backend before the per-sector decompositions.
pub fn eigh_block_sparse_with_backend<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    nrow: usize,
) -> Result<BlockSparseEighResult<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    check_bsp_data_layout_order_matches(tensor.data(), backend, "eigh_block_sparse")?;
    let (w, v) =
        eigh_block_sparse_with_policy_dense(backend, tensor.data(), nrow, ExecPolicy::Sequential)?;
    Ok((w, BlockSparseTensor::from_data(v)))
}

/// Block-sparse eigenvalues-only self-adjoint decomposition, using the supplied
/// backend.
///
/// Counterpart of [`eigh_block_sparse_with_backend`] that discards the
/// eigenvectors, returning only the per-sector eigenvalues.
pub fn eigvalsh_block_sparse_with_backend<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    nrow: usize,
) -> Result<BlockScalars<T::Real, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    let (w, _v) = eigh_block_sparse_with_backend(backend, tensor, nrow)?;
    Ok(w)
}

/// Block-sparse general (non-Hermitian) eigenvalue decomposition, using the
/// supplied backend.
///
/// The operand must be a QN-square operator: identity flux and a symmetric
/// fused-sector universe (every fused sector paired with its dual at equal
/// dimension). Unlike [`eigh_block_sparse_with_backend`] it makes no Hermiticity
/// assumption, and returns complex per-sector eigenvalues (in the dense kernel's
/// order, no canonical sort) and the complex eigenvector tensor (legs
/// `[row_legs..., bond(In)]`, identity flux). The layout-order invariant is
/// checked against the supplied backend before the per-sector decompositions.
pub fn eig_block_sparse_with_backend<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    nrow: usize,
) -> Result<BlockSparseEigResult<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    check_bsp_data_layout_order_matches(tensor.data(), backend, "eig_block_sparse")?;
    let (w, v) =
        eig_block_sparse_with_policy_dense(backend, tensor.data(), nrow, ExecPolicy::Sequential)?;
    Ok((w, BlockSparseTensor::from_data(v)))
}

/// Block-sparse eigenvalues-only general decomposition, using the supplied
/// backend.
///
/// Counterpart of [`eig_block_sparse_with_backend`] that discards the
/// eigenvectors, returning only the per-sector complex eigenvalues.
pub fn eigvals_block_sparse_with_backend<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    nrow: usize,
) -> Result<BlockScalars<T::Complex, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    let (w, _v) = eig_block_sparse_with_backend(backend, tensor, nrow)?;
    Ok(w)
}

/// Block-sparse per-sector diagonal scaling, using the supplied backend.
pub fn diagonal_scale_block_sparse_with_backend<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    weights: &BlockScalars<T::Real, S>,
    axis: usize,
) -> Result<BlockSparseTensor<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    check_bsp_data_layout_order_matches(tensor.data(), backend, "diagonal_scale_block_sparse")?;
    let result = diagonal_scale_block_sparse_dense(backend, tensor.data(), weights, axis)?;
    Ok(BlockSparseTensor::from_data(result))
}
