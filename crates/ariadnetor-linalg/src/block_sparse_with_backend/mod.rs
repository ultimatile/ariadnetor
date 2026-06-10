//! Explicit-backend operation paths for block-sparse tensors.
//!
//! Call-site-backend counterparts of the legacy tensor-derived block-sparse
//! operations. As with the dense paths, the backend is supplied as `&Arc<B>`
//! and the tensor's own backend is never consulted.
//!
//! The legacy paths guard `layout.order() == backend.preferred_order()` with a
//! `debug_assert!` against the tensor's own (construction-pinned) backend. Here
//! the backend is not pinned to the tensor, so the invariant is enforced with
//! the release-active [`check_bsp_data_layout_order_matches`] against the
//! supplied backend, returning [`LinalgError`] on mismatch.

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy};
use arnet_tensor::{BlockSparseTensor, Direction, Sector};

use crate::block_sparse_contract::{
    BlockSparseContractResult, BlockSparseContractResultBsp,
    contract_block_sparse_with_policy_dense,
};
use crate::block_sparse_decomp::{
    BlockSingularValues, BlockSparseQrResult, BlockSparseSvdResult, BlockSparseTruncSvdResult,
    lq_block_sparse_with_policy_dense, qr_block_sparse_with_policy_dense,
    svd_block_sparse_with_policy_dense, trunc_svd_block_sparse_with_policy_dense,
};
use crate::block_sparse_fuse::fuse_legs_block_sparse_dense;
use crate::block_sparse_permute::permute_block_sparse_dense;
use crate::block_sparse_scale::diagonal_scale_block_sparse_dense;
use crate::decomposition::TruncSvdParams;
use crate::error::LinalgError;
use crate::tensor_bridge::check_bsp_data_layout_order_matches;

#[cfg(test)]
mod tests;

/// Explicit-backend counterpart of [`crate::svd_block_sparse`].
pub fn svd_block_sparse_with_backend<T: Scalar, S: Sector, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &BlockSparseTensor<T, S, B>,
    nrow: usize,
) -> Result<BlockSparseSvdResult<T, S, B>, LinalgError> {
    check_bsp_data_layout_order_matches(tensor.data(), &**backend, "svd_block_sparse")?;
    let (u, s, vt) = svd_block_sparse_with_policy_dense(
        &**backend,
        tensor.data(),
        nrow,
        ExecPolicy::Sequential,
    )?;
    Ok((
        BlockSparseTensor::with_backend(u, backend.clone()),
        s,
        BlockSparseTensor::with_backend(vt, backend.clone()),
    ))
}

/// Explicit-backend counterpart of [`crate::trunc_svd_block_sparse`].
pub fn trunc_svd_block_sparse_with_backend<T: Scalar, S: Sector, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &BlockSparseTensor<T, S, B>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<BlockSparseTruncSvdResult<T, S, B>, LinalgError> {
    check_bsp_data_layout_order_matches(tensor.data(), &**backend, "trunc_svd_block_sparse")?;
    let (u, s, vt, err) = trunc_svd_block_sparse_with_policy_dense(
        &**backend,
        tensor.data(),
        nrow,
        params,
        ExecPolicy::Sequential,
    )?;
    Ok((
        BlockSparseTensor::with_backend(u, backend.clone()),
        s,
        BlockSparseTensor::with_backend(vt, backend.clone()),
        err,
    ))
}

/// Explicit-backend counterpart of [`crate::qr_block_sparse`].
pub fn qr_block_sparse_with_backend<T: Scalar, S: Sector, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &BlockSparseTensor<T, S, B>,
    nrow: usize,
) -> Result<BlockSparseQrResult<T, S, B>, LinalgError> {
    check_bsp_data_layout_order_matches(tensor.data(), &**backend, "qr_block_sparse")?;
    let (q, r) =
        qr_block_sparse_with_policy_dense(&**backend, tensor.data(), nrow, ExecPolicy::Sequential)?;
    Ok((
        BlockSparseTensor::with_backend(q, backend.clone()),
        BlockSparseTensor::with_backend(r, backend.clone()),
    ))
}

/// Explicit-backend counterpart of [`crate::lq_block_sparse`].
pub fn lq_block_sparse_with_backend<T: Scalar, S: Sector, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &BlockSparseTensor<T, S, B>,
    nrow: usize,
) -> Result<BlockSparseQrResult<T, S, B>, LinalgError> {
    check_bsp_data_layout_order_matches(tensor.data(), &**backend, "lq_block_sparse")?;
    let (l, q) =
        lq_block_sparse_with_policy_dense(&**backend, tensor.data(), nrow, ExecPolicy::Sequential)?;
    Ok((
        BlockSparseTensor::with_backend(l, backend.clone()),
        BlockSparseTensor::with_backend(q, backend.clone()),
    ))
}

/// Explicit-backend counterpart of [`crate::contract_block_sparse`].
///
/// The layout-order invariant is checked against the supplied backend for both
/// operands before the per-sector GEMMs.
pub fn contract_block_sparse_with_backend<T: Scalar, S: Sector, B: ComputeBackend>(
    backend: &Arc<B>,
    lhs: &BlockSparseTensor<T, S, B>,
    rhs: &BlockSparseTensor<T, S, B>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
) -> Result<BlockSparseContractResult<T, S, B>, LinalgError> {
    check_bsp_data_layout_order_matches(lhs.data(), &**backend, "contract_block_sparse: lhs")?;
    check_bsp_data_layout_order_matches(rhs.data(), &**backend, "contract_block_sparse: rhs")?;
    let result = contract_block_sparse_with_policy_dense(
        &**backend,
        lhs.data(),
        rhs.data(),
        axes_lhs,
        axes_rhs,
        ExecPolicy::Sequential,
    )?;
    match result {
        BlockSparseContractResultBsp::Tensor(t) => Ok(BlockSparseContractResult::Tensor(
            BlockSparseTensor::with_backend(t, backend.clone()),
        )),
        BlockSparseContractResultBsp::Scalar(s) => Ok(BlockSparseContractResult::Scalar(s)),
    }
}

/// Explicit-backend counterpart of [`crate::permute_block_sparse`].
pub fn permute_block_sparse_with_backend<T, S, B>(
    backend: &Arc<B>,
    tensor: &BlockSparseTensor<T, S, B>,
    perm: &[usize],
) -> Result<BlockSparseTensor<T, S, B>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    check_bsp_data_layout_order_matches(tensor.data(), &**backend, "permute_block_sparse")?;
    let result = permute_block_sparse_dense(&**backend, tensor.data(), perm)?;
    Ok(BlockSparseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::fuse_legs_block_sparse`].
pub fn fuse_legs_block_sparse_with_backend<T, S, B>(
    backend: &Arc<B>,
    tensor: &BlockSparseTensor<T, S, B>,
    start: usize,
    count: usize,
    fused_direction: Direction,
) -> Result<BlockSparseTensor<T, S, B>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    check_bsp_data_layout_order_matches(tensor.data(), &**backend, "fuse_legs_block_sparse")?;
    let result =
        fuse_legs_block_sparse_dense(&**backend, tensor.data(), start, count, fused_direction)?;
    Ok(BlockSparseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::diagonal_scale_block_sparse`].
pub fn diagonal_scale_block_sparse_with_backend<T, S, B>(
    backend: &Arc<B>,
    tensor: &BlockSparseTensor<T, S, B>,
    weights: &BlockSingularValues<T::Real, S>,
    axis: usize,
) -> Result<BlockSparseTensor<T, S, B>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    check_bsp_data_layout_order_matches(tensor.data(), &**backend, "diagonal_scale_block_sparse")?;
    let result = diagonal_scale_block_sparse_dense(&**backend, tensor.data(), weights, axis)?;
    Ok(BlockSparseTensor::with_backend(result, backend.clone()))
}
