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

/// Block-sparse thin SVD via the fused sector method, using the supplied backend.
pub fn svd_block_sparse_with_backend<T: Scalar, S: Sector, B: ComputeBackend>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    nrow: usize,
) -> Result<BlockSparseSvdResult<T, S>, LinalgError> {
    check_bsp_data_layout_order_matches(tensor.data(), backend, "svd_block_sparse")?;
    let (u, s, vt) =
        svd_block_sparse_with_policy_dense(backend, tensor.data(), nrow, ExecPolicy::Sequential)?;
    Ok((
        BlockSparseTensor::from_data(u),
        s,
        BlockSparseTensor::from_data(vt),
    ))
}

/// Block-sparse truncated SVD via the fused sector method, using the supplied backend.
pub fn trunc_svd_block_sparse_with_backend<T: Scalar, S: Sector, B: ComputeBackend>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<BlockSparseTruncSvdResult<T, S>, LinalgError> {
    check_bsp_data_layout_order_matches(tensor.data(), backend, "trunc_svd_block_sparse")?;
    let (u, s, vt, err) = trunc_svd_block_sparse_with_policy_dense(
        backend,
        tensor.data(),
        nrow,
        params,
        ExecPolicy::Sequential,
    )?;
    Ok((
        BlockSparseTensor::from_data(u),
        s,
        BlockSparseTensor::from_data(vt),
        err,
    ))
}

/// Block-sparse QR via the fused sector method, using the supplied backend.
pub fn qr_block_sparse_with_backend<T: Scalar, S: Sector, B: ComputeBackend>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    nrow: usize,
) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
    check_bsp_data_layout_order_matches(tensor.data(), backend, "qr_block_sparse")?;
    let (q, r) =
        qr_block_sparse_with_policy_dense(backend, tensor.data(), nrow, ExecPolicy::Sequential)?;
    Ok((
        BlockSparseTensor::from_data(q),
        BlockSparseTensor::from_data(r),
    ))
}

/// Block-sparse LQ via the fused sector method, using the supplied backend.
pub fn lq_block_sparse_with_backend<T: Scalar, S: Sector, B: ComputeBackend>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    nrow: usize,
) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
    check_bsp_data_layout_order_matches(tensor.data(), backend, "lq_block_sparse")?;
    let (l, q) =
        lq_block_sparse_with_policy_dense(backend, tensor.data(), nrow, ExecPolicy::Sequential)?;
    Ok((
        BlockSparseTensor::from_data(l),
        BlockSparseTensor::from_data(q),
    ))
}

/// Block-sparse contraction over the given axis pairs, using the supplied backend.
///
/// The layout-order invariant is checked against the supplied backend for both
/// operands before the per-sector GEMMs.
pub fn contract_block_sparse_with_backend<T: Scalar, S: Sector, B: ComputeBackend>(
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
    B: ComputeBackend,
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
    B: ComputeBackend,
{
    check_bsp_data_layout_order_matches(tensor.data(), backend, "fuse_legs_block_sparse")?;
    let result =
        fuse_legs_block_sparse_dense(backend, tensor.data(), start, count, fused_direction)?;
    Ok(BlockSparseTensor::from_data(result))
}

/// Block-sparse per-sector diagonal scaling, using the supplied backend.
pub fn diagonal_scale_block_sparse_with_backend<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensor<T, S>,
    weights: &BlockSingularValues<T::Real, S>,
    axis: usize,
) -> Result<BlockSparseTensor<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    check_bsp_data_layout_order_matches(tensor.data(), backend, "diagonal_scale_block_sparse")?;
    let result = diagonal_scale_block_sparse_dense(backend, tensor.data(), weights, axis)?;
    Ok(BlockSparseTensor::from_data(result))
}
