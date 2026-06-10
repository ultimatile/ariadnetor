//! Host-defaulting method forms of the block-sparse explicit-backend
//! operations. See the parent module for the authority semantics.

use arnet_core::Scalar;
use arnet_tensor::{BlockSparseTensor, Direction, Host, Sector};

use crate::block_sparse_contract::BlockSparseContractResult;
use crate::block_sparse_decomp::{
    BlockSingularValues, BlockSparseQrResult, BlockSparseSvdResult, BlockSparseTruncSvdResult,
};
use crate::block_sparse_with_backend::{
    contract_block_sparse_with_backend, diagonal_scale_block_sparse_with_backend,
    fuse_legs_block_sparse_with_backend, lq_block_sparse_with_backend,
    permute_block_sparse_with_backend, qr_block_sparse_with_backend, svd_block_sparse_with_backend,
    trunc_svd_block_sparse_with_backend,
};
use crate::decomposition::TruncSvdParams;
use crate::error::LinalgError;

/// Host-defaulting method forms of the block-sparse explicit-backend
/// operations.
///
/// Method names drop the `_block_sparse` suffix of their free-fn twins —
/// the receiver type already says it. The dense inherent
/// `DenseTensor::fuse_legs(range)` is a different operation with a
/// different signature; this trait's [`fuse_legs`](Self::fuse_legs)
/// exists only on block-sparse receivers.
pub trait BlockSparseHostOps<T: Scalar, S: Sector> {
    /// Host-defaulting counterpart of [`crate::svd_block_sparse_with_backend`].
    fn svd(&self, nrow: usize) -> Result<BlockSparseSvdResult<T, S, Host>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::trunc_svd_block_sparse_with_backend`].
    fn trunc_svd(
        &self,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<BlockSparseTruncSvdResult<T, S, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::qr_block_sparse_with_backend`].
    fn qr(&self, nrow: usize) -> Result<BlockSparseQrResult<T, S, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::lq_block_sparse_with_backend`].
    fn lq(&self, nrow: usize) -> Result<BlockSparseQrResult<T, S, Host>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::contract_block_sparse_with_backend`]; the receiver is the
    /// left operand.
    fn contract(
        &self,
        rhs: &BlockSparseTensor<T, S, Host>,
        axes_lhs: &[usize],
        axes_rhs: &[usize],
    ) -> Result<BlockSparseContractResult<T, S, Host>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::permute_block_sparse_with_backend`].
    fn permute(&self, perm: &[usize]) -> Result<BlockSparseTensor<T, S, Host>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::fuse_legs_block_sparse_with_backend`].
    fn fuse_legs(
        &self,
        start: usize,
        count: usize,
        fused_direction: Direction,
    ) -> Result<BlockSparseTensor<T, S, Host>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::diagonal_scale_block_sparse_with_backend`].
    fn diagonal_scale(
        &self,
        weights: &BlockSingularValues<T::Real, S>,
        axis: usize,
    ) -> Result<BlockSparseTensor<T, S, Host>, LinalgError>;
}

impl<T: Scalar, S: Sector> BlockSparseHostOps<T, S> for BlockSparseTensor<T, S, Host> {
    fn svd(&self, nrow: usize) -> Result<BlockSparseSvdResult<T, S, Host>, LinalgError> {
        svd_block_sparse_with_backend(&Host::shared(), self, nrow)
    }

    fn trunc_svd(
        &self,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<BlockSparseTruncSvdResult<T, S, Host>, LinalgError> {
        trunc_svd_block_sparse_with_backend(&Host::shared(), self, nrow, params)
    }

    fn qr(&self, nrow: usize) -> Result<BlockSparseQrResult<T, S, Host>, LinalgError> {
        qr_block_sparse_with_backend(&Host::shared(), self, nrow)
    }

    fn lq(&self, nrow: usize) -> Result<BlockSparseQrResult<T, S, Host>, LinalgError> {
        lq_block_sparse_with_backend(&Host::shared(), self, nrow)
    }

    fn contract(
        &self,
        rhs: &BlockSparseTensor<T, S, Host>,
        axes_lhs: &[usize],
        axes_rhs: &[usize],
    ) -> Result<BlockSparseContractResult<T, S, Host>, LinalgError> {
        contract_block_sparse_with_backend(&Host::shared(), self, rhs, axes_lhs, axes_rhs)
    }

    fn permute(&self, perm: &[usize]) -> Result<BlockSparseTensor<T, S, Host>, LinalgError> {
        permute_block_sparse_with_backend(&Host::shared(), self, perm)
    }

    fn fuse_legs(
        &self,
        start: usize,
        count: usize,
        fused_direction: Direction,
    ) -> Result<BlockSparseTensor<T, S, Host>, LinalgError> {
        fuse_legs_block_sparse_with_backend(&Host::shared(), self, start, count, fused_direction)
    }

    fn diagonal_scale(
        &self,
        weights: &BlockSingularValues<T::Real, S>,
        axis: usize,
    ) -> Result<BlockSparseTensor<T, S, Host>, LinalgError> {
        diagonal_scale_block_sparse_with_backend(&Host::shared(), self, weights, axis)
    }
}
