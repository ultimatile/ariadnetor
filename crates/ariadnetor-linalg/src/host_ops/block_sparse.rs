//! Host-defaulting method forms of the block-sparse explicit-backend
//! operations. See the parent module for the authority semantics.

use arnet_core::Scalar;
use arnet_tensor::{BlockSparseTensor, Direction, Host, Sector};

use crate::block_sparse_contract::BlockSparseContractResult;
use crate::block_sparse_decomp::{
    BlockScalars, BlockSparseEigResult, BlockSparseEighResult, BlockSparseQrResult,
    BlockSparseSvdResult, BlockSparseTruncSvdResult,
};
use crate::block_sparse_with_backend::{
    contract_block_sparse_with_backend, diagonal_scale_block_sparse_with_backend,
    eig_block_sparse_with_backend, eigh_block_sparse_with_backend,
    eigvals_block_sparse_with_backend, eigvalsh_block_sparse_with_backend,
    expm_antihermitian_block_sparse_with_backend, expm_block_sparse_with_backend,
    expm_hermitian_block_sparse_with_backend, fuse_legs_block_sparse_with_backend,
    permute_block_sparse_with_backend, trace_block_sparse_with_backend,
};
use crate::decompose_dispatch::{lq, qr, svd, trunc_svd};
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
    /// Host-defaulting counterpart of [`crate::svd`].
    fn svd(&self, nrow: usize) -> Result<BlockSparseSvdResult<T, S>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::trunc_svd`].
    fn trunc_svd(
        &self,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<BlockSparseTruncSvdResult<T, S>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::qr`].
    fn qr(&self, nrow: usize) -> Result<BlockSparseQrResult<T, S>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::lq`].
    fn lq(&self, nrow: usize) -> Result<BlockSparseQrResult<T, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::contract_block_sparse_with_backend`]; the receiver is the
    /// left operand.
    fn contract(
        &self,
        rhs: &BlockSparseTensor<T, S>,
        axes_lhs: &[usize],
        axes_rhs: &[usize],
    ) -> Result<BlockSparseContractResult<T, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::permute_block_sparse_with_backend`].
    fn permute(&self, perm: &[usize]) -> Result<BlockSparseTensor<T, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::fuse_legs_block_sparse_with_backend`].
    fn fuse_legs(
        &self,
        start: usize,
        count: usize,
        fused_direction: Direction,
    ) -> Result<BlockSparseTensor<T, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::diagonal_scale_block_sparse_with_backend`].
    fn diagonal_scale(
        &self,
        weights: &BlockScalars<T::Real, S>,
        axis: usize,
    ) -> Result<BlockSparseTensor<T, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::trace_block_sparse_with_backend`].
    fn trace(&self, pairs: &[(usize, usize)]) -> Result<BlockSparseTensor<T, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::eigh_block_sparse_with_backend`].
    fn eigh(&self, nrow: usize) -> Result<BlockSparseEighResult<T, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::eigvalsh_block_sparse_with_backend`].
    fn eigvalsh(&self, nrow: usize) -> Result<BlockScalars<T::Real, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::eig_block_sparse_with_backend`].
    fn eig(&self, nrow: usize) -> Result<BlockSparseEigResult<T, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::eigvals_block_sparse_with_backend`].
    fn eigvals(&self, nrow: usize) -> Result<BlockScalars<T::Complex, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::expm_block_sparse_with_backend`].
    fn expm(&self, nrow: usize) -> Result<BlockSparseTensor<T, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::expm_hermitian_block_sparse_with_backend`].
    fn expm_hermitian(&self, nrow: usize) -> Result<BlockSparseTensor<T, S>, LinalgError>;

    /// Host-defaulting counterpart of
    /// [`crate::expm_antihermitian_block_sparse_with_backend`].
    fn expm_antihermitian(&self, nrow: usize) -> Result<BlockSparseTensor<T, S>, LinalgError>;
}

impl<T: Scalar, S: Sector> BlockSparseHostOps<T, S> for BlockSparseTensor<T, S> {
    fn svd(&self, nrow: usize) -> Result<BlockSparseSvdResult<T, S>, LinalgError> {
        svd(Host::shared().as_ref(), self, nrow)
    }

    fn trunc_svd(
        &self,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<BlockSparseTruncSvdResult<T, S>, LinalgError> {
        trunc_svd(Host::shared().as_ref(), self, nrow, params)
    }

    fn qr(&self, nrow: usize) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
        qr(Host::shared().as_ref(), self, nrow)
    }

    fn lq(&self, nrow: usize) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
        lq(Host::shared().as_ref(), self, nrow)
    }

    fn contract(
        &self,
        rhs: &BlockSparseTensor<T, S>,
        axes_lhs: &[usize],
        axes_rhs: &[usize],
    ) -> Result<BlockSparseContractResult<T, S>, LinalgError> {
        contract_block_sparse_with_backend(Host::shared().as_ref(), self, rhs, axes_lhs, axes_rhs)
    }

    fn permute(&self, perm: &[usize]) -> Result<BlockSparseTensor<T, S>, LinalgError> {
        permute_block_sparse_with_backend(Host::shared().as_ref(), self, perm)
    }

    fn fuse_legs(
        &self,
        start: usize,
        count: usize,
        fused_direction: Direction,
    ) -> Result<BlockSparseTensor<T, S>, LinalgError> {
        fuse_legs_block_sparse_with_backend(
            Host::shared().as_ref(),
            self,
            start,
            count,
            fused_direction,
        )
    }

    fn diagonal_scale(
        &self,
        weights: &BlockScalars<T::Real, S>,
        axis: usize,
    ) -> Result<BlockSparseTensor<T, S>, LinalgError> {
        diagonal_scale_block_sparse_with_backend(Host::shared().as_ref(), self, weights, axis)
    }

    fn trace(&self, pairs: &[(usize, usize)]) -> Result<BlockSparseTensor<T, S>, LinalgError> {
        trace_block_sparse_with_backend(Host::shared().as_ref(), self, pairs)
    }

    fn eigh(&self, nrow: usize) -> Result<BlockSparseEighResult<T, S>, LinalgError> {
        eigh_block_sparse_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn eigvalsh(&self, nrow: usize) -> Result<BlockScalars<T::Real, S>, LinalgError> {
        eigvalsh_block_sparse_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn eig(&self, nrow: usize) -> Result<BlockSparseEigResult<T, S>, LinalgError> {
        eig_block_sparse_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn eigvals(&self, nrow: usize) -> Result<BlockScalars<T::Complex, S>, LinalgError> {
        eigvals_block_sparse_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn expm(&self, nrow: usize) -> Result<BlockSparseTensor<T, S>, LinalgError> {
        expm_block_sparse_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn expm_hermitian(&self, nrow: usize) -> Result<BlockSparseTensor<T, S>, LinalgError> {
        expm_hermitian_block_sparse_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn expm_antihermitian(&self, nrow: usize) -> Result<BlockSparseTensor<T, S>, LinalgError> {
        expm_antihermitian_block_sparse_with_backend(Host::shared().as_ref(), self, nrow)
    }
}
