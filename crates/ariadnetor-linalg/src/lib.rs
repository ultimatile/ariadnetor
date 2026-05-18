//! Backend-agnostic linear algebra API for Ariadnetor
//!
//! Provides high-level tensor operations that delegate to a [`ComputeBackend`]
//! for the actual computation. This decouples tensor data from compute libraries
//! (faer, HPTT, etc.) so that `ariadnetor-tensor` carries no heavy dependencies.
//!
//! # Operations
//!
//! - [`transpose`]: Permute tensor axes via backend
//! - [`contract`]: Tensor contraction via Einstein summation (permute + GEMM)
//! - [`contract_block_sparse`]: Block-sparse tensor contraction (block pairing + GEMM)
//! - [`svd_block_sparse`]: Block-sparse SVD via fused sector method
//! - [`trunc_svd_block_sparse`]: Truncated block-sparse SVD with bond dimension control
//! - [`qr_block_sparse`]: Block-sparse QR decomposition via fused sector method
//! - [`lq_block_sparse`]: Block-sparse LQ decomposition via fused sector method
//! - [`permute_block_sparse`]: Block-sparse tensor axis permutation
//! - [`fuse_legs_block_sparse`]: Block-sparse consecutive leg fusion
//! - [`scale`]: Scalar multiplication (out-of-place)
//! - [`norm`]: Frobenius norm
//! - [`normalize`]: Normalize to unit norm (out-of-place)
//! - [`linear_combine`]: Linear combination of tensors
//! - [`trace`]: Partial trace over bond index pairs
//! - [`diag`]: Diagonal extraction and construction
//! - [`svd`]: Thin SVD decomposition via backend
//! - [`trunc_svd`]: Truncated SVD with bond dimension control
//! - [`qr`]: Thin QR decomposition via backend
//! - [`lq`]: Thin LQ decomposition via backend
//! - [`eigh`]: Self-adjoint eigenvalue decomposition via backend
//! - [`eigvalsh`]: Eigenvalues-only variant of `eigh`
//! - [`eig`]: General eigenvalue decomposition via backend
//! - [`eigvals`]: Eigenvalues-only variant of `eig`
//! - [`expm_hermitian`]: Matrix exponential for Hermitian matrices
//! - [`expm_antihermitian`]: Matrix exponential for anti-Hermitian matrices
//! - [`solve`]: Linear solve AX = B via backend (LU decomposition)
//! - [`inverse`]: Matrix inverse via LU decomposition

mod block_sparse_contract;
mod block_sparse_decomp;
mod block_sparse_decomp_canonical;
mod block_sparse_fuse;
mod block_sparse_layout_ops;
mod block_sparse_permute;
mod block_sparse_scale;
mod contract;
mod decomposition;
mod eigen;
mod einsum;
mod error;
mod expm;
mod scalar_ops;
mod solve;
mod transpose;

#[cfg(test)]
pub(crate) mod test_util;

pub use arnet_core::backend::ComputeBackend;
pub use error::LinalgError;

pub use block_sparse_contract::{
    BlockSparseContractResult, BlockSparseContractResultRepr, contract_block_sparse,
    contract_block_sparse_repr, contract_block_sparse_with_policy,
    contract_block_sparse_with_policy_repr,
};
pub use block_sparse_decomp::{
    BlockSingularValues, BlockSparseQrResult, BlockSparseQrResultRepr, BlockSparseSvdResult,
    BlockSparseSvdResultRepr, BlockSparseTruncSvdResult, BlockSparseTruncSvdResultRepr,
    lq_block_sparse_repr, lq_block_sparse_with_policy_repr, qr_block_sparse_repr,
    qr_block_sparse_with_policy_repr, svd_block_sparse_repr, svd_block_sparse_with_policy_repr,
    trunc_svd_block_sparse_repr, trunc_svd_block_sparse_with_policy_repr,
};
pub use block_sparse_decomp_canonical::{
    lq_block_sparse, lq_block_sparse_with_policy, qr_block_sparse, qr_block_sparse_with_policy,
    svd_block_sparse, svd_block_sparse_with_policy, trunc_svd_block_sparse,
    trunc_svd_block_sparse_with_policy,
};
pub use block_sparse_fuse::{fuse_legs_block_sparse, fuse_legs_block_sparse_repr};
pub use block_sparse_permute::{permute_block_sparse, permute_block_sparse_repr};
pub use block_sparse_scale::{diagonal_scale_block_sparse, diagonal_scale_block_sparse_repr};
pub use contract::{contract, contract_dense, contract_with_policy, contract_with_policy_dense};
pub use decomposition::{
    LqResult, LqResultDense, QrResult, QrResultDense, SvdResult, SvdResultDense, TruncSvdParams,
    TruncSvdResult, TruncSvdResultDense, lq, lq_dense, lq_with_policy, lq_with_policy_dense, qr,
    qr_dense, qr_with_policy, qr_with_policy_dense, svd, svd_dense, svd_with_policy,
    svd_with_policy_dense, trunc_svd, trunc_svd_dense, trunc_svd_with_policy,
    trunc_svd_with_policy_dense,
};
pub use eigen::{
    EigResult, EigResultDense, EighResult, EighResultDense, eig, eig_dense, eig_with_policy,
    eig_with_policy_dense, eigh, eigh_dense, eigh_with_policy, eigh_with_policy_dense, eigvals,
    eigvals_dense, eigvalsh, eigvalsh_dense,
};
pub use einsum::{einsum, einsum_dense};
pub use expm::{
    expm, expm_antihermitian, expm_antihermitian_dense, expm_dense, expm_hermitian,
    expm_hermitian_dense,
};
pub use scalar_ops::{
    diag, diag_dense, diagonal_scale, diagonal_scale_dense, linear_combine, linear_combine_dense,
    norm, norm_dense, normalize, normalize_dense, scale, scale_dense, trace, trace_dense,
};
pub use solve::{
    inverse, inverse_dense, solve, solve_dense, solve_with_policy, solve_with_policy_dense,
};
pub use transpose::{
    conjugate_transpose, conjugate_transpose_dense, conjugate_transpose_with_policy,
    conjugate_transpose_with_policy_dense, transpose, transpose_dense, transpose_with_policy,
    transpose_with_policy_dense,
};
