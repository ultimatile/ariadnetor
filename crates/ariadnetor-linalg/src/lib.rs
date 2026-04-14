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
mod block_sparse_scale;
mod contract;
mod decomposition;
mod eigen;
mod einsum;
mod error;
mod expm;
mod reorder;
mod scalar_ops;
mod solve;
mod transpose;

pub use arnet_core::backend::ComputeBackend;
pub use error::LinalgError;

pub use block_sparse_contract::{BlockSparseContractResult, contract_block_sparse};
pub use block_sparse_decomp::{
    BlockSingularValues, BlockSparseQrResult, BlockSparseSvdResult, BlockSparseTruncSvdResult,
    lq_block_sparse, qr_block_sparse, svd_block_sparse, trunc_svd_block_sparse,
};
pub use block_sparse_scale::diagonal_scale_block_sparse;
pub use contract::contract;
pub use decomposition::{
    LqResult, QrResult, SvdResult, TruncSvdParams, TruncSvdResult, lq, qr, svd, trunc_svd,
};
pub use eigen::{EigResult, EighResult, eig, eigh, eigvals, eigvalsh};
pub use einsum::einsum;
pub use expm::{expm, expm_antihermitian, expm_hermitian};
pub use reorder::{flat_index, reorder};
pub use scalar_ops::{diag, diagonal_scale, linear_combine, norm, normalize, scale, trace};
pub use solve::{inverse, solve};
pub use transpose::{conjugate_transpose, transpose};
