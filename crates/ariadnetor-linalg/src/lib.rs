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
mod block_sparse_fuse;
mod block_sparse_permute;
mod block_sparse_scale;
mod block_sparse_with_backend;
mod contract;
mod decomposition;
mod eigen;
mod einsum;
mod error;
mod expm;
mod host_ops;
mod scalar_ops;
mod solve;
mod tensor_bridge;
mod transpose;
mod with_backend;

#[cfg(test)]
pub(crate) mod test_util;

pub use arnet_core::backend::ComputeBackend;
pub use error::LinalgError;

pub use block_sparse_contract::{BlockSparseContractResult, contract_block_sparse};
pub use block_sparse_decomp::{
    BlockSingularValues, BlockSparseQrResult, BlockSparseSvdResult, BlockSparseTruncSvdResult,
    lq_block_sparse, qr_block_sparse, svd_block_sparse, trunc_svd_block_sparse,
};
pub use block_sparse_fuse::fuse_legs_block_sparse;
pub use block_sparse_permute::permute_block_sparse;
pub use block_sparse_scale::diagonal_scale_block_sparse;
pub use contract::{contract, contract_with_policy};
pub use decomposition::{
    LqResult, QrResult, SvdResult, TruncSvdParams, TruncSvdResult, lq, lq_with_policy, qr,
    qr_with_policy, svd, svd_with_policy, trunc_svd, trunc_svd_with_policy,
};
pub use eigen::{
    EigResult, EighResult, eig, eig_with_policy, eigh, eigh_with_policy, eigvals, eigvalsh,
};
pub use einsum::einsum;
pub use expm::{expm, expm_antihermitian, expm_hermitian};
pub use scalar_ops::{diag, diagonal_scale, trace};
pub use solve::{inverse, solve, solve_with_policy};
pub use transpose::{transpose, transpose_with_policy};

// Explicit-backend operation paths (backend supplied at the call site).
pub use block_sparse_with_backend::{
    contract_block_sparse_with_backend, diagonal_scale_block_sparse_with_backend,
    fuse_legs_block_sparse_with_backend, lq_block_sparse_with_backend,
    permute_block_sparse_with_backend, qr_block_sparse_with_backend, svd_block_sparse_with_backend,
    trunc_svd_block_sparse_with_backend,
};
pub use with_backend::{
    contract_with_backend, diag_with_backend, diagonal_scale_with_backend, eig_with_backend,
    eigh_with_backend, eigvals_with_backend, eigvalsh_with_backend, einsum_with_backend,
    expm_antihermitian_with_backend, expm_hermitian_with_backend, expm_with_backend,
    inverse_with_backend, lq_with_backend, qr_with_backend, solve_with_backend, svd_with_backend,
    trace_with_backend, transpose_with_backend, trunc_svd_with_backend,
};

// Ergonomic Host-defaulting method surface over the explicit-backend paths.
pub use host_ops::{BlockSparseHostOps, DenseHostOps};
