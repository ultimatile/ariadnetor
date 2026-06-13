//! Backend-agnostic linear algebra API for Ariadnetor
//!
//! Provides high-level tensor operations that delegate to a [`ComputeBackend`]
//! for the actual computation. This decouples tensor data from compute libraries
//! (faer, HPTT, etc.) so that `ariadnetor-tensor` carries no heavy dependencies.
//!
//! # Operations
//!
//! Most operations are exposed in two forms over the same kernels:
//!
//! - An explicit-backend free function (`*_with_backend`) that takes the
//!   backend at the call site — e.g. [`svd_with_backend`], [`contract_with_backend`],
//!   [`transpose_with_backend`], [`trace_with_backend`], the block-sparse
//!   family ([`svd_block_sparse_with_backend`], [`contract_block_sparse_with_backend`],
//!   …). The `*_with_policy` variants ([`svd_with_policy`], …) add an explicit
//!   [`ExecPolicy`](arnet_core::backend::ExecPolicy).
//! - An ergonomic method form on tensors over the default [`Host`](arnet_tensor::Host)
//!   substrate via the [`DenseHostOps`] / [`BlockSparseHostOps`] extension
//!   traits (`t.svd(nrow)` instead of `svd_with_backend(&backend, &t, nrow)`).
//!
//! [`einsum_with_backend`] is the exception: it has no method form, because its
//! operands are a slice with no natural receiver, so the explicit-backend free
//! function is the only form.
//!
//! Covered operations: transpose, contraction (dense and block-sparse),
//! Einstein summation, partial trace, diagonal extraction / scaling, SVD /
//! truncated SVD / QR / LQ (dense and block-sparse), self-adjoint and general
//! eigenvalue decomposition, the Hermitian / anti-Hermitian / general matrix
//! exponential, linear solve, matrix inverse, block-sparse permutation and leg
//! fusion.

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

pub use block_sparse_contract::BlockSparseContractResult;
pub use block_sparse_decomp::{
    BlockSingularValues, BlockSparseQrResult, BlockSparseSvdResult, BlockSparseTruncSvdResult,
};
pub use contract::contract_with_policy;
pub use decomposition::{
    LqResult, QrResult, SvdResult, TruncSvdParams, TruncSvdResult, lq_with_policy, qr_with_policy,
    svd_with_policy, trunc_svd_with_policy,
};
pub use eigen::{EigResult, EighResult, eig_with_policy, eigh_with_policy};
pub use solve::solve_with_policy;
pub use transpose::transpose_with_policy;

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
