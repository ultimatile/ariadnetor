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
//! - An explicit-backend free function that takes the backend at the call
//!   site. The non-decomposition ops use the `*_with_backend` form — e.g.
//!   [`contract_with_backend`], [`permute_with_backend`],
//!   [`trace_with_backend`], the block-sparse family
//!   ([`contract_block_sparse_with_backend`], …). The four decompositions
//!   (`svd` / `trunc_svd` / `qr` / `lq`) instead dispatch over layout through
//!   the unified [`svd`] / [`trunc_svd`] / [`qr`] / [`lq`] free fns
//!   ([`LinalgDecompose`]), so one call serves both Dense and BlockSparse. The
//!   `*_with_policy` variants add an explicit
//!   [`ExecPolicy`](arnet_core::backend::ExecPolicy); they are published under
//!   bare names through the [`expert`] module (`expert::permute`,
//!   `expert::svd`, …).
//! - An ergonomic method form on tensors over the default [`Host`](arnet_tensor::Host)
//!   substrate via the [`DenseHostOps`] / [`BlockSparseHostOps`] extension
//!   traits (`t.svd(nrow)` instead of `svd(&backend, &t, nrow)`).
//!
//! [`einsum_with_backend`] is the exception: it has no method form, because its
//! operands are a slice with no natural receiver, so the explicit-backend free
//! function is the only form.
//!
//! Covered operations: axis permutation (dense and block-sparse), contraction
//! (dense and block-sparse), Einstein summation, partial trace (dense and
//! block-sparse), diagonal extraction / scaling, SVD / truncated SVD / QR / LQ
//! (dense and block-sparse), self-adjoint and general eigenvalue decomposition, the
//! Hermitian / anti-Hermitian / general matrix exponential, linear solve,
//! matrix inverse, and block-sparse leg fusion.

#![deny(missing_docs)]

mod block_sparse_contract;
mod block_sparse_decomp;
mod block_sparse_fuse;
mod block_sparse_permute;
mod block_sparse_scale;
mod block_sparse_trace;
mod block_sparse_with_backend;
mod contract;
mod decompose_dispatch;
mod decomposition;
mod eigen;
mod einsum;
mod error;
mod expm;
mod host_ops;
mod perm;
mod scalar_ops;
mod solve;
mod tensor_bridge;
mod transpose;
mod with_backend;

#[cfg(test)]
pub(crate) mod test_util;

pub mod expert;

pub use arnet_core::backend::ComputeBackend;
pub use error::LinalgError;

pub use block_sparse_contract::BlockSparseContractResult;
pub use block_sparse_decomp::{
    BlockScalars, BlockSparseEighResult, BlockSparseQrResult, BlockSparseSvdResult,
    BlockSparseTruncSvdResult,
};
pub use decomposition::{LqResult, QrResult, SvdResult, TruncSvdParams, TruncSvdResult};
pub use eigen::{EigResult, EighResult};

// Layout-keyed decomposition dispatch: the unified `svd` / `trunc_svd` / `qr` /
// `lq` entry points serve both Dense and BlockSparse via [`LinalgDecompose`].
// The policy-explicit forms are published under bare names through [`expert`].
pub use decompose_dispatch::{LinalgDecompose, lq, qr, svd, trunc_svd};

// Explicit-backend operation paths (backend supplied at the call site). The
// decomposition ops are not here — they dispatch over layout through the
// unified free fns above.
pub use block_sparse_with_backend::{
    contract_block_sparse_with_backend, diagonal_scale_block_sparse_with_backend,
    eigh_block_sparse_with_backend, eigvalsh_block_sparse_with_backend,
    fuse_legs_block_sparse_with_backend, permute_block_sparse_with_backend,
    trace_block_sparse_with_backend,
};
pub use with_backend::{
    contract_with_backend, diag_with_backend, diagonal_scale_with_backend, eig_with_backend,
    eigh_with_backend, eigvals_with_backend, eigvalsh_with_backend, einsum_with_backend,
    expm_antihermitian_with_backend, expm_hermitian_with_backend, expm_with_backend,
    inverse_with_backend, permute_with_backend, solve_with_backend, trace_with_backend,
};

// Ergonomic Host-defaulting method surface over the explicit-backend paths.
pub use host_ops::{BlockSparseHostOps, DenseHostOps};
