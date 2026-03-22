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

mod contract;
mod decomposition;
mod eigen;
mod einsum;
mod expm;
mod scalar_ops;
mod solve;
mod transpose;

pub use arnet_core::backend::ComputeBackend;

pub use contract::contract;
pub use decomposition::{
    LqResult, QrResult, SvdResult, TruncSvdParams, TruncSvdResult, lq, qr, svd, trunc_svd,
};
pub use eigen::{EigResult, EighResult, eig, eigh, eigvals, eigvalsh};
pub use einsum::einsum;
pub use expm::{expm, expm_antihermitian, expm_hermitian};
pub use scalar_ops::{diag, diagonal_scale, linear_combine, norm, normalize, scale, trace};
pub use solve::{inverse, solve};
pub use transpose::{conjugate_transpose, transpose};
