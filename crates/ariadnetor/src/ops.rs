//! Re-exports of the `arnet_linalg` Tensor-typed free-fn surface.
//!
//! With the #262 redo, `arnet_linalg` itself accepts `&DenseTensor<T, B>`
//! and returns `DenseTensor<T, B>`, so the umbrella no longer needs the
//! pre-redo `bridge_in` / `bridge_out` copies around each call site.
//!
//! Inherent `DenseTensor` scalar ops (`scale`, `norm`, `normalize`,
//! `linear_combine`) still live as thin wrappers below, since they
//! short-circuit the `arnet_linalg` dispatch entirely and operate on
//! the joined-form storage directly.

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::LinalgError;
use arnet_tensor::DenseTensor;

// ============================================================================
// Result type aliases — re-exported from arnet_linalg
// ============================================================================

pub use arnet_linalg::{EigResult, EighResult, LqResult, QrResult, SvdResult, TruncSvdResult};

// ============================================================================
// Tensor-typed free fns — re-exported from arnet_linalg
// ============================================================================

pub use arnet_linalg::{
    contract, diag, eig, eigh, eigvals, eigvalsh, einsum, expm, expm_antihermitian, expm_hermitian,
    inverse, lq, qr, solve, svd, trace, transpose, trunc_svd,
};

// ============================================================================
// Inherent-method shortcuts
//
// These four ops operate on the joined-form storage directly (no
// backend kernel involved), so they remain umbrella-level shims rather
// than passing through arnet_linalg.
// ============================================================================

/// Scale tensor by a scalar factor (out-of-place).
pub fn scale<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    factor: S,
) -> DenseTensor<S, B> {
    tensor.scaled(factor)
}

/// Frobenius norm.
pub fn norm<S: Scalar, B: ComputeBackend>(tensor: &DenseTensor<S, B>) -> S::Real {
    tensor.norm()
}

/// Normalize to unit norm (out-of-place).
pub fn normalize<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
) -> (DenseTensor<S, B>, S::Real) {
    tensor.normalized()
}

/// Linear combination of tensors.
pub fn linear_combine<S: Scalar, B: ComputeBackend>(
    tensors: &[&DenseTensor<S, B>],
    coefs: &[S],
) -> Result<DenseTensor<S, B>, LinalgError> {
    DenseTensor::linear_combine(tensors, coefs).map_err(LinalgError::InvalidArgument)
}
