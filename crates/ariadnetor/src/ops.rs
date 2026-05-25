//! Re-exports of the `arnet_linalg` Tensor-typed free-fn surface.
//!
//! `arnet_linalg` accepts `&DenseTensor<T, B>` and returns
//! `DenseTensor<T, B>`, so the umbrella re-exports each call site
//! directly without copy bridges.
//!
//! `linear_combine` is the lone umbrella shim below: it adapts the
//! inherent `DenseTensor::linear_combine`'s `String` error into
//! `LinalgError` for callers that expect the linalg error type.

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
// ============================================================================

/// Linear combination of tensors.
pub fn linear_combine<S: Scalar, B: ComputeBackend>(
    tensors: &[&DenseTensor<S, B>],
    coefs: &[S],
) -> Result<DenseTensor<S, B>, LinalgError> {
    DenseTensor::linear_combine(tensors, coefs).map_err(LinalgError::InvalidArgument)
}
