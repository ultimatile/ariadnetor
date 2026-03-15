//! Pluggable compute backend trait
//!
//! Based on dev-docs/design/gpu_readiness_plan.md

use crate::scalar::Scalar;

/// Device type for backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
}

/// GEMM operation descriptor
pub struct GemmDescriptor<'a, T> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: T,
    pub a: &'a [T],
    pub b: &'a [T],
    pub beta: T,
    pub c: &'a mut [T],
    pub trans_a: bool,
    pub trans_b: bool,
}

/// Transpose operation descriptor
pub struct TransposeDescriptor<'a, T> {
    pub input: &'a [T],
    pub output: &'a mut [T],
    pub shape: &'a [usize],
    pub perm: &'a [usize],
}

/// Thin SVD operation descriptor: A = U * diag(S) * Vt
///
/// Computes the thin SVD of an m×n matrix A (row-major).
/// Outputs: U (m×k, row-major), S (k singular values), Vt (k×n, row-major)
/// where k = min(m, n).
pub struct SvdDescriptor<'a, T: Scalar> {
    pub m: usize,
    pub n: usize,
    pub a: &'a [T],
    pub u: &'a mut [T],
    pub s: &'a mut [T::Real],
    pub vt: &'a mut [T],
}

/// Thin QR decomposition descriptor: A = Q * R
///
/// Computes the thin QR of an m×n matrix A (row-major).
/// Outputs: Q (m×k, row-major), R (k×n, row-major)
/// where k = min(m, n).
pub struct QrDescriptor<'a, T> {
    pub m: usize,
    pub n: usize,
    pub a: &'a [T],
    pub q: &'a mut [T],
    pub r: &'a mut [T],
}

/// Thin LQ decomposition descriptor: A = L * Q
///
/// Computes the thin LQ of an m×n matrix A (row-major).
/// Outputs: L (m×k, row-major), Q (k×n, row-major)
/// where k = min(m, n).
pub struct LqDescriptor<'a, T> {
    pub m: usize,
    pub n: usize,
    pub a: &'a [T],
    pub l: &'a mut [T],
    pub q: &'a mut [T],
}

/// Self-adjoint eigenvalue decomposition descriptor: A = V * diag(W) * V^H
///
/// Computes eigenvalues and eigenvectors of an n×n self-adjoint matrix A (row-major).
/// Outputs: W (n real eigenvalues, ascending), V (n×n eigenvectors, row-major)
pub struct EighDescriptor<'a, T: Scalar> {
    pub n: usize,
    pub a: &'a [T],
    pub w: &'a mut [T::Real],
    pub v: &'a mut [T],
}

/// General eigenvalue decomposition descriptor
///
/// Computes eigenvalues and right eigenvectors of an n×n matrix A (row-major).
/// Outputs are always complex: W (n complex eigenvalues), V (n×n eigenvectors, row-major)
pub struct EigDescriptor<'a, T: Scalar> {
    pub n: usize,
    pub a: &'a [T],
    pub w: &'a mut [T::Complex],
    pub v: &'a mut [T::Complex],
}

/// Pluggable compute backend trait
pub trait ComputeBackend: Send + Sync {
    /// Backend name
    fn name(&self) -> &'static str;

    /// Device type
    fn device_type(&self) -> DeviceType;

    /// Check if backend is available
    fn is_available(&self) -> bool {
        true
    }

    /// GEMM: C = alpha * A * B + beta * C
    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError>;

    /// Transpose tensor
    fn transpose<T: Scalar>(&self, desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError>;

    /// Thin SVD: A = U * diag(S) * Vt
    fn svd<T: Scalar>(&self, _desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("svd".into()))
    }

    /// Thin QR: A = Q * R
    fn qr<T: Scalar>(&self, _desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("qr".into()))
    }

    /// Thin LQ: A = L * Q
    fn lq<T: Scalar>(&self, _desc: LqDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("lq".into()))
    }

    /// Self-adjoint eigenvalue decomposition: A = V * diag(W) * V^H
    fn eigh<T: Scalar>(&self, _desc: EighDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("eigh".into()))
    }

    /// General eigenvalue decomposition
    fn eig<T: Scalar>(&self, _desc: EigDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("eig".into()))
    }
}

/// Backend error
#[derive(Debug)]
pub enum BackendError {
    NotSupported(String),
    InvalidDimension(String),
    ExecutionFailed(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSupported(msg) => write!(f, "Not supported: {}", msg),
            Self::InvalidDimension(msg) => write!(f, "Invalid dimension: {}", msg),
            Self::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}
