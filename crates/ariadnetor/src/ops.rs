//! High-level free functions that extract the backend from Tensor.
//!
//! These wrap `arnet_linalg::*` by pulling the backend from the tensor
//! arguments, so users never need to pass `&backend` explicitly. The
//! umbrella's `DenseTensor` carries a `DenseTensorData<T>` payload,
//! which is exactly what the migrated linalg pub fns accept, so the
//! wrappers pass `tensor.data()` straight through without copying
//! (issue #259, drift E).

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::LinalgError;

use crate::{DenseTensor, Tensor};

// ============================================================================
// Result type aliases
// ============================================================================

/// QR result: `(Q, R)`.
pub type QrResult<S, B> = (DenseTensor<S, B>, DenseTensor<S, B>);

/// LQ result: `(L, Q)`.
pub type LqResult<S, B> = (DenseTensor<S, B>, DenseTensor<S, B>);

/// Self-adjoint eigenvalue result: `(eigenvalues, eigenvectors)`.
pub type EighResult<S, B> = (DenseTensor<<S as Scalar>::Real, B>, DenseTensor<S, B>);

/// General eigenvalue result: `(eigenvalues, eigenvectors)` (complex).
pub type EigResult<S, B> = (
    DenseTensor<<S as Scalar>::Complex, B>,
    DenseTensor<<S as Scalar>::Complex, B>,
);

// ============================================================================
// Bridging helper
// ============================================================================

fn wrap<S: Scalar, B: ComputeBackend>(
    data: arnet_tensor::DenseTensorData<S>,
    backend: &Arc<B>,
) -> DenseTensor<S, B> {
    Tensor::with_backend(data, Arc::clone(backend))
}

// ============================================================================
// Contraction / Einsum
// ============================================================================

/// Tensor contraction via einsum notation.
pub fn contract<S: Scalar, B: ComputeBackend>(
    lhs: &DenseTensor<S, B>,
    rhs: &DenseTensor<S, B>,
    notation: &str,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let result = arnet_linalg::contract(lhs.backend(), lhs.data(), rhs.data(), notation)?;
    Ok(wrap(result, lhs.backend_arc()))
}

/// Einsum over N tensors.
pub fn einsum<S: Scalar, B: ComputeBackend>(
    notation: &str,
    tensors: &[&DenseTensor<S, B>],
) -> Result<DenseTensor<S, B>, LinalgError> {
    assert!(!tensors.is_empty(), "einsum requires at least one tensor");
    let backend_arc = tensors[0].backend_arc();
    let refs: Vec<&arnet_tensor::DenseTensorData<S>> = tensors.iter().map(|t| t.data()).collect();
    let result = arnet_linalg::einsum(tensors[0].backend(), &refs, notation)?;
    Ok(wrap(result, backend_arc))
}

// ============================================================================
// Transpose
// ============================================================================

/// Permute tensor axes.
pub fn transpose<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    perm: &[usize],
) -> Result<DenseTensor<S, B>, LinalgError> {
    let result = arnet_linalg::transpose(tensor.backend(), tensor.data(), perm)?;
    Ok(wrap(result, tensor.backend_arc()))
}

// ============================================================================
// Decompositions
// ============================================================================

/// Thin QR: A = Q * R.
pub fn qr<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<QrResult<S, B>, LinalgError> {
    let (q, r) = arnet_linalg::qr(tensor.backend(), tensor.data(), nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(q, ba), wrap(r, ba)))
}

/// Thin LQ: A = L * Q.
pub fn lq<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<LqResult<S, B>, LinalgError> {
    let (l, q) = arnet_linalg::lq(tensor.backend(), tensor.data(), nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(l, ba), wrap(q, ba)))
}

// ============================================================================
// Eigenvalue
// ============================================================================

/// Self-adjoint eigenvalue decomposition.
pub fn eigh<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<EighResult<S, B>, LinalgError> {
    let (w, v) = arnet_linalg::eigh(tensor.backend(), tensor.data(), nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(w, ba), wrap(v, ba)))
}

/// Eigenvalues of a self-adjoint matrix.
pub fn eigvalsh<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S::Real, B>, LinalgError> {
    let w = arnet_linalg::eigvalsh(tensor.backend(), tensor.data(), nrow)?;
    Ok(wrap(w, tensor.backend_arc()))
}

/// General eigenvalue decomposition.
pub fn eig<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<EigResult<S, B>, LinalgError> {
    let (w, v) = arnet_linalg::eig(tensor.backend(), tensor.data(), nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(w, ba), wrap(v, ba)))
}

/// Eigenvalues of a general matrix.
pub fn eigvals<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S::Complex, B>, LinalgError> {
    let w = arnet_linalg::eigvals(tensor.backend(), tensor.data(), nrow)?;
    Ok(wrap(w, tensor.backend_arc()))
}

// ============================================================================
// Matrix exponential
// ============================================================================

/// Matrix exponential for Hermitian matrices (eigh-based).
pub fn expm_hermitian<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let result = arnet_linalg::expm_hermitian(tensor.backend(), tensor.data(), nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// Matrix exponential for anti-Hermitian matrices (eigh-based).
pub fn expm_antihermitian<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let result = arnet_linalg::expm_antihermitian(tensor.backend(), tensor.data(), nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// General matrix exponential (Padé approximation).
pub fn expm<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let result = arnet_linalg::expm(tensor.backend(), tensor.data(), nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

// ============================================================================
// Linear solve
// ============================================================================

/// Solve AX = B via LU decomposition.
pub fn solve<S: Scalar, B: ComputeBackend>(
    a: &DenseTensor<S, B>,
    b: &DenseTensor<S, B>,
    nrow_a: usize,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let result = arnet_linalg::solve(a.backend(), a.data(), b.data(), nrow_a)?;
    Ok(wrap(result, a.backend_arc()))
}

/// Matrix inverse via LU decomposition.
pub fn inverse<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let result = arnet_linalg::inverse(tensor.backend(), tensor.data(), nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

// ============================================================================
// Scalar operations (backend-free, but return Tensor)
// ============================================================================

/// Scale tensor by a scalar factor (out-of-place).
pub fn scale<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    factor: S,
) -> DenseTensor<S, B> {
    let result = arnet_linalg::scale(tensor.data(), factor);
    wrap(result, tensor.backend_arc())
}

/// Frobenius norm.
pub fn norm<S: Scalar, B: ComputeBackend>(tensor: &DenseTensor<S, B>) -> S::Real {
    arnet_linalg::norm(tensor.data())
}

/// Normalize to unit norm (out-of-place).
pub fn normalize<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
) -> (DenseTensor<S, B>, S::Real) {
    let (result, n) = arnet_linalg::normalize(tensor.data());
    (wrap(result, tensor.backend_arc()), n)
}

/// Partial trace over bond index pairs.
pub fn trace<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    pairs: &[(usize, usize)],
) -> Result<DenseTensor<S, B>, LinalgError> {
    let result = arnet_linalg::trace(tensor.data(), pairs)?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// Diagonal extraction (2D → 1D) or construction (1D → 2D). Returns
/// a `DenseTensor`; rank-1 outputs encode the diagonal as a plain
/// 1-D tensor.
pub fn diag<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let result = arnet_linalg::diag(tensor.data())?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// Linear combination of tensors.
pub fn linear_combine<S: Scalar, B: ComputeBackend>(
    tensors: &[&DenseTensor<S, B>],
    coefs: &[S],
) -> Result<DenseTensor<S, B>, LinalgError> {
    if tensors.is_empty() {
        return Err(LinalgError::InvalidArgument(
            "Cannot combine empty tensor list".to_string(),
        ));
    }
    let refs: Vec<&arnet_tensor::DenseTensorData<S>> = tensors.iter().map(|t| t.data()).collect();
    let result = arnet_linalg::linear_combine(&refs, coefs)?;
    Ok(wrap(result, tensors[0].backend_arc()))
}
