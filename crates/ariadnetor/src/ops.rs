//! High-level free functions that extract the backend from Tensor.
//!
//! These wrap `arnet_linalg::*` by pulling the backend from the tensor arguments,
//! so users never need to pass `&backend` explicitly.

use std::ops::Mul;
use std::sync::Arc;

use arnet_core::backend::{BackendError, ComputeBackend};
use arnet_core::scalar::Scalar;
use arnet_tensor::{DenseTensor, TensorStorage};
use num_traits::Zero;

use crate::Tensor;

// ============================================================================
// Result type aliases
// ============================================================================

/// SVD result: `(U, S, Vt)` as Tensors.
pub type SvdResult<T, B> = (Tensor<T, B>, Tensor<<T as Scalar>::Real, B>, Tensor<T, B>);

/// Truncated SVD result: `(U, S, Vt, truncation_error)`.
pub type TruncSvdResult<T, B> = (
    Tensor<T, B>,
    Tensor<<T as Scalar>::Real, B>,
    Tensor<T, B>,
    <T as Scalar>::Real,
);

/// QR result: `(Q, R)`.
pub type QrResult<T, B> = (Tensor<T, B>, Tensor<T, B>);

/// LQ result: `(L, Q)`.
pub type LqResult<T, B> = (Tensor<T, B>, Tensor<T, B>);

/// Self-adjoint eigenvalue result: `(eigenvalues, eigenvectors)`.
pub type EighResult<T, B> = (Tensor<<T as Scalar>::Real, B>, Tensor<T, B>);

/// General eigenvalue result: `(eigenvalues, eigenvectors)` (complex).
pub type EigResult<T, B> = (Tensor<<T as Scalar>::Complex, B>, Tensor<<T as Scalar>::Complex, B>);

// ============================================================================
// Helpers
// ============================================================================

/// Extract DenseTensor from Tensor's storage, panicking if not Dense.
fn dense<T: Clone, B: ComputeBackend>(tensor: &Tensor<T, B>) -> &DenseTensor<T> {
    match &tensor.storage {
        TensorStorage::Dense(d) => d,
    }
}

/// Wrap a DenseTensor result back into a Tensor with the given backend.
fn wrap<T, B: ComputeBackend>(dense: DenseTensor<T>, backend: &Arc<B>) -> Tensor<T, B> {
    Tensor::with_backend(TensorStorage::Dense(dense), Arc::clone(backend))
}

// ============================================================================
// Contraction / Einsum
// ============================================================================

/// Tensor contraction via einsum notation.
pub fn contract<T: Scalar, B: ComputeBackend>(
    lhs: &Tensor<T, B>,
    rhs: &Tensor<T, B>,
    notation: &str,
) -> Result<Tensor<T, B>, BackendError> {
    let result = arnet_linalg::contract(lhs.backend(), dense(lhs), dense(rhs), notation)?;
    Ok(wrap(result, lhs.backend_arc()))
}

/// Einsum over N tensors.
pub fn einsum<T: Scalar, B: ComputeBackend>(
    notation: &str,
    tensors: &[&Tensor<T, B>],
) -> Result<Tensor<T, B>, BackendError> {
    assert!(!tensors.is_empty(), "einsum requires at least one tensor");
    let backend_arc = tensors[0].backend_arc();
    let dense_refs: Vec<&DenseTensor<T>> = tensors.iter().map(|t| dense(t)).collect();
    let result = arnet_linalg::einsum(tensors[0].backend(), &dense_refs, notation)?;
    Ok(wrap(result, backend_arc))
}

// ============================================================================
// Transpose
// ============================================================================

/// Permute tensor axes.
pub fn transpose<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    perm: &[usize],
) -> Result<Tensor<T, B>, BackendError> {
    let result = arnet_linalg::transpose(tensor.backend(), dense(tensor), perm)?;
    Ok(wrap(result, tensor.backend_arc()))
}

// ============================================================================
// Decompositions
// ============================================================================

/// Thin SVD: A = U * diag(S) * Vt.
pub fn svd<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<SvdResult<T, B>, BackendError> {
    let (u, s, vt) = arnet_linalg::svd(tensor.backend(), dense(tensor), nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(u, ba), wrap(s, ba), wrap(vt, ba)))
}

/// Truncated SVD with bond dimension control.
pub fn trunc_svd<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
    params: &arnet_linalg::TruncSvdParams,
) -> Result<TruncSvdResult<T, B>, BackendError> {
    let (u, s, vt, err) =
        arnet_linalg::trunc_svd(tensor.backend(), dense(tensor), nrow, params)?;
    let ba = tensor.backend_arc();
    Ok((wrap(u, ba), wrap(s, ba), wrap(vt, ba), err))
}

/// Thin QR: A = Q * R.
pub fn qr<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<QrResult<T, B>, BackendError> {
    let (q, r) = arnet_linalg::qr(tensor.backend(), dense(tensor), nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(q, ba), wrap(r, ba)))
}

/// Thin LQ: A = L * Q.
pub fn lq<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<LqResult<T, B>, BackendError> {
    let (l, q) = arnet_linalg::lq(tensor.backend(), dense(tensor), nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(l, ba), wrap(q, ba)))
}

// ============================================================================
// Eigenvalue
// ============================================================================

/// Self-adjoint eigenvalue decomposition.
pub fn eigh<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<EighResult<T, B>, BackendError> {
    let (w, v) = arnet_linalg::eigh(tensor.backend(), dense(tensor), nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(w, ba), wrap(v, ba)))
}

/// Eigenvalues of a self-adjoint matrix.
pub fn eigvalsh<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<Tensor<T::Real, B>, BackendError> {
    let w = arnet_linalg::eigvalsh(tensor.backend(), dense(tensor), nrow)?;
    Ok(wrap(w, tensor.backend_arc()))
}

/// General eigenvalue decomposition.
pub fn eig<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<EigResult<T, B>, BackendError> {
    let (w, v) = arnet_linalg::eig(tensor.backend(), dense(tensor), nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(w, ba), wrap(v, ba)))
}

/// Eigenvalues of a general matrix.
pub fn eigvals<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<Tensor<T::Complex, B>, BackendError> {
    let w = arnet_linalg::eigvals(tensor.backend(), dense(tensor), nrow)?;
    Ok(wrap(w, tensor.backend_arc()))
}

// ============================================================================
// Matrix exponential
// ============================================================================

/// Matrix exponential for Hermitian matrices (eigh-based).
pub fn expm_hermitian<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<Tensor<T, B>, BackendError> {
    let result = arnet_linalg::expm_hermitian(tensor.backend(), dense(tensor), nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// Matrix exponential for anti-Hermitian matrices (eigh-based).
pub fn expm_antihermitian<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<Tensor<T, B>, BackendError> {
    let result = arnet_linalg::expm_antihermitian(tensor.backend(), dense(tensor), nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// General matrix exponential (Padé approximation).
pub fn expm<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<Tensor<T, B>, BackendError> {
    let result = arnet_linalg::expm(tensor.backend(), dense(tensor), nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

// ============================================================================
// Linear solve
// ============================================================================

/// Solve AX = B via LU decomposition.
pub fn solve<T: Scalar, B: ComputeBackend>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    nrow_a: usize,
) -> Result<Tensor<T, B>, BackendError> {
    let result = arnet_linalg::solve(a.backend(), dense(a), dense(b), nrow_a)?;
    Ok(wrap(result, a.backend_arc()))
}

/// Matrix inverse via LU decomposition.
pub fn inverse<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    nrow: usize,
) -> Result<Tensor<T, B>, BackendError> {
    let result = arnet_linalg::inverse(tensor.backend(), dense(tensor), nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

// ============================================================================
// Scalar operations (backend-free, but return Tensor)
// ============================================================================

/// Scale tensor by a scalar factor (out-of-place).
pub fn scale<T, B: ComputeBackend>(tensor: &Tensor<T, B>, factor: T) -> Tensor<T, B>
where
    T: Clone + Mul<Output = T>,
{
    let result = arnet_linalg::scale(dense(tensor), factor);
    wrap(result, tensor.backend_arc())
}

/// Frobenius norm.
pub fn norm<T: Scalar, B: ComputeBackend>(tensor: &Tensor<T, B>) -> T::Real {
    arnet_linalg::norm(dense(tensor))
}

/// Normalize to unit norm (out-of-place).
pub fn normalize<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
) -> (Tensor<T, B>, T::Real) {
    let (result, n) = arnet_linalg::normalize(dense(tensor));
    (wrap(result, tensor.backend_arc()), n)
}

/// Partial trace over bond index pairs.
pub fn trace<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
    pairs: &[(usize, usize)],
) -> Result<Tensor<T, B>, String> {
    let result = arnet_linalg::trace(dense(tensor), pairs)?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// Diagonal extraction (2D → 1D) or construction (1D → 2D).
pub fn diag<T: Scalar, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
) -> Result<Tensor<T, B>, String> {
    let result = arnet_linalg::diag(dense(tensor))?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// Linear combination of tensors.
pub fn linear_combine<T, B: ComputeBackend>(
    tensors: &[&Tensor<T, B>],
    coefs: &[T],
) -> Result<Tensor<T, B>, String>
where
    T: Clone + Zero + std::ops::Add<Output = T> + Mul<Output = T>,
{
    assert!(!tensors.is_empty(), "Cannot combine empty tensor list");
    let dense_refs: Vec<&DenseTensor<T>> = tensors.iter().map(|t| dense(t)).collect();
    let result = arnet_linalg::linear_combine(&dense_refs, coefs)?;
    Ok(wrap(result, tensors[0].backend_arc()))
}
