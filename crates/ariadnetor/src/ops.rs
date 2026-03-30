//! High-level free functions that extract the backend from Tensor.
//!
//! These wrap `arnet_linalg::*` by pulling the backend from the tensor arguments,
//! so users never need to pass `&backend` explicitly.

use std::ops::Mul;
use std::sync::Arc;

use arnet_core::backend::ComputeBackend;
use arnet_core::scalar::Scalar;
use arnet_linalg::LinalgError;
use arnet_tensor::Dense;
use num_traits::Zero;

use crate::{DiagTensor, Tensor};

// ============================================================================
// Result type aliases
// ============================================================================

/// SVD result: `(U, S, Vt)` where S is a [`DiagTensor`].
pub type SvdResult<S, B> = (
    Tensor<Dense<S>, B>,
    DiagTensor<<S as Scalar>::Real, B>,
    Tensor<Dense<S>, B>,
);

/// Truncated SVD result: `(U, S, Vt, truncation_error)` where S is a [`DiagTensor`].
pub type TruncSvdResult<S, B> = (
    Tensor<Dense<S>, B>,
    DiagTensor<<S as Scalar>::Real, B>,
    Tensor<Dense<S>, B>,
    <S as Scalar>::Real,
);

/// QR result: `(Q, R)`.
pub type QrResult<S, B> = (Tensor<Dense<S>, B>, Tensor<Dense<S>, B>);

/// LQ result: `(L, Q)`.
pub type LqResult<S, B> = (Tensor<Dense<S>, B>, Tensor<Dense<S>, B>);

/// Self-adjoint eigenvalue result: `(eigenvalues, eigenvectors)`.
pub type EighResult<S, B> = (Tensor<Dense<<S as Scalar>::Real>, B>, Tensor<Dense<S>, B>);

/// General eigenvalue result: `(eigenvalues, eigenvectors)` (complex).
pub type EigResult<S, B> = (
    Tensor<Dense<<S as Scalar>::Complex>, B>,
    Tensor<Dense<<S as Scalar>::Complex>, B>,
);

// ============================================================================
// Helpers
// ============================================================================

/// Wrap a Dense result back into a Tensor with the given backend.
fn wrap<S: Clone, B: ComputeBackend>(dense: Dense<S>, backend: &Arc<B>) -> Tensor<Dense<S>, B> {
    Tensor::with_backend(dense, Arc::clone(backend))
}

/// Wrap a 1D Dense as a DiagTensor with the given backend.
fn wrap_diag<S: Clone, B: ComputeBackend>(dense: Dense<S>, backend: &Arc<B>) -> DiagTensor<S, B> {
    DiagTensor::from_vec_with_backend(dense.data().to_vec(), Arc::clone(backend))
}

// ============================================================================
// Contraction / Einsum
// ============================================================================

/// Tensor contraction via einsum notation.
pub fn contract<S: Scalar, B: ComputeBackend>(
    lhs: &Tensor<Dense<S>, B>,
    rhs: &Tensor<Dense<S>, B>,
    notation: &str,
) -> Result<Tensor<Dense<S>, B>, LinalgError> {
    let result = arnet_linalg::contract(lhs.backend(), &lhs.storage, &rhs.storage, notation)?;
    Ok(wrap(result, lhs.backend_arc()))
}

/// Einsum over N tensors.
pub fn einsum<S: Scalar, B: ComputeBackend>(
    notation: &str,
    tensors: &[&Tensor<Dense<S>, B>],
) -> Result<Tensor<Dense<S>, B>, LinalgError> {
    assert!(!tensors.is_empty(), "einsum requires at least one tensor");
    let backend_arc = tensors[0].backend_arc();
    let dense_refs: Vec<&Dense<S>> = tensors.iter().map(|t| &t.storage).collect();
    let result = arnet_linalg::einsum(tensors[0].backend(), &dense_refs, notation)?;
    Ok(wrap(result, backend_arc))
}

// ============================================================================
// Transpose
// ============================================================================

/// Permute tensor axes.
pub fn transpose<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    perm: &[usize],
) -> Result<Tensor<Dense<S>, B>, LinalgError> {
    let result = arnet_linalg::transpose(tensor.backend(), &tensor.storage, perm)?;
    Ok(wrap(result, tensor.backend_arc()))
}

// ============================================================================
// Decompositions
// ============================================================================

/// Thin SVD: A = U * diag(S) * Vt.
///
/// Returns singular values as a [`DiagTensor`], making their diagonal nature explicit.
pub fn svd<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<SvdResult<S, B>, LinalgError> {
    let (u, s, vt) = arnet_linalg::svd(tensor.backend(), &tensor.storage, nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(u, ba), wrap_diag(s, ba), wrap(vt, ba)))
}

/// Truncated SVD with bond dimension control.
///
/// Returns singular values as a [`DiagTensor`], making their diagonal nature explicit.
pub fn trunc_svd<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
    params: &arnet_linalg::TruncSvdParams,
) -> Result<TruncSvdResult<S, B>, LinalgError> {
    let (u, s, vt, err) = arnet_linalg::trunc_svd(tensor.backend(), &tensor.storage, nrow, params)?;
    let ba = tensor.backend_arc();
    Ok((wrap(u, ba), wrap_diag(s, ba), wrap(vt, ba), err))
}

/// Thin QR: A = Q * R.
pub fn qr<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<QrResult<S, B>, LinalgError> {
    let (q, r) = arnet_linalg::qr(tensor.backend(), &tensor.storage, nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(q, ba), wrap(r, ba)))
}

/// Thin LQ: A = L * Q.
pub fn lq<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<LqResult<S, B>, LinalgError> {
    let (l, q) = arnet_linalg::lq(tensor.backend(), &tensor.storage, nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(l, ba), wrap(q, ba)))
}

// ============================================================================
// Eigenvalue
// ============================================================================

/// Self-adjoint eigenvalue decomposition.
pub fn eigh<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<EighResult<S, B>, LinalgError> {
    let (w, v) = arnet_linalg::eigh(tensor.backend(), &tensor.storage, nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(w, ba), wrap(v, ba)))
}

/// Eigenvalues of a self-adjoint matrix.
pub fn eigvalsh<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<Tensor<Dense<S::Real>, B>, LinalgError> {
    let w = arnet_linalg::eigvalsh(tensor.backend(), &tensor.storage, nrow)?;
    Ok(wrap(w, tensor.backend_arc()))
}

/// General eigenvalue decomposition.
pub fn eig<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<EigResult<S, B>, LinalgError> {
    let (w, v) = arnet_linalg::eig(tensor.backend(), &tensor.storage, nrow)?;
    let ba = tensor.backend_arc();
    Ok((wrap(w, ba), wrap(v, ba)))
}

/// Eigenvalues of a general matrix.
pub fn eigvals<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<Tensor<Dense<S::Complex>, B>, LinalgError> {
    let w = arnet_linalg::eigvals(tensor.backend(), &tensor.storage, nrow)?;
    Ok(wrap(w, tensor.backend_arc()))
}

// ============================================================================
// Matrix exponential
// ============================================================================

/// Matrix exponential for Hermitian matrices (eigh-based).
pub fn expm_hermitian<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<Tensor<Dense<S>, B>, LinalgError> {
    let result = arnet_linalg::expm_hermitian(tensor.backend(), &tensor.storage, nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// Matrix exponential for anti-Hermitian matrices (eigh-based).
pub fn expm_antihermitian<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<Tensor<Dense<S>, B>, LinalgError> {
    let result = arnet_linalg::expm_antihermitian(tensor.backend(), &tensor.storage, nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// General matrix exponential (Padé approximation).
pub fn expm<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<Tensor<Dense<S>, B>, LinalgError> {
    let result = arnet_linalg::expm(tensor.backend(), &tensor.storage, nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

// ============================================================================
// Linear solve
// ============================================================================

/// Solve AX = B via LU decomposition.
pub fn solve<S: Scalar, B: ComputeBackend>(
    a: &Tensor<Dense<S>, B>,
    b: &Tensor<Dense<S>, B>,
    nrow_a: usize,
) -> Result<Tensor<Dense<S>, B>, LinalgError> {
    let result = arnet_linalg::solve(a.backend(), &a.storage, &b.storage, nrow_a)?;
    Ok(wrap(result, a.backend_arc()))
}

/// Matrix inverse via LU decomposition.
pub fn inverse<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    nrow: usize,
) -> Result<Tensor<Dense<S>, B>, LinalgError> {
    let result = arnet_linalg::inverse(tensor.backend(), &tensor.storage, nrow)?;
    Ok(wrap(result, tensor.backend_arc()))
}

// ============================================================================
// Scalar operations (backend-free, but return Tensor)
// ============================================================================

/// Scale tensor by a scalar factor (out-of-place).
pub fn scale<S, B: ComputeBackend>(tensor: &Tensor<Dense<S>, B>, factor: S) -> Tensor<Dense<S>, B>
where
    S: Clone + Mul<Output = S>,
{
    let result = arnet_linalg::scale(&tensor.storage, factor);
    wrap(result, tensor.backend_arc())
}

/// Frobenius norm.
pub fn norm<S: Scalar, B: ComputeBackend>(tensor: &Tensor<Dense<S>, B>) -> S::Real {
    arnet_linalg::norm(&tensor.storage)
}

/// Normalize to unit norm (out-of-place).
pub fn normalize<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
) -> (Tensor<Dense<S>, B>, S::Real) {
    let (result, n) = arnet_linalg::normalize(&tensor.storage);
    (wrap(result, tensor.backend_arc()), n)
}

/// Partial trace over bond index pairs.
pub fn trace<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
    pairs: &[(usize, usize)],
) -> Result<Tensor<Dense<S>, B>, String> {
    let result = arnet_linalg::trace(&tensor.storage, pairs)?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// Diagonal extraction (2D → 1D) or construction (1D → 2D).
pub fn diag<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
) -> Result<Tensor<Dense<S>, B>, String> {
    let result = arnet_linalg::diag(&tensor.storage)?;
    Ok(wrap(result, tensor.backend_arc()))
}

/// Extract the diagonal of a square matrix as a [`DiagTensor`].
///
/// Unlike [`diag`] which returns a plain `Tensor`, this preserves
/// the diagonal semantics in the type system.
pub fn diag_extract<S: Scalar, B: ComputeBackend>(
    tensor: &Tensor<Dense<S>, B>,
) -> Result<crate::DiagTensor<S, B>, String> {
    crate::DiagTensor::from_matrix(tensor)
}

/// Linear combination of tensors.
pub fn linear_combine<S, B: ComputeBackend>(
    tensors: &[&Tensor<Dense<S>, B>],
    coefs: &[S],
) -> Result<Tensor<Dense<S>, B>, String>
where
    S: Clone + Zero + std::ops::Add<Output = S> + Mul<Output = S>,
{
    assert!(!tensors.is_empty(), "Cannot combine empty tensor list");
    let dense_refs: Vec<&Dense<S>> = tensors.iter().map(|t| &t.storage).collect();
    let result = arnet_linalg::linear_combine(&dense_refs, coefs)?;
    Ok(wrap(result, tensors[0].backend_arc()))
}
