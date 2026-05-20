//! High-level free functions that extract the backend from Tensor.
//!
//! These wrap `arnet_linalg::*` by pulling the backend from the tensor
//! arguments, so users never need to pass `&backend` explicitly. The
//! `bridge_in` / `bridge_out` helpers copy each `DenseTensorData<T>`
//! into a `Dense<T>` and back, because `arnet_linalg` is the
//! authoritative consumer of `Dense<T>`.

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::LinalgError;
use arnet_tensor::{Dense, DenseTensor, DenseTensorData, Tensor};

// ============================================================================
// Result type aliases
// ============================================================================

/// QR result: `(Q, R)`.
pub type QrResult<S, B> = (DenseTensor<S, B>, DenseTensor<S, B>);

/// LQ result: `(L, Q)`.
pub type LqResult<S, B> = (DenseTensor<S, B>, DenseTensor<S, B>);

/// SVD result: `(U, S, Vt)` with singular values as a real-valued
/// 1D `DenseTensor`.
pub type SvdResult<S, B> = (
    DenseTensor<S, B>,
    DenseTensor<<S as Scalar>::Real, B>,
    DenseTensor<S, B>,
);

/// Truncated SVD result: `(U, S, Vt, truncation_error)`.
pub type TruncSvdResult<S, B> = (
    DenseTensor<S, B>,
    DenseTensor<<S as Scalar>::Real, B>,
    DenseTensor<S, B>,
    <S as Scalar>::Real,
);

/// Self-adjoint eigenvalue result: `(eigenvalues, eigenvectors)`.
pub type EighResult<S, B> = (DenseTensor<<S as Scalar>::Real, B>, DenseTensor<S, B>);

/// General eigenvalue result: `(eigenvalues, eigenvectors)` (complex).
pub type EigResult<S, B> = (
    DenseTensor<<S as Scalar>::Complex, B>,
    DenseTensor<<S as Scalar>::Complex, B>,
);

// ============================================================================
// Bridging helpers
// ============================================================================

/// Build a `Dense<S>` from a `DenseTensorData<S>` for passing to
/// `arnet_linalg`. The element buffer is copied.
fn bridge_in<S: Scalar>(td: &DenseTensorData<S>) -> Dense<S> {
    Dense::new(td.data().to_vec(), td.shape().to_vec(), td.order())
}

/// Wrap a linalg-produced `Dense<S>` back into a `DenseTensor<S, B>`.
fn bridge_out<S: Scalar, B: ComputeBackend>(
    dense: Dense<S>,
    backend: &Arc<B>,
) -> DenseTensor<S, B> {
    let shape = dense.shape().to_vec();
    let order = dense.order();
    let data = dense.data().to_vec();
    let td = DenseTensorData::from_raw_parts(data, shape, order);
    Tensor::with_backend(td, Arc::clone(backend))
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
    let lhs_d = bridge_in(lhs.data());
    let rhs_d = bridge_in(rhs.data());
    let result = arnet_linalg::contract(lhs.backend(), &lhs_d, &rhs_d, notation)?;
    Ok(bridge_out(result, lhs.backend_arc()))
}

/// Einsum over N tensors.
pub fn einsum<S: Scalar, B: ComputeBackend>(
    notation: &str,
    tensors: &[&DenseTensor<S, B>],
) -> Result<DenseTensor<S, B>, LinalgError> {
    assert!(!tensors.is_empty(), "einsum requires at least one tensor");
    let backend_arc = tensors[0].backend_arc();
    let owned: Vec<Dense<S>> = tensors.iter().map(|t| bridge_in(t.data())).collect();
    let refs: Vec<&Dense<S>> = owned.iter().collect();
    let result = arnet_linalg::einsum(tensors[0].backend(), &refs, notation)?;
    Ok(bridge_out(result, backend_arc))
}

// ============================================================================
// Transpose
// ============================================================================

/// Permute tensor axes.
pub fn transpose<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    perm: &[usize],
) -> Result<DenseTensor<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let result = arnet_linalg::transpose(tensor.backend(), &d, perm)?;
    Ok(bridge_out(result, tensor.backend_arc()))
}

// ============================================================================
// Decompositions
// ============================================================================

/// Thin SVD: A = U · diag(S) · Vt. Singular values are returned as a
/// real-valued 1D `DenseTensor`.
pub fn svd<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<SvdResult<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let (u, s, vt) = arnet_linalg::svd(tensor.backend(), &d, nrow)?;
    let ba = tensor.backend_arc();
    Ok((bridge_out(u, ba), bridge_out(s, ba), bridge_out(vt, ba)))
}

/// Truncated SVD with bond dimension control. Returns `(U, S, Vt,
/// truncation_error)`.
pub fn trunc_svd<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
    params: &arnet_linalg::TruncSvdParams,
) -> Result<TruncSvdResult<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let (u, s, vt, err) = arnet_linalg::trunc_svd(tensor.backend(), &d, nrow, params)?;
    let ba = tensor.backend_arc();
    Ok((
        bridge_out(u, ba),
        bridge_out(s, ba),
        bridge_out(vt, ba),
        err,
    ))
}

/// Thin QR: A = Q * R.
pub fn qr<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<QrResult<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let (q, r) = arnet_linalg::qr(tensor.backend(), &d, nrow)?;
    let ba = tensor.backend_arc();
    Ok((bridge_out(q, ba), bridge_out(r, ba)))
}

/// Thin LQ: A = L * Q.
pub fn lq<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<LqResult<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let (l, q) = arnet_linalg::lq(tensor.backend(), &d, nrow)?;
    let ba = tensor.backend_arc();
    Ok((bridge_out(l, ba), bridge_out(q, ba)))
}

// ============================================================================
// Eigenvalue
// ============================================================================

/// Self-adjoint eigenvalue decomposition.
pub fn eigh<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<EighResult<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let (w, v) = arnet_linalg::eigh(tensor.backend(), &d, nrow)?;
    let ba = tensor.backend_arc();
    Ok((bridge_out(w, ba), bridge_out(v, ba)))
}

/// Eigenvalues of a self-adjoint matrix.
pub fn eigvalsh<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S::Real, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let w = arnet_linalg::eigvalsh(tensor.backend(), &d, nrow)?;
    Ok(bridge_out(w, tensor.backend_arc()))
}

/// General eigenvalue decomposition.
pub fn eig<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<EigResult<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let (w, v) = arnet_linalg::eig(tensor.backend(), &d, nrow)?;
    let ba = tensor.backend_arc();
    Ok((bridge_out(w, ba), bridge_out(v, ba)))
}

/// Eigenvalues of a general matrix.
pub fn eigvals<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S::Complex, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let w = arnet_linalg::eigvals(tensor.backend(), &d, nrow)?;
    Ok(bridge_out(w, tensor.backend_arc()))
}

// ============================================================================
// Matrix exponential
// ============================================================================

/// Matrix exponential for Hermitian matrices (eigh-based).
pub fn expm_hermitian<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let result = arnet_linalg::expm_hermitian(tensor.backend(), &d, nrow)?;
    Ok(bridge_out(result, tensor.backend_arc()))
}

/// Matrix exponential for anti-Hermitian matrices (eigh-based).
pub fn expm_antihermitian<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let result = arnet_linalg::expm_antihermitian(tensor.backend(), &d, nrow)?;
    Ok(bridge_out(result, tensor.backend_arc()))
}

/// General matrix exponential (Padé approximation).
pub fn expm<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let result = arnet_linalg::expm(tensor.backend(), &d, nrow)?;
    Ok(bridge_out(result, tensor.backend_arc()))
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
    let a_d = bridge_in(a.data());
    let b_d = bridge_in(b.data());
    let result = arnet_linalg::solve(a.backend(), &a_d, &b_d, nrow_a)?;
    Ok(bridge_out(result, a.backend_arc()))
}

/// Matrix inverse via LU decomposition.
pub fn inverse<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    nrow: usize,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let result = arnet_linalg::inverse(tensor.backend(), &d, nrow)?;
    Ok(bridge_out(result, tensor.backend_arc()))
}

// ============================================================================
// Scalar operations (backend-free, but return Tensor)
// ============================================================================

/// Scale tensor by a scalar factor (out-of-place).
pub fn scale<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    factor: S,
) -> DenseTensor<S, B> {
    let d = bridge_in(tensor.data());
    let result = arnet_linalg::scale(&d, factor);
    bridge_out(result, tensor.backend_arc())
}

/// Frobenius norm.
pub fn norm<S: Scalar, B: ComputeBackend>(tensor: &DenseTensor<S, B>) -> S::Real {
    let d = bridge_in(tensor.data());
    arnet_linalg::norm(&d)
}

/// Normalize to unit norm (out-of-place).
pub fn normalize<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
) -> (DenseTensor<S, B>, S::Real) {
    let d = bridge_in(tensor.data());
    let (result, n) = arnet_linalg::normalize(&d);
    (bridge_out(result, tensor.backend_arc()), n)
}

/// Partial trace over bond index pairs.
pub fn trace<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
    pairs: &[(usize, usize)],
) -> Result<DenseTensor<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let result = arnet_linalg::trace(&d, pairs)?;
    Ok(bridge_out(result, tensor.backend_arc()))
}

/// Diagonal extraction (2D → 1D) or construction (1D → 2D). Returns
/// a `DenseTensor`; rank-1 outputs encode the diagonal as a plain
/// 1-D tensor.
pub fn diag<S: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<S, B>,
) -> Result<DenseTensor<S, B>, LinalgError> {
    let d = bridge_in(tensor.data());
    let result = arnet_linalg::diag(&d)?;
    Ok(bridge_out(result, tensor.backend_arc()))
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
    let owned: Vec<Dense<S>> = tensors.iter().map(|t| bridge_in(t.data())).collect();
    let refs: Vec<&Dense<S>> = owned.iter().collect();
    let result = arnet_linalg::linear_combine(&refs, coefs)?;
    Ok(bridge_out(result, tensors[0].backend_arc()))
}
