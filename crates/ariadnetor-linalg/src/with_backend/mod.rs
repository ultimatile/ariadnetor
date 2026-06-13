//! Explicit-backend operation paths for dense tensors.
//!
//! Each function here takes the backend at the call site and never consults a
//! tensor's own backend — tensors no longer carry one. The backend is taken as
//! `&B`: results are built with [`DenseTensor::from_data`], which stores no
//! backend, so no owned handle is needed. These paths delegate to the same
//! `ComputeBackend`-bounded kernels and tighten no bound; capability
//! enforcement (`OpsFor`) is layered on later.

use std::ops::Mul;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_tensor::{DenseTensor, DenseTensorData};

use crate::decomposition::{
    LqResult, QrResult, SvdResult, TruncSvdParams, TruncSvdResult, lq_dense, qr_dense, svd_dense,
    trunc_svd_dense,
};
use crate::eigen::{EigResult, EighResult, eig_dense, eigh_dense};
use crate::error::LinalgError;
use crate::expm::{expm_antihermitian_dense, expm_dense, expm_hermitian_dense};
use crate::scalar_ops::{diag_dense, diagonal_scale_dense, trace_dense};
use crate::solve::{inverse_dense, solve_dense};
use crate::transpose::transpose_dense;
use crate::{contract::contract_dense, einsum::einsum_dense};

#[cfg(test)]
mod tests;

/// Thin SVD of a tensor reshaped as a matrix, using the supplied backend.
pub fn svd_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<SvdResult<T>, LinalgError> {
    let (u, s, vt) = svd_dense(backend, tensor.data(), nrow)?;
    Ok((
        DenseTensor::from_data(u),
        DenseTensor::from_data(s),
        DenseTensor::from_data(vt),
    ))
}

/// Truncated SVD of a tensor reshaped as a matrix, using the supplied backend.
pub fn trunc_svd_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<TruncSvdResult<T>, LinalgError> {
    let (u, s, vt, err) = trunc_svd_dense(backend, tensor.data(), nrow, params)?;
    Ok((
        DenseTensor::from_data(u),
        DenseTensor::from_data(s),
        DenseTensor::from_data(vt),
        err,
    ))
}

/// Thin QR of a tensor reshaped as a matrix, using the supplied backend.
pub fn qr_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<QrResult<T>, LinalgError> {
    let (q, r) = qr_dense(backend, tensor.data(), nrow)?;
    Ok((DenseTensor::from_data(q), DenseTensor::from_data(r)))
}

/// Thin LQ of a tensor reshaped as a matrix, using the supplied backend.
pub fn lq_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<LqResult<T>, LinalgError> {
    let (l, q) = lq_dense(backend, tensor.data(), nrow)?;
    Ok((DenseTensor::from_data(l), DenseTensor::from_data(q)))
}

/// Self-adjoint eigenvalue decomposition, using the supplied backend.
pub fn eigh_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<EighResult<T>, LinalgError> {
    let (w, v) = eigh_dense(backend, tensor.data(), nrow)?;
    Ok((DenseTensor::from_data(w), DenseTensor::from_data(v)))
}

/// Eigenvalues-only self-adjoint decomposition, using the supplied backend.
pub fn eigvalsh_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T::Real>, LinalgError> {
    let (w, _v) = eigh_with_backend(backend, tensor, nrow)?;
    Ok(w)
}

/// General eigenvalue decomposition, using the supplied backend.
pub fn eig_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<EigResult<T>, LinalgError> {
    let (w, v) = eig_dense(backend, tensor.data(), nrow)?;
    Ok((DenseTensor::from_data(w), DenseTensor::from_data(v)))
}

/// Eigenvalues-only general decomposition, using the supplied backend.
pub fn eigvals_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T::Complex>, LinalgError> {
    let (w, _v) = eig_with_backend(backend, tensor, nrow)?;
    Ok(w)
}

/// Pure tensor contraction of two operands, using the supplied backend.
pub fn contract_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    lhs: &DenseTensor<T>,
    rhs: &DenseTensor<T>,
    notation: &str,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = contract_dense(backend, lhs.data(), rhs.data(), notation)?;
    Ok(DenseTensor::from_data(result))
}

/// N-input Einstein summation, using the supplied backend.
pub fn einsum_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensors: &[&DenseTensor<T>],
    notation: &str,
) -> Result<DenseTensor<T>, LinalgError> {
    if tensors.is_empty() {
        return Err(LinalgError::InvalidArgument(
            "einsum requires at least 1 input".to_string(),
        ));
    }
    let data_refs: Vec<&DenseTensorData<T>> = tensors.iter().map(|t| t.data()).collect();
    let result = einsum_dense(backend, &data_refs, notation)?;
    Ok(DenseTensor::from_data(result))
}

/// Axis permutation (transpose) of a dense tensor, using the supplied backend.
pub fn transpose_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    perm: &[usize],
) -> Result<DenseTensor<T>, LinalgError> {
    let result = transpose_dense(backend, tensor.data(), perm)?;
    Ok(DenseTensor::from_data(result))
}

/// Partial trace over bond index pairs. The backend argument is accepted for
/// API uniformity with the other twins; the partial trace needs no kernel, so
/// it is unused here.
pub fn trace_with_backend<T: Scalar, B: ComputeBackend>(
    _backend: &B,
    tensor: &DenseTensor<T>,
    pairs: &[(usize, usize)],
) -> Result<DenseTensor<T>, LinalgError> {
    let result = trace_dense(tensor.data(), pairs)?;
    Ok(DenseTensor::from_data(result))
}

/// Diagonal extraction / construction. The backend argument is accepted for
/// API uniformity with the other twins; this operation needs no kernel, so it
/// is unused here.
pub fn diag_with_backend<T: Scalar, B: ComputeBackend>(
    _backend: &B,
    tensor: &DenseTensor<T>,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = diag_dense(tensor.data())?;
    Ok(DenseTensor::from_data(result))
}

/// Per-slice diagonal scaling along `axis`, using the supplied backend.
pub fn diagonal_scale_with_backend<T, S, B>(
    backend: &B,
    tensor: &DenseTensor<T>,
    weights: &[S],
    axis: usize,
) -> Result<DenseTensor<T>, LinalgError>
where
    T: Clone + Mul<S, Output = T> + 'static,
    S: Clone,
    B: ComputeBackend,
{
    let result = diagonal_scale_dense(backend, tensor.data(), weights, axis)?;
    Ok(DenseTensor::from_data(result))
}

/// Linear solve `AX = B` via LU, using the supplied backend.
pub fn solve_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    a: &DenseTensor<T>,
    b: &DenseTensor<T>,
    nrow_a: usize,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = solve_dense(backend, a.data(), b.data(), nrow_a)?;
    Ok(DenseTensor::from_data(result))
}

/// Matrix inverse via LU, using the supplied backend.
pub fn inverse_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = inverse_dense(backend, tensor.data(), nrow)?;
    Ok(DenseTensor::from_data(result))
}

/// General matrix exponential, using the supplied backend.
pub fn expm_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = expm_dense(backend, tensor.data(), nrow)?;
    Ok(DenseTensor::from_data(result))
}

/// Hermitian matrix exponential, using the supplied backend.
pub fn expm_hermitian_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = expm_hermitian_dense(backend, tensor.data(), nrow)?;
    Ok(DenseTensor::from_data(result))
}

/// Anti-Hermitian matrix exponential, using the supplied backend.
pub fn expm_antihermitian_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = expm_antihermitian_dense(backend, tensor.data(), nrow)?;
    Ok(DenseTensor::from_data(result))
}
