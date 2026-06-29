//! Explicit-backend operation paths for dense tensors.
//!
//! Each function here takes the backend at the call site and never consults a
//! tensor's own backend — tensors no longer carry one. The backend is taken as
//! `&B`: results are built with [`DenseTensor::from_data`], which stores no
//! backend, so no owned handle is needed.
//!
//! The dense operation surface is gated by
//! [`OpsFor<DenseStorage<T>>`](arnet_tensor::OpsFor): a backend that has not
//! declared it operates on dense storage cannot be passed here. Internal
//! kernels stay `ComputeBackend`-bound; they are reachable only through this
//! gate. `diagonal_scale` dispatches over the tensor type via
//! [`LinalgScale`](crate::LinalgScale) and lives in `scale_dispatch`, not here.

use arnet_core::Scalar;
use arnet_tensor::{DenseStorage, DenseTensor, DenseTensorData, OpsFor};

use crate::eigen::{EigResult, EighResult, eig_dense, eigh_dense};
use crate::einsum::einsum_dense;
use crate::error::LinalgError;
use crate::expm::{expm_antihermitian_dense, expm_dense, expm_hermitian_dense};
use crate::scalar_ops::{diag_dense, trace_dense};
use crate::solve::{inverse_dense, solve_dense};
use crate::transpose::transpose_dense;

#[cfg(test)]
mod tests;

/// Self-adjoint eigenvalue decomposition, using the supplied backend.
pub fn eigh_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<EighResult<T>, LinalgError> {
    let (w, v) = eigh_dense(backend, tensor.data(), nrow)?;
    Ok((DenseTensor::from_data(w), DenseTensor::from_data(v)))
}

/// Eigenvalues-only self-adjoint decomposition, using the supplied backend.
pub fn eigvalsh_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T::Real>, LinalgError> {
    let (w, _v) = eigh_with_backend(backend, tensor, nrow)?;
    Ok(w)
}

/// General eigenvalue decomposition, using the supplied backend.
pub fn eig_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<EigResult<T>, LinalgError> {
    let (w, v) = eig_dense(backend, tensor.data(), nrow)?;
    Ok((DenseTensor::from_data(w), DenseTensor::from_data(v)))
}

/// Eigenvalues-only general decomposition, using the supplied backend.
pub fn eigvals_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T::Complex>, LinalgError> {
    let (w, _v) = eig_with_backend(backend, tensor, nrow)?;
    Ok(w)
}

/// N-input Einstein summation, using the supplied backend.
pub fn einsum_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
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

/// General axis permutation of a dense tensor, using the supplied backend.
pub fn permute_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
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
pub fn trace_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
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
pub fn diag_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    _backend: &B,
    tensor: &DenseTensor<T>,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = diag_dense(tensor.data())?;
    Ok(DenseTensor::from_data(result))
}

/// Linear solve `AX = B` via LU, using the supplied backend.
pub fn solve_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    a: &DenseTensor<T>,
    b: &DenseTensor<T>,
    nrow_a: usize,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = solve_dense(backend, a.data(), b.data(), nrow_a)?;
    Ok(DenseTensor::from_data(result))
}

/// Matrix inverse via LU, using the supplied backend.
pub fn inverse_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = inverse_dense(backend, tensor.data(), nrow)?;
    Ok(DenseTensor::from_data(result))
}

/// General matrix exponential, using the supplied backend.
pub fn expm_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = expm_dense(backend, tensor.data(), nrow)?;
    Ok(DenseTensor::from_data(result))
}

/// Hermitian matrix exponential, using the supplied backend.
pub fn expm_hermitian_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = expm_hermitian_dense(backend, tensor.data(), nrow)?;
    Ok(DenseTensor::from_data(result))
}

/// Anti-Hermitian matrix exponential, using the supplied backend.
pub fn expm_antihermitian_with_backend<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = expm_antihermitian_dense(backend, tensor.data(), nrow)?;
    Ok(DenseTensor::from_data(result))
}
