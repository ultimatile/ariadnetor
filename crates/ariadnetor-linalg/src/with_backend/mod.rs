//! Explicit-backend operation paths for dense tensors.
//!
//! Each function here is the call-site-backend counterpart of a legacy
//! tensor-derived operation: the backend is supplied as an argument and the
//! tensor's own backend is never consulted. The backend is taken as
//! `&Arc<B>` rather than `&B` because the Stage A result types still carry
//! `Arc<B>` and [`ComputeBackend`] does not require `Clone`, so an owned
//! handle is needed to wrap each result. These paths delegate to the same
//! `ComputeBackend`-bounded kernels the legacy wrappers use and tighten no
//! bound: capability enforcement (`OpsFor`) is layered on later.

use std::ops::Mul;
use std::sync::Arc;

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

/// Explicit-backend counterpart of [`crate::svd`].
pub fn svd_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<SvdResult<T, B>, LinalgError> {
    let (u, s, vt) = svd_dense(&**backend, tensor.data(), nrow)?;
    Ok((
        DenseTensor::with_backend(u, backend.clone()),
        DenseTensor::with_backend(s, backend.clone()),
        DenseTensor::with_backend(vt, backend.clone()),
    ))
}

/// Explicit-backend counterpart of [`crate::trunc_svd`].
pub fn trunc_svd_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<TruncSvdResult<T, B>, LinalgError> {
    let (u, s, vt, err) = trunc_svd_dense(&**backend, tensor.data(), nrow, params)?;
    Ok((
        DenseTensor::with_backend(u, backend.clone()),
        DenseTensor::with_backend(s, backend.clone()),
        DenseTensor::with_backend(vt, backend.clone()),
        err,
    ))
}

/// Explicit-backend counterpart of [`crate::qr`].
pub fn qr_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<QrResult<T, B>, LinalgError> {
    let (q, r) = qr_dense(&**backend, tensor.data(), nrow)?;
    Ok((
        DenseTensor::with_backend(q, backend.clone()),
        DenseTensor::with_backend(r, backend.clone()),
    ))
}

/// Explicit-backend counterpart of [`crate::lq`].
pub fn lq_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<LqResult<T, B>, LinalgError> {
    let (l, q) = lq_dense(&**backend, tensor.data(), nrow)?;
    Ok((
        DenseTensor::with_backend(l, backend.clone()),
        DenseTensor::with_backend(q, backend.clone()),
    ))
}

/// Explicit-backend counterpart of [`crate::eigh`].
pub fn eigh_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<EighResult<T, B>, LinalgError> {
    let (w, v) = eigh_dense(&**backend, tensor.data(), nrow)?;
    Ok((
        DenseTensor::with_backend(w, backend.clone()),
        DenseTensor::with_backend(v, backend.clone()),
    ))
}

/// Explicit-backend counterpart of [`crate::eigvalsh`].
pub fn eigvalsh_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<DenseTensor<T::Real, B>, LinalgError> {
    let (w, _v) = eigh_with_backend(backend, tensor, nrow)?;
    Ok(w)
}

/// Explicit-backend counterpart of [`crate::eig`].
pub fn eig_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<EigResult<T, B>, LinalgError> {
    let (w, v) = eig_dense(&**backend, tensor.data(), nrow)?;
    Ok((
        DenseTensor::with_backend(w, backend.clone()),
        DenseTensor::with_backend(v, backend.clone()),
    ))
}

/// Explicit-backend counterpart of [`crate::eigvals`].
pub fn eigvals_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<DenseTensor<T::Complex, B>, LinalgError> {
    let (w, _v) = eig_with_backend(backend, tensor, nrow)?;
    Ok(w)
}

/// Explicit-backend counterpart of [`crate::contract`].
pub fn contract_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    lhs: &DenseTensor<T, B>,
    rhs: &DenseTensor<T, B>,
    notation: &str,
) -> Result<DenseTensor<T, B>, LinalgError> {
    let result = contract_dense(&**backend, lhs.data(), rhs.data(), notation)?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::einsum`].
pub fn einsum_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensors: &[&DenseTensor<T, B>],
    notation: &str,
) -> Result<DenseTensor<T, B>, LinalgError> {
    if tensors.is_empty() {
        return Err(LinalgError::InvalidArgument(
            "einsum requires at least 1 input".to_string(),
        ));
    }
    let data_refs: Vec<&DenseTensorData<T>> = tensors.iter().map(|t| t.data()).collect();
    let result = einsum_dense(&**backend, &data_refs, notation)?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::transpose`].
pub fn transpose_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    perm: &[usize],
) -> Result<DenseTensor<T, B>, LinalgError> {
    let result = transpose_dense(&**backend, tensor.data(), perm)?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::trace`].
pub fn trace_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    pairs: &[(usize, usize)],
) -> Result<DenseTensor<T, B>, LinalgError> {
    let result = trace_dense(tensor.data(), pairs)?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::diag`].
pub fn diag_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
) -> Result<DenseTensor<T, B>, LinalgError> {
    let result = diag_dense(tensor.data())?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::diagonal_scale`].
pub fn diagonal_scale_with_backend<T, S, B>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    weights: &[S],
    axis: usize,
) -> Result<DenseTensor<T, B>, LinalgError>
where
    T: Clone + Mul<S, Output = T> + 'static,
    S: Clone,
    B: ComputeBackend,
{
    let result = diagonal_scale_dense(&**backend, tensor.data(), weights, axis)?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::solve`].
pub fn solve_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    a: &DenseTensor<T, B>,
    b: &DenseTensor<T, B>,
    nrow_a: usize,
) -> Result<DenseTensor<T, B>, LinalgError> {
    let result = solve_dense(&**backend, a.data(), b.data(), nrow_a)?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::inverse`].
pub fn inverse_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<DenseTensor<T, B>, LinalgError> {
    let result = inverse_dense(&**backend, tensor.data(), nrow)?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::expm`].
pub fn expm_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<DenseTensor<T, B>, LinalgError> {
    let result = expm_dense(&**backend, tensor.data(), nrow)?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::expm_hermitian`].
pub fn expm_hermitian_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<DenseTensor<T, B>, LinalgError> {
    let result = expm_hermitian_dense(&**backend, tensor.data(), nrow)?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}

/// Explicit-backend counterpart of [`crate::expm_antihermitian`].
pub fn expm_antihermitian_with_backend<T: Scalar, B: ComputeBackend>(
    backend: &Arc<B>,
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<DenseTensor<T, B>, LinalgError> {
    let result = expm_antihermitian_dense(&**backend, tensor.data(), nrow)?;
    Ok(DenseTensor::with_backend(result, backend.clone()))
}
