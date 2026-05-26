use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, TransposeDescriptor};
use arnet_tensor::{DenseTensor, DenseTensorData, normalize_to_data};

use crate::error::LinalgError;

/// Transpose (permute axes) of a dense tensor.
///
/// # Arguments
///
/// * `tensor` - Input tensor (backend flows from the tensor)
/// * `perm` - Permutation of axes (e.g., `[1, 0]` transposes a 2D tensor)
///
/// # Errors
///
/// Returns `LinalgError` if the backend fails to execute the transpose.
pub fn transpose<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    perm: &[usize],
) -> Result<DenseTensor<T, B>, LinalgError> {
    let policy = tensor.backend().par_for_transpose(tensor.shape());
    transpose_with_policy(tensor, perm, policy)
}

/// Conjugate transpose (permute axes + element-wise conjugation).
///
/// For real types the conjugation is a no-op, so this is equivalent to
/// [`transpose`]. For complex types each element is conjugated during
/// the permutation, fusing two passes into one.
///
/// # Errors
///
/// Returns `LinalgError` if the backend fails to execute the transpose.
pub fn conjugate_transpose<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    perm: &[usize],
) -> Result<DenseTensor<T, B>, LinalgError> {
    let policy = tensor.backend().par_for_transpose(tensor.shape());
    conjugate_transpose_with_policy(tensor, perm, policy)
}

/// Transpose with caller-specified execution policy.
///
/// Expert-layer counterpart of [`transpose`]; the default wrapper consults
/// `backend.par_for_transpose`, while this entry point takes `policy`
/// directly.
pub fn transpose_with_policy<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    perm: &[usize],
    policy: ExecPolicy,
) -> Result<DenseTensor<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let result = transpose_inner(tensor.backend(), tensor.data(), perm, false, policy)?;
    Ok(wrap(result, backend_arc))
}

/// Conjugate transpose with caller-specified execution policy.
///
/// Expert-layer counterpart of [`conjugate_transpose`].
pub fn conjugate_transpose_with_policy<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    perm: &[usize],
    policy: ExecPolicy,
) -> Result<DenseTensor<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let result = transpose_inner(tensor.backend(), tensor.data(), perm, true, policy)?;
    Ok(wrap(result, backend_arc))
}

/// Crate-internal kernel for [`transpose`] / [`transpose_with_policy`].
///
/// Self-tunes via `par_for_transpose` so other kernels (`contract`,
/// `einsum`) can reuse the transpose without re-paying the wrap /
/// unwrap cost. Operates directly on the joined [`DenseTensorData<T>`]
/// surface.
pub(crate) fn transpose_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    perm: &[usize],
) -> Result<DenseTensorData<T>, LinalgError> {
    let policy = backend.par_for_transpose(tensor.shape());
    transpose_inner(backend, tensor, perm, false, policy)
}

/// Crate-internal counterpart of [`transpose_dense`] that conjugates as
/// it permutes.
pub(crate) fn conjugate_transpose_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    perm: &[usize],
) -> Result<DenseTensorData<T>, LinalgError> {
    let policy = backend.par_for_transpose(tensor.shape());
    transpose_inner(backend, tensor, perm, true, policy)
}

/// Shared implementation for transpose and conjugate transpose.
///
/// Reads `tensor.data()` (via the storage half) under the backend's
/// preferred order; normalize_to_data inserts a reorder at the
/// boundary when the caller's tensor is in a different order.
pub(crate) fn transpose_inner<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    perm: &[usize],
    conj: bool,
    policy: ExecPolicy,
) -> Result<DenseTensorData<T>, LinalgError> {
    let order = backend.preferred_order();
    let new_shape: Vec<usize> = perm.iter().map(|&i| tensor.shape()[i]).collect();
    let total = tensor.len();

    if total == 0 {
        return Ok(DenseTensorData::from_raw_parts(vec![], new_shape, order));
    }

    let input = normalize_to_data(tensor, order);
    let mut output = vec![T::zero(); total];

    let desc = TransposeDescriptor {
        input: input.storage().data(),
        output: &mut output,
        shape: tensor.shape(),
        perm,
        order,
        conj,
        policy,
    };

    backend.transpose(desc)?;

    Ok(DenseTensorData::from_raw_parts(output, new_shape, order))
}

fn wrap<T, B>(data: DenseTensorData<T>, backend: Arc<B>) -> DenseTensor<T, B>
where
    T: Clone,
    B: ComputeBackend,
{
    DenseTensor::with_backend(data, backend)
}
