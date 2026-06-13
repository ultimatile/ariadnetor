use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, TransposeDescriptor};
use arnet_tensor::{DenseTensor, DenseTensorData, normalize_to_data};

use crate::error::LinalgError;

/// Transpose with an explicit backend and caller-specified execution policy.
///
/// Expert-layer counterpart of [`crate::transpose_with_backend`]; that entry
/// point consults `backend.par_for_transpose`, while this one takes `policy`
/// directly. The backend is supplied at the call site and the tensor's own
/// backend is never consulted.
pub fn transpose_with_policy<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    perm: &[usize],
    policy: ExecPolicy,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = transpose_inner(backend, tensor.data(), perm, false, policy)?;
    Ok(DenseTensor::from_data(result))
}

/// Crate-internal kernel shared by [`transpose_with_policy`] and the
/// explicit-backend [`crate::transpose_with_backend`] path.
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
