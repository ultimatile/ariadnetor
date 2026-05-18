use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, TransposeDescriptor};
use arnet_tensor::{DenseTensorData, normalize_to};

use crate::error::LinalgError;

/// Transpose (permute axes) of a dense tensor using the provided backend.
///
/// # Arguments
///
/// * `backend` - Compute backend for the transpose operation
/// * `tensor` - Input tensor
/// * `perm` - Permutation of axes (e.g., `[1, 0]` transposes a 2D tensor)
///
/// # Errors
///
/// Returns `LinalgError` if the backend fails to execute the transpose.
pub fn transpose<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    perm: &[usize],
) -> Result<DenseTensorData<T>, LinalgError> {
    let policy = backend.par_for_transpose(tensor.shape());
    transpose_inner(backend, tensor, perm, false, policy)
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
pub fn conjugate_transpose<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    perm: &[usize],
) -> Result<DenseTensorData<T>, LinalgError> {
    let policy = backend.par_for_transpose(tensor.shape());
    transpose_inner(backend, tensor, perm, true, policy)
}

/// Transpose with caller-specified execution policy.
///
/// Expert-layer counterpart of [`transpose`]; the default wrapper consults
/// `backend.par_for_transpose`, while this entry point takes `policy`
/// directly.
pub fn transpose_with_policy<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    perm: &[usize],
    policy: ExecPolicy,
) -> Result<DenseTensorData<T>, LinalgError> {
    transpose_inner(backend, tensor, perm, false, policy)
}

/// Conjugate transpose with caller-specified execution policy.
///
/// Expert-layer counterpart of [`conjugate_transpose`].
pub fn conjugate_transpose_with_policy<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    perm: &[usize],
    policy: ExecPolicy,
) -> Result<DenseTensorData<T>, LinalgError> {
    transpose_inner(backend, tensor, perm, true, policy)
}

/// Shared implementation for transpose and conjugate transpose.
fn transpose_inner<T: Scalar>(
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

    // The backend kernel reads `input` under `order` semantics. If the
    // caller's tensor is laid out in a different order, normalize at
    // the boundary so the kernel sees the layout it expects.
    let input = normalize_to(tensor, order);

    let mut output = vec![T::zero(); total];

    let desc = TransposeDescriptor {
        input: input.data(),
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
