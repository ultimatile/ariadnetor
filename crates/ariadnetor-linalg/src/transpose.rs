use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, TransposeDescriptor};
use arnet_tensor::{Dense, DenseTensor, normalize_to};

use crate::error::LinalgError;
use crate::tensor_bridge::wrap_dense;

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
    let dense = tensor.data().as_dense();
    let result = transpose_inner(tensor.backend(), &dense, perm, false, policy)?;
    Ok(wrap_dense(result, backend_arc))
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
    let dense = tensor.data().as_dense();
    let result = transpose_inner(tensor.backend(), &dense, perm, true, policy)?;
    Ok(wrap_dense(result, backend_arc))
}

/// Crate-internal kernel for [`transpose`] / [`transpose_with_policy`] on
/// the legacy `Dense<T>` form. Self-tunes via `par_for_transpose` so other
/// kernels (`contract`, `einsum`) can reuse the transpose without re-paying
/// the wrap/unwrap cost.
pub(crate) fn transpose_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    perm: &[usize],
) -> Result<Dense<T>, LinalgError> {
    let policy = backend.par_for_transpose(tensor.shape());
    transpose_inner(backend, tensor, perm, false, policy)
}

/// Crate-internal counterpart of [`transpose_dense`] that conjugates as
/// it permutes.
pub(crate) fn conjugate_transpose_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    perm: &[usize],
) -> Result<Dense<T>, LinalgError> {
    let policy = backend.par_for_transpose(tensor.shape());
    transpose_inner(backend, tensor, perm, true, policy)
}

/// Shared implementation for transpose and conjugate transpose.
///
/// Crate-internal kernel: takes the legacy `Dense<T>` representation so
/// callsites inside the crate (e.g. `contract`, `einsum`) can reuse the
/// kernel without paying the wrap / unwrap cost twice.
pub(crate) fn transpose_inner<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    perm: &[usize],
    conj: bool,
    policy: ExecPolicy,
) -> Result<Dense<T>, LinalgError> {
    let order = backend.preferred_order();
    let new_shape: Vec<usize> = perm.iter().map(|&i| tensor.shape()[i]).collect();
    let total = tensor.len();

    if total == 0 {
        return Ok(Dense::new(vec![], new_shape, order));
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

    Ok(Dense::new(output, new_shape, order))
}
