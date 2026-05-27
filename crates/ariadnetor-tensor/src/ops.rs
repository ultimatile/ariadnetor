use num_traits::{One, Zero};
use std::ops::{Add, Mul};

use arnet_core::backend::ComputeBackend;

use crate::{DenseTensor, DenseTensorData, TensorError};

/// Linear combination of dense tensors: `Σ_i coefs[i] * tensors[i]`.
///
/// All inputs must share the same shape and memory order. The result
/// is wrapped against `tensors[0]`'s backend `Arc`; callers must ensure
/// every input shares the same backend `Arc` (a mismatch silently
/// labels the output with the first tensor's backend, which is wrong
/// for backends carrying state).
///
/// # Errors
///
/// Returns [`TensorError::InvalidArgument`] if the list is empty, the
/// tensor and coefficient counts differ, or any tensor's shape or
/// memory order differs from `tensors[0]`'s.
pub fn linear_combine<T, B>(
    tensors: &[&DenseTensor<T, B>],
    coefs: &[T],
) -> Result<DenseTensor<T, B>, TensorError>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
    B: ComputeBackend,
{
    if tensors.is_empty() {
        return Err(TensorError::InvalidArgument(
            "Cannot combine empty tensor list".to_string(),
        ));
    }
    let backend_arc = tensors[0].backend_arc().clone();
    let data_refs: Vec<&DenseTensorData<T>> = tensors.iter().map(|t| t.data()).collect();
    let result = DenseTensorData::linear_combine(&data_refs, coefs)?;
    Ok(DenseTensor::with_backend(result, backend_arc))
}

/// Sum of dense tensors (all coefficients = 1).
///
/// Equivalent to `linear_combine(tensors, &[T::one(); tensors.len()])`.
pub fn add_all<T, B>(tensors: &[&DenseTensor<T, B>]) -> Result<DenseTensor<T, B>, TensorError>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>,
    B: ComputeBackend,
{
    let coefs = vec![T::one(); tensors.len()];
    linear_combine(tensors, &coefs)
}
