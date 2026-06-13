use num_traits::{One, Zero};
use std::ops::{Add, Mul};

use crate::{DenseTensor, DenseTensorData, TensorError};

/// Linear combination of dense tensors: `Σ_i coefs[i] * tensors[i]`.
///
/// All inputs must share the same shape and memory order.
///
/// # Errors
///
/// Returns [`TensorError::InvalidArgument`] if the list is empty, the
/// tensor and coefficient counts differ, or any tensor's shape or
/// memory order differs from `tensors[0]`'s.
pub fn linear_combine<T>(
    tensors: &[&DenseTensor<T>],
    coefs: &[T],
) -> Result<DenseTensor<T>, TensorError>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
{
    if tensors.is_empty() {
        return Err(TensorError::InvalidArgument(
            "Cannot combine empty tensor list".to_string(),
        ));
    }
    let data_refs: Vec<&DenseTensorData<T>> = tensors.iter().map(|t| t.data()).collect();
    let result = DenseTensorData::linear_combine(&data_refs, coefs)?;
    Ok(DenseTensor::from_data(result))
}

/// Sum of dense tensors (all coefficients = 1).
///
/// Equivalent to `linear_combine(tensors, &[T::one(); tensors.len()])`.
pub fn add_all<T>(tensors: &[&DenseTensor<T>]) -> Result<DenseTensor<T>, TensorError>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    let coefs = vec![T::one(); tensors.len()];
    linear_combine(tensors, &coefs)
}
