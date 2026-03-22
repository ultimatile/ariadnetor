use arnet_core::backend::{BackendError, ComputeBackend, TransposeDescriptor};
use arnet_core::scalar::Scalar;
use arnet_tensor::{DenseTensor, MemoryOrder};

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
/// Returns `BackendError` if the backend fails to execute the transpose.
pub fn transpose<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    perm: &[usize],
) -> Result<DenseTensor<T>, BackendError> {
    let new_shape: Vec<usize> = perm.iter().map(|&i| tensor.shape()[i]).collect();
    let total = tensor.len();

    if total == 0 {
        let empty = DenseTensor::from_data_with_order(vec![], new_shape, MemoryOrder::RowMajor);
        return Ok(empty.to_contiguous(backend.preferred_order()));
    }

    // Ensure row-major contiguous input for the transpose backend
    let rm = tensor.to_contiguous(MemoryOrder::RowMajor);
    let mut output = vec![T::zero(); total];

    let desc = TransposeDescriptor {
        input: rm.data(),
        output: &mut output,
        shape: tensor.shape(),
        perm,
    };

    backend.transpose(desc)?;

    let result = DenseTensor::from_data_with_order(output, new_shape, MemoryOrder::RowMajor);
    Ok(result.to_contiguous(backend.preferred_order()))
}
