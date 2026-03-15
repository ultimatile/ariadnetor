use arnet_core::backend::{BackendError, ComputeBackend, TransposeDescriptor};
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;

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
        return Ok(DenseTensor::from_data(vec![], new_shape));
    }

    let mut output = vec![T::zero(); total];

    let desc = TransposeDescriptor {
        input: tensor.data(),
        output: &mut output,
        shape: tensor.shape(),
        perm,
    };

    backend.transpose(desc)?;

    Ok(DenseTensor::from_data(output, new_shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arnet_cpu::CpuBackend;

    #[test]
    fn test_transpose_f64_2d() {
        let backend = CpuBackend::new();
        let tensor =
            DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_f64_3d() {
        let backend = CpuBackend::new();
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

        let result = transpose(&backend, &tensor, &[2, 0, 1]).unwrap();

        assert_eq!(result.shape(), &[4, 2, 3]);
        assert_eq!(result.len(), 24);
        // input[0][0][0] = 0 → output[0][0][0]
        assert_eq!(result.get(&[0, 0, 0]), 0.0);
        // input[0][0][1] = 1 → output[1][0][0]
        assert_eq!(result.get(&[1, 0, 0]), 1.0);
    }

    #[test]
    fn test_transpose_f32_2d() {
        let backend = CpuBackend::new();
        let tensor =
            DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_complex_f64_2d() {
        use num_complex::Complex;

        let backend = CpuBackend::new();
        let input = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
            Complex::new(9.0, 10.0),
            Complex::new(11.0, 12.0),
        ];
        let tensor = DenseTensor::from_data(input, vec![2, 3]);

        let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.get(&[0, 0]), Complex::new(1.0, 2.0));
        assert_eq!(result.get(&[0, 1]), Complex::new(7.0, 8.0));
        assert_eq!(result.get(&[1, 0]), Complex::new(3.0, 4.0));
        assert_eq!(result.get(&[1, 1]), Complex::new(9.0, 10.0));
    }

    #[test]
    fn test_transpose_empty_tensor() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![], vec![0, 3]);

        let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

        assert_eq!(result.shape(), &[3, 0]);
        assert_eq!(result.len(), 0);
    }
}
