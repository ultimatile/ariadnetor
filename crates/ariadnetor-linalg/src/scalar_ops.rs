use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;
use num_traits::{Float, One, Zero};
use std::ops::{Add, Mul};

/// Scale tensor by a scalar factor (out-of-place).
///
/// Returns a new tensor with all elements multiplied by `factor`.
///
/// # Examples
///
/// ```
/// use arnet_linalg::scale;
/// use arnet_tensor::DenseTensor;
///
/// let tensor = DenseTensor::<f64>::ones(vec![2, 3]);
/// let scaled = scale(&tensor, 2.5);
/// assert_eq!(scaled.get(&[0, 0]), 2.5);
/// ```
pub fn scale<T>(tensor: &DenseTensor<T>, factor: T) -> DenseTensor<T>
where
    T: Clone + Mul<Output = T>,
{
    let data: Vec<T> = tensor
        .data()
        .iter()
        .map(|x| x.clone() * factor.clone())
        .collect();
    DenseTensor::from_data(data, tensor.shape().to_vec())
}

/// Compute the Frobenius norm of a tensor.
///
/// Returns sqrt(sum |element|^2) as a real value.
///
/// # Examples
///
/// ```
/// use arnet_linalg::norm;
/// use arnet_tensor::DenseTensor;
///
/// let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
/// let n = norm(&tensor);
/// assert!((n - 2.0).abs() < 1e-10);
/// ```
pub fn norm<T: Scalar>(tensor: &DenseTensor<T>) -> T::Real {
    let sum_sq = tensor
        .data()
        .iter()
        .map(|&x| {
            let a = x.abs();
            a * a
        })
        .fold(T::Real::zero(), |acc, x| acc + x);
    sum_sq.sqrt()
}

/// Normalize a tensor to unit Frobenius norm (out-of-place).
///
/// Returns `(normalized_tensor, original_norm)`.
/// Panics if the tensor has zero norm.
///
/// # Examples
///
/// ```
/// use arnet_linalg::normalize;
/// use arnet_tensor::DenseTensor;
///
/// let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
/// let (normalized, n) = normalize(&tensor);
/// assert!((n - 2.0).abs() < 1e-10);
/// assert!((arnet_linalg::norm(&normalized) - 1.0).abs() < 1e-10);
/// ```
pub fn normalize<T: Scalar>(tensor: &DenseTensor<T>) -> (DenseTensor<T>, T::Real) {
    let n = norm(tensor);
    assert!(n != T::Real::zero(), "Cannot normalize zero tensor");
    let inv_norm = T::Real::one() / n;
    let data: Vec<T> = tensor
        .data()
        .iter()
        .map(|&x| x.scale_real(inv_norm))
        .collect();
    (DenseTensor::from_data(data, tensor.shape().to_vec()), n)
}

/// Linear combination of tensors: sum_i coefs[i] * tensors[i].
///
/// All tensors must have the same shape.
///
/// # Errors
///
/// Returns an error if tensors have different shapes, the list is empty,
/// or tensors and coefficients have different lengths.
///
/// # Examples
///
/// ```
/// use arnet_linalg::linear_combine;
/// use arnet_tensor::DenseTensor;
///
/// let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
/// let b = DenseTensor::<f64>::constant(vec![2, 2], 2.0);
///
/// // 2*a + 3*b = 2*1 + 3*2 = 8
/// let result = linear_combine(&[&a, &b], &[2.0, 3.0]).unwrap();
/// assert_eq!(result.get(&[0, 0]), 8.0);
/// ```
pub fn linear_combine<T>(
    tensors: &[&DenseTensor<T>],
    coefs: &[T],
) -> Result<DenseTensor<T>, String>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
{
    if tensors.is_empty() {
        return Err("Cannot combine empty tensor list".to_string());
    }
    if tensors.len() != coefs.len() {
        return Err(format!(
            "Mismatched lengths: {} tensors vs {} coefficients",
            tensors.len(),
            coefs.len()
        ));
    }
    let shape = tensors[0].shape();
    for t in &tensors[1..] {
        if t.shape() != shape {
            return Err("All tensors must have the same shape".to_string());
        }
    }
    let len = tensors[0].len();
    let mut result = vec![T::zero(); len];
    for (tensor, coef) in tensors.iter().zip(coefs) {
        for (r, val) in result.iter_mut().zip(tensor.data()) {
            *r = r.clone() + coef.clone() * val.clone();
        }
    }
    Ok(DenseTensor::from_data(result, shape.to_vec()))
}

/// Partial trace over matched bond index pairs.
///
/// Each pair `(a, b)` traces over two bond indices by summing over
/// their shared diagonal. Paired bonds must have the same dimension.
/// The output tensor retains only the non-paired bonds in their original order.
///
/// # TCI-spec
///
/// Corresponds to `tci::trace` overload (2) (out-of-place).
///
/// # Errors
///
/// Returns an error if:
/// - A bond index is out of range
/// - A bond index appears in more than one pair
/// - Paired bonds have different dimensions
/// - `a == b` in any pair
///
/// # Examples
///
/// ```
/// use arnet_linalg::trace;
/// use arnet_tensor::DenseTensor;
///
/// // Matrix trace: tr([[1,2],[3,4]]) = 1 + 4 = 5
/// let mat = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// let result = trace(&mat, &[(0, 1)]).unwrap();
/// assert_eq!(result.shape(), &[1]);
/// assert_eq!(result.get(&[0]), 5.0);
/// ```
pub fn trace<T: Scalar>(
    tensor: &DenseTensor<T>,
    pairs: &[(usize, usize)],
) -> Result<DenseTensor<T>, String> {
    let rank = tensor.rank();
    let shape = tensor.shape();

    // Empty pairs: return a clone
    if pairs.is_empty() {
        return Ok(tensor.clone());
    }

    // Validate pairs
    let mut used = vec![false; rank];
    let mut trace_dims = Vec::with_capacity(pairs.len());
    for &(a, b) in pairs {
        if a >= rank || b >= rank {
            return Err(format!(
                "Bond index out of range: ({a}, {b}) for rank {rank}"
            ));
        }
        if a == b {
            return Err(format!("Self-pair not allowed: ({a}, {b})"));
        }
        if used[a] || used[b] {
            return Err(format!(
                "Bond index used in multiple pairs: ({a}, {b})"
            ));
        }
        if shape[a] != shape[b] {
            return Err(format!(
                "Dimension mismatch for pair ({a}, {b}): {} vs {}",
                shape[a], shape[b]
            ));
        }
        used[a] = true;
        used[b] = true;
        trace_dims.push(shape[a]);
    }

    // Free indices: those not in any pair, in original order
    let free_indices: Vec<usize> = (0..rank).filter(|i| !used[*i]).collect();
    let output_shape: Vec<usize> = if free_indices.is_empty() {
        vec![1] // Scalar result
    } else {
        free_indices.iter().map(|&i| shape[i]).collect()
    };
    let output_len: usize = output_shape.iter().product();

    // Precompute input strides (row-major)
    let input_strides = compute_strides(shape);

    // Precompute output strides for coordinate iteration
    let out_iter_shape: Vec<usize> = free_indices.iter().map(|&i| shape[i]).collect();

    // Total number of trace elements per output element
    let trace_total: usize = trace_dims.iter().product();

    let mut result = vec![T::zero(); output_len];

    // Iterate over each output element
    let mut out_coords = vec![0usize; free_indices.len()];
    let mut trace_coords = vec![0usize; pairs.len()];
    for (out_idx, res) in result.iter_mut().enumerate() {
        // Decode output flat index to coordinates
        if !free_indices.is_empty() {
            decode_coords(out_idx, &out_iter_shape, &mut out_coords);
        }

        let mut sum = T::zero();

        // Iterate over cartesian product of trace dimensions
        for trace_idx in 0..trace_total {
            // Decode trace flat index to coordinates
            decode_coords(trace_idx, &trace_dims, &mut trace_coords);

            // Build full input coordinate as flat index
            let mut flat = 0usize;
            for (i, &fi) in free_indices.iter().enumerate() {
                flat += out_coords[i] * input_strides[fi];
            }
            for (p, &(a, b)) in pairs.iter().enumerate() {
                let t = trace_coords[p];
                flat += t * input_strides[a] + t * input_strides[b];
            }

            sum = sum + tensor.data()[flat];
        }

        *res = sum;
    }

    Ok(DenseTensor::from_data(result, output_shape))
}

/// Compute row-major strides from shape.
pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Decode a flat index into coordinates given shape (row-major).
pub(crate) fn decode_coords(mut flat: usize, shape: &[usize], coords: &mut [usize]) {
    for i in (0..shape.len()).rev() {
        coords[i] = flat % shape[i];
        flat /= shape[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Scale tests ---

    #[test]
    fn test_scale_f64() {
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let scaled = scale(&tensor, 2.5);
        assert_eq!(scaled.get(&[0, 0]), 2.5);
        assert_eq!(scaled.get(&[0, 1]), 5.0);
        assert_eq!(scaled.get(&[1, 0]), 7.5);
        assert_eq!(scaled.get(&[1, 1]), 10.0);
        // Original unchanged
        assert_eq!(tensor.get(&[0, 0]), 1.0);
    }

    #[test]
    fn test_scale_complex() {
        use num_complex::Complex;
        let tensor = DenseTensor::from_data(
            vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
            vec![2],
        );
        let scaled = scale(&tensor, Complex::new(2.0, 3.0));
        // (1+0i)*(2+3i) = 2+3i
        assert_eq!(scaled.get(&[0]), Complex::new(2.0, 3.0));
        // (0+1i)*(2+3i) = -3+2i
        assert_eq!(scaled.get(&[1]), Complex::new(-3.0, 2.0));
    }

    // --- Norm tests ---

    #[test]
    fn test_norm_f64() {
        let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
        let n = norm(&tensor);
        assert!((n - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_complex() {
        use num_complex::Complex;
        // |3+4i| = 5, so norm of single element [3+4i] = 5
        let tensor = DenseTensor::from_data(vec![Complex::new(3.0, 4.0)], vec![1]);
        let n: f64 = norm(&tensor);
        assert!((n - 5.0).abs() < 1e-10);
    }

    // --- Normalize tests ---

    #[test]
    fn test_normalize_f64() {
        let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
        let (normalized, n) = normalize(&tensor);
        assert!((n - 2.0).abs() < 1e-10);
        assert!((norm(&normalized) - 1.0).abs() < 1e-10);
        // Original unchanged
        assert_eq!(tensor.get(&[0, 0]), 1.0);
    }

    #[test]
    #[should_panic(expected = "Cannot normalize zero tensor")]
    fn test_normalize_zero_panics() {
        let tensor = DenseTensor::<f64>::zeros(vec![2, 2]);
        let _ = normalize(&tensor);
    }

    // --- Linear combine tests ---

    #[test]
    fn test_linear_combine_basic() {
        let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
        let b = DenseTensor::<f64>::constant(vec![2, 2], 2.0);
        let result = linear_combine(&[&a, &b], &[3.0, 4.0]).unwrap();
        // 3*1 + 4*2 = 11
        assert_eq!(result.get(&[0, 0]), 11.0);
    }

    #[test]
    fn test_linear_combine_shape_mismatch() {
        let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
        let b = DenseTensor::<f64>::constant(vec![3, 3], 2.0);
        assert!(linear_combine(&[&a, &b], &[1.0, 1.0]).is_err());
    }

    #[test]
    fn test_linear_combine_empty() {
        let result = linear_combine::<f64>(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_combine_length_mismatch() {
        let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
        assert!(linear_combine(&[&a], &[1.0, 2.0]).is_err());
    }

    // --- Trace tests ---

    #[test]
    fn test_trace_matrix() {
        // tr([[1,2],[3,4]]) = 1 + 4 = 5
        let mat = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = trace(&mat, &[(0, 1)]).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.get(&[0]), 5.0);
    }

    #[test]
    fn test_trace_3x3_identity() {
        // tr(I_3) = 3
        let mut data = vec![0.0; 9];
        data[0] = 1.0;
        data[4] = 1.0;
        data[8] = 1.0;
        let mat = DenseTensor::<f64>::from_data(data, vec![3, 3]);
        let result = trace(&mat, &[(0, 1)]).unwrap();
        assert_eq!(result.get(&[0]), 3.0);
    }

    #[test]
    fn test_trace_partial_rank3() {
        // A[i,j,k] shape [2,3,3], trace over (1,2) → B[i] shape [2]
        // B[i] = Σ_j A[i,j,j]
        let mut data = vec![0.0; 18]; // 2*3*3
        // A[0,0,0]=1, A[0,1,1]=2, A[0,2,2]=3 → B[0] = 6
        data[0] = 1.0; // [0,0,0]
        data[4] = 2.0; // [0,1,1]
        data[8] = 3.0; // [0,2,2]
        // A[1,0,0]=4, A[1,1,1]=5, A[1,2,2]=6 → B[1] = 15
        data[9] = 4.0;  // [1,0,0]
        data[13] = 5.0; // [1,1,1]
        data[17] = 6.0; // [1,2,2]
        let tensor = DenseTensor::from_data(data, vec![2, 3, 3]);
        let result = trace(&tensor, &[(1, 2)]).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get(&[0]), 6.0);
        assert_eq!(result.get(&[1]), 15.0);
    }

    #[test]
    fn test_trace_tci_example() {
        // TCI spec example: shape {3, 4, 2, 4, 2}, pairs {{1,3}, {2,4}} → shape {3}
        // result[i0] = Σ_{t1,t2} A[i0, t1, t2, t1, t2]
        let shape = vec![3, 4, 2, 4, 2];
        let total: usize = shape.iter().product();
        let mut data = vec![0.0f64; total];

        // Set specific elements to verify correctness
        // A[0, 0, 0, 0, 0] = 1.0
        // A[0, 1, 1, 1, 1] = 2.0
        // result[0] should be 3.0
        let strides = vec![4 * 2 * 4 * 2, 2 * 4 * 2, 4 * 2, 2, 1]; // [64, 16, 8, 2, 1]
        data[0 * strides[0] + 0 * strides[1] + 0 * strides[2] + 0 * strides[3] + 0 * strides[4]] =
            1.0;
        data[0 * strides[0] + 1 * strides[1] + 1 * strides[2] + 1 * strides[3] + 1 * strides[4]] =
            2.0;
        // A[1, 2, 0, 2, 0] = 5.0
        data[1 * strides[0] + 2 * strides[1] + 0 * strides[2] + 2 * strides[3] + 0 * strides[4]] =
            5.0;

        let tensor = DenseTensor::from_data(data, shape);
        let result = trace(&tensor, &[(1, 3), (2, 4)]).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.get(&[0]), 3.0);
        assert_eq!(result.get(&[1]), 5.0);
        assert_eq!(result.get(&[2]), 0.0);
    }

    #[test]
    fn test_trace_full_contraction() {
        // All bonds paired → scalar
        // A[i,j,i,j] shape [2,3,2,3], pairs [(0,2),(1,3)]
        // result = Σ_{i,j} A[i,j,i,j]
        let shape = vec![2, 3, 2, 3];
        let total: usize = shape.iter().product();
        let mut data = vec![0.0f64; total];
        let strides = vec![18, 6, 3, 1]; // [3*2*3, 2*3, 3, 1]
        // A[0,0,0,0]=1, A[1,1,1,1]=2, A[0,2,0,2]=3
        data[0] = 1.0;
        data[1 * strides[0] + 1 * strides[1] + 1 * strides[2] + 1 * strides[3]] = 2.0;
        data[0 * strides[0] + 2 * strides[1] + 0 * strides[2] + 2 * strides[3]] = 3.0;
        let tensor = DenseTensor::from_data(data, shape);
        let result = trace(&tensor, &[(0, 2), (1, 3)]).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.get(&[0]), 6.0);
    }

    #[test]
    fn test_trace_empty_pairs() {
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = trace(&tensor, &[]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), tensor.data());
    }

    #[test]
    fn test_trace_dimension_mismatch() {
        let tensor = DenseTensor::<f64>::from_data(vec![0.0; 6], vec![2, 3]);
        assert!(trace(&tensor, &[(0, 1)]).is_err());
    }

    #[test]
    fn test_trace_index_out_of_range() {
        let tensor = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
        assert!(trace(&tensor, &[(0, 5)]).is_err());
    }

    #[test]
    fn test_trace_self_pair() {
        let tensor = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
        assert!(trace(&tensor, &[(1, 1)]).is_err());
    }

    #[test]
    fn test_trace_duplicate_index() {
        let tensor = DenseTensor::<f64>::from_data(vec![0.0; 8], vec![2, 2, 2]);
        assert!(trace(&tensor, &[(0, 1), (1, 2)]).is_err());
    }
}
