use arnet_core::scalar::Scalar;
use arnet_tensor::{DenseTensor, MemoryOrder};
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
    let data: Vec<T> = tensor.iter().map(|x| x.clone() * factor.clone()).collect();
    DenseTensor::from_data_with_order(data, tensor.shape().to_vec(), tensor.memory_order())
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
    let data: Vec<T> = tensor.iter().map(|&x| x.scale_real(inv_norm)).collect();
    (
        DenseTensor::from_data_with_order(data, tensor.shape().to_vec(), tensor.memory_order()),
        n,
    )
}

/// Linear combination of tensors: Σ coefs\[i\] * tensors\[i\].
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
pub fn linear_combine<T>(tensors: &[&DenseTensor<T>], coefs: &[T]) -> Result<DenseTensor<T>, String>
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
    let order = tensors[0].memory_order();
    for (tensor, coef) in tensors.iter().zip(coefs) {
        let t = tensor.to_contiguous(order);
        for (r, val) in result.iter_mut().zip(t.data()) {
            *r = r.clone() + coef.clone() * val.clone();
        }
    }
    Ok(DenseTensor::from_data_with_order(
        result,
        shape.to_vec(),
        order,
    ))
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
/// use arnet_tensor::{DenseTensor, MemoryOrder};
///
/// // Matrix trace: tr([[1,2],[3,4]]) = 1 + 4 = 5
/// let mat = DenseTensor::<f64>::from_data_with_order(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
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
            return Err(format!("Bond index used in multiple pairs: ({a}, {b})"));
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

    // Ensure row-major for direct indexing
    let rm = tensor.to_contiguous(MemoryOrder::RowMajor);

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
    let rm_data = rm.data();
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

            sum = sum + rm_data[flat];
        }

        *res = sum;
    }

    Ok(DenseTensor::from_data_with_order(
        result,
        output_shape,
        MemoryOrder::RowMajor,
    ))
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

/// Extract the diagonal of a square matrix, or construct a diagonal matrix from a vector.
///
/// - **Matrix → Vector**: If input has shape `[n, n]`, returns a vector of length `n`
///   containing the diagonal elements.
/// - **Vector → Matrix**: If input has shape `[n]`, returns an `n×n` matrix with the
///   input elements on the diagonal and zeros elsewhere.
///
/// # Errors
///
/// Returns an error if the input is a non-square matrix (rank 2 with mismatched dimensions)
/// or has rank > 2.
pub fn diag<T: Scalar>(tensor: &DenseTensor<T>) -> Result<DenseTensor<T>, String> {
    let shape = tensor.shape();
    match shape.len() {
        1 => {
            // Vector → diagonal matrix (1D tensors are unambiguous)
            let n = shape[0];
            let mut data = vec![T::zero(); n * n];
            for i in 0..n {
                data[i * n + i] = tensor.data()[i];
            }
            Ok(DenseTensor::from_data_with_order(
                data,
                vec![n, n],
                MemoryOrder::RowMajor,
            ))
        }
        2 => {
            // Matrix → diagonal vector: use get() for layout-agnostic access
            let (m, n) = (shape[0], shape[1]);
            if m != n {
                return Err(format!("diag requires a square matrix, got {m}×{n}"));
            }
            let data: Vec<T> = (0..n).map(|i| tensor.get(&[i, i])).collect();
            Ok(DenseTensor::from_data_with_order(
                data,
                vec![n],
                MemoryOrder::RowMajor,
            ))
        }
        r => Err(format!("diag requires rank 1 or 2, got rank {r}")),
    }
}

/// Scale each slice along `axis` by the corresponding weight.
///
/// Equivalent to multiplying by a diagonal matrix along the given axis:
/// `result[..., i, ...] = tensor[..., i, ...] * weights[i]` where `i`
/// is the index along `axis`.
///
/// # Errors
///
/// Returns an error if `axis` is out of range or `weights.len()` does not
/// match the dimension along `axis`.
///
/// # Examples
///
/// ```
/// use arnet_linalg::diagonal_scale;
/// use arnet_tensor::{DenseTensor, MemoryOrder};
///
/// // 2×3 matrix, scale columns by [1, 2, 3]
/// let m = DenseTensor::from_data_with_order(
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
///     vec![2, 3],
///     MemoryOrder::RowMajor,
/// );
/// let scaled = diagonal_scale(&m, &[1.0, 2.0, 3.0], 1).unwrap();
/// assert_eq!(scaled.get(&[0, 0]), 1.0);
/// assert_eq!(scaled.get(&[0, 1]), 4.0);
/// assert_eq!(scaled.get(&[1, 2]), 18.0);
/// ```
pub fn diagonal_scale<T, S>(
    tensor: &DenseTensor<T>,
    weights: &[S],
    axis: usize,
) -> Result<DenseTensor<T>, String>
where
    T: Clone + Mul<S, Output = T> + 'static,
    S: Clone,
{
    if axis >= tensor.rank() {
        return Err(format!(
            "axis {axis} out of range for rank {}",
            tensor.rank()
        ));
    }
    if weights.len() != tensor.shape()[axis] {
        return Err(format!(
            "weights length {} doesn't match axis {axis} dimension {}",
            weights.len(),
            tensor.shape()[axis]
        ));
    }
    Ok(tensor.map_with_index(|idx, val| val.clone() * weights[idx[axis]].clone()))
}
