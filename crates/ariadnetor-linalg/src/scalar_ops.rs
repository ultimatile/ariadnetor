use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_tensor::{Dense, DenseTensor};
use num_traits::{Float, One, Zero};
use std::ops::{Add, Mul};

use crate::error::LinalgError;
use crate::tensor_bridge::{wrap_dense, wrap_dense_as};
use arnet_tensor::{flat_index, normalize_to};

/// Scale tensor by a scalar factor (out-of-place).
///
/// Returns a new tensor with all elements multiplied by `factor`.
///
/// # Examples
///
/// ```rust,ignore
/// use arnet_linalg::scale;
/// use arnet_tensor::DenseTensor;
///
/// let tensor = DenseTensor::<f64>::ones(vec![2, 3]);
/// let scaled = scale(&tensor, 2.5);
/// ```
pub fn scale<T, B>(tensor: &DenseTensor<T, B>, factor: T) -> DenseTensor<T, B>
where
    T: Clone + Mul<Output = T>,
    B: ComputeBackend,
{
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let result = scale_dense(&dense, factor);
    wrap_dense(result, backend_arc)
}

/// Internal kernel for [`scale`] operating on the legacy `Dense<T>` form.
pub(crate) fn scale_dense<T>(tensor: &Dense<T>, factor: T) -> Dense<T>
where
    T: Clone + Mul<Output = T>,
{
    let data: Vec<T> = tensor
        .data()
        .iter()
        .map(|x| x.clone() * factor.clone())
        .collect();
    Dense::new(data, tensor.shape().to_vec(), tensor.order())
}

/// Compute the Frobenius norm of a tensor.
///
/// Returns sqrt(sum |element|^2) as a real value.
///
/// # Examples
///
/// ```rust,ignore
/// use arnet_linalg::norm;
/// use arnet_tensor::DenseTensor;
///
/// let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
/// let n = norm(&tensor);
/// assert!((n - 2.0).abs() < 1e-10);
/// ```
pub fn norm<T: Scalar, B: ComputeBackend>(tensor: &DenseTensor<T, B>) -> T::Real {
    let dense = tensor.data().as_dense();
    norm_dense(&dense)
}

/// Internal kernel for [`norm`] operating on the legacy `Dense<T>` form.
pub(crate) fn norm_dense<T: Scalar>(tensor: &Dense<T>) -> T::Real {
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
/// ```rust,ignore
/// use arnet_linalg::normalize;
/// use arnet_tensor::DenseTensor;
///
/// let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
/// let (normalized, n) = normalize(&tensor);
/// assert!((n - 2.0).abs() < 1e-10);
/// ```
pub fn normalize<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
) -> (DenseTensor<T, B>, T::Real) {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let (result, n) = normalize_dense(&dense);
    (wrap_dense(result, backend_arc), n)
}

/// Internal kernel for [`normalize`] operating on the legacy `Dense<T>` form.
pub(crate) fn normalize_dense<T: Scalar>(tensor: &Dense<T>) -> (Dense<T>, T::Real) {
    let n = norm_dense(tensor);
    assert!(n != T::Real::zero(), "Cannot normalize zero tensor");
    let inv_norm = T::Real::one() / n;
    let data: Vec<T> = tensor
        .data()
        .iter()
        .map(|&x| x.scale_real(inv_norm))
        .collect();
    (Dense::new(data, tensor.shape().to_vec(), tensor.order()), n)
}

/// Linear combination of tensors: sum coefs[i] * tensors[i].
///
/// All tensors must have the same shape. The result is wrapped against
/// `tensors[0]`'s backend Arc; callers must ensure every input shares the
/// same backend Arc (a mismatch silently labels the output with the first
/// tensor's backend, which is wrong for backends carrying state).
///
/// # Errors
///
/// Returns an error if tensors have different shapes, the list is empty,
/// or tensors and coefficients have different lengths.
///
/// # Examples
///
/// ```rust,ignore
/// use arnet_linalg::linear_combine;
/// use arnet_tensor::DenseTensor;
///
/// let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
/// let b = DenseTensor::<f64>::constant(vec![2, 2], 2.0);
///
/// // 2*a + 3*b = 2*1 + 3*2 = 8
/// let result = linear_combine(&[&a, &b], &[2.0, 3.0]).unwrap();
/// ```
pub fn linear_combine<T, B>(
    tensors: &[&DenseTensor<T, B>],
    coefs: &[T],
) -> Result<DenseTensor<T, B>, LinalgError>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
    B: ComputeBackend,
{
    if tensors.is_empty() {
        return Err(LinalgError::InvalidArgument(
            "Cannot combine empty tensor list".to_string(),
        ));
    }
    let backend_arc = tensors[0].backend_arc().clone();
    // Materialize Dense views; they hold Arcs into the same buffers so
    // this is O(N) Arc-clones rather than O(N) data copies.
    let dense_views: Vec<Dense<T>> = tensors.iter().map(|t| t.data().as_dense()).collect();
    let dense_refs: Vec<&Dense<T>> = dense_views.iter().collect();
    let result = linear_combine_dense(&dense_refs, coefs)?;
    Ok(wrap_dense(result, backend_arc))
}

/// Internal kernel for [`linear_combine`] operating on legacy `Dense<T>` slices.
pub(crate) fn linear_combine_dense<T>(
    tensors: &[&Dense<T>],
    coefs: &[T],
) -> Result<Dense<T>, LinalgError>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
{
    if tensors.is_empty() {
        return Err(LinalgError::InvalidArgument(
            "Cannot combine empty tensor list".to_string(),
        ));
    }
    if tensors.len() != coefs.len() {
        return Err(LinalgError::InvalidArgument(format!(
            "Mismatched lengths: {} tensors vs {} coefficients",
            tensors.len(),
            coefs.len()
        )));
    }
    let shape = tensors[0].shape();
    let order = tensors[0].order();
    for t in &tensors[1..] {
        if t.shape() != shape {
            return Err(LinalgError::InvalidArgument(
                "All tensors must have the same shape".to_string(),
            ));
        }
        if t.order() != order {
            return Err(LinalgError::InvalidArgument(format!(
                "All tensors must have the same memory order; got {:?} and {:?}",
                order,
                t.order()
            )));
        }
    }
    let len = tensors[0].len();
    let mut result = vec![T::zero(); len];
    // All tensors share order; iterate element-wise in storage order.
    for (tensor, coef) in tensors.iter().zip(coefs) {
        for (r, val) in result.iter_mut().zip(tensor.data()) {
            *r = r.clone() + coef.clone() * val.clone();
        }
    }
    Ok(Dense::new(result, shape.to_vec(), order))
}

/// Partial trace over matched bond index pairs.
///
/// Each pair `(a, b)` traces over two bond indices by summing over
/// their shared diagonal. Paired bonds must have the same dimension.
/// The output tensor retains only the non-paired bonds in their original order.
///
/// The implementation requires RowMajor-laid input for direct coordinate
/// decomposition. The function normalizes the input to RowMajor at entry
/// so callers may pass tensors in any order.
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
/// ```rust,ignore
/// use arnet_linalg::trace;
/// use arnet_tensor::DenseTensor;
///
/// // Matrix trace: tr(I_2) = 2 (eye preserves the active backend's preferred order)
/// let mat = DenseTensor::<f64>::eye(2);
/// let result = trace(&mat, &[(0, 1)]).unwrap();
/// assert_eq!(result.shape(), &[1]);
/// ```
pub fn trace<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    pairs: &[(usize, usize)],
) -> Result<DenseTensor<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let result = trace_dense(&dense, pairs)?;
    Ok(wrap_dense(result, backend_arc))
}

/// Internal kernel for [`trace`] operating on the legacy `Dense<T>` form.
pub(crate) fn trace_dense<T: Scalar>(
    tensor: &Dense<T>,
    pairs: &[(usize, usize)],
) -> Result<Dense<T>, LinalgError> {
    let rank = tensor.rank();
    let shape = tensor.shape();

    // Empty pairs: return a clone
    if pairs.is_empty() {
        return Ok(tensor.clone());
    }

    // Normalize input to RowMajor for the direct-indexing implementation below.
    let rm_tensor = normalize_to(tensor, MemoryOrder::RowMajor);
    let tensor: &Dense<T> = &rm_tensor;

    // Validate pairs
    let mut used = vec![false; rank];
    let mut trace_dims = Vec::with_capacity(pairs.len());
    for &(a, b) in pairs {
        if a >= rank || b >= rank {
            return Err(LinalgError::InvalidArgument(format!(
                "Bond index out of range: ({a}, {b}) for rank {rank}"
            )));
        }
        if a == b {
            return Err(LinalgError::InvalidArgument(format!(
                "Self-pair not allowed: ({a}, {b})"
            )));
        }
        if used[a] || used[b] {
            return Err(LinalgError::InvalidArgument(format!(
                "Bond index used in multiple pairs: ({a}, {b})"
            )));
        }
        if shape[a] != shape[b] {
            return Err(LinalgError::InvalidArgument(format!(
                "Dimension mismatch for pair ({a}, {b}): {} vs {}",
                shape[a], shape[b]
            )));
        }
        used[a] = true;
        used[b] = true;
        trace_dims.push(shape[a]);
    }

    // Data is assumed to be in RowMajor layout for direct indexing
    let data = tensor.data();

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

            sum = sum + data[flat];
        }

        *res = sum;
    }

    Ok(Dense::new(result, output_shape, MemoryOrder::RowMajor))
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
/// - **Matrix -> Vector**: If input has shape `[n, n]`, returns a vector of length `n`
///   containing the diagonal elements. Data is assumed to be in RowMajor layout.
/// - **Vector -> Matrix**: If input has shape `[n]`, returns an `n x n` matrix with the
///   input elements on the diagonal and zeros elsewhere (RowMajor layout).
///
/// # Errors
///
/// Returns an error if the input is a non-square matrix (rank 2 with mismatched dimensions)
/// or has rank > 2.
pub fn diag<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
) -> Result<DenseTensor<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let result = diag_dense(&dense)?;
    Ok(wrap_dense(result, backend_arc))
}

/// Internal kernel for [`diag`] operating on the legacy `Dense<T>` form.
pub(crate) fn diag_dense<T: Scalar>(tensor: &Dense<T>) -> Result<Dense<T>, LinalgError> {
    let shape = tensor.shape();
    match shape.len() {
        1 => {
            // Vector -> diagonal matrix. Output in input's order; for a 1D
            // input, layout is invariant so we propagate input.order().
            let n = shape[0];
            let mut data = vec![T::zero(); n * n];
            for i in 0..n {
                data[i * n + i] = tensor.data()[i];
            }
            Ok(Dense::new(data, vec![n, n], tensor.order()))
        }
        2 => {
            // Matrix -> diagonal vector: normalize to RowMajor for direct indexing.
            let (m, n) = (shape[0], shape[1]);
            if m != n {
                return Err(LinalgError::InvalidArgument(format!(
                    "diag requires a square matrix, got {m}x{n}"
                )));
            }
            let rm = normalize_to(tensor, MemoryOrder::RowMajor);
            let raw = rm.data();
            let coords_rm = MemoryOrder::RowMajor;
            let data: Vec<T> = (0..n)
                .map(|i| raw[flat_index(&[i, i], shape, coords_rm)])
                .collect();
            // 1D output: layout is invariant; propagate the input's order.
            Ok(Dense::new(data, vec![n], tensor.order()))
        }
        r => Err(LinalgError::InvalidArgument(format!(
            "diag requires rank 1 or 2, got rank {r}"
        ))),
    }
}

/// Scale each slice along `axis` by the corresponding weight.
///
/// Equivalent to multiplying by a diagonal matrix along the given axis:
/// `result[..., i, ...] = tensor[..., i, ...] * weights[i]` where `i`
/// is the index along `axis`.
///
/// Memory layout is determined by the backend's `preferred_order()`.
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if `axis` is out of range or
/// `weights.len()` does not match the dimension along `axis`.
///
/// # Examples
///
/// ```rust,ignore
/// use arnet_linalg::diagonal_scale;
/// use arnet_tensor::DenseTensor;
///
/// let m = DenseTensor::<f64>::ones(vec![2, 3]);
/// let scaled = diagonal_scale(&m, &[1.0, 2.0, 3.0], 1).unwrap();
/// ```
pub fn diagonal_scale<T, S, B>(
    tensor: &DenseTensor<T, B>,
    weights: &[S],
    axis: usize,
) -> Result<DenseTensor<T, B>, LinalgError>
where
    T: Clone + Mul<S, Output = T> + 'static,
    S: Clone,
    B: ComputeBackend,
{
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let result = diagonal_scale_dense(tensor.backend(), &dense, weights, axis)?;
    Ok(wrap_dense_as(result, backend_arc))
}

/// Internal kernel for [`diagonal_scale`] operating on the legacy `Dense<T>` form.
pub(crate) fn diagonal_scale_dense<T, S>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    weights: &[S],
    axis: usize,
) -> Result<Dense<T>, LinalgError>
where
    T: Clone + Mul<S, Output = T> + 'static,
    S: Clone,
{
    diagonal_scale_inner(tensor, weights, axis, backend.preferred_order())
}

/// Inner implementation with explicit memory order (for internal use and testing).
fn diagonal_scale_inner<T, S>(
    tensor: &Dense<T>,
    weights: &[S],
    axis: usize,
    order: MemoryOrder,
) -> Result<Dense<T>, LinalgError>
where
    T: Clone + Mul<S, Output = T> + 'static,
    S: Clone,
{
    if axis >= tensor.rank() {
        return Err(LinalgError::InvalidArgument(format!(
            "axis {axis} out of range for rank {}",
            tensor.rank()
        )));
    }
    if weights.len() != tensor.shape()[axis] {
        return Err(LinalgError::InvalidArgument(format!(
            "weights length {} doesn't match axis {axis} dimension {}",
            weights.len(),
            tensor.shape()[axis]
        )));
    }

    let total = tensor.len();
    if total == 0 {
        return Ok(Dense::new(Vec::new(), tensor.shape().to_vec(), order));
    }

    // The strip-length computation below assumes the input data is laid
    // out in `order`; normalize to that order if needed.
    let normalized = normalize_to(tensor, order);
    let tensor: &Dense<T> = &normalized;
    let shape = tensor.shape();
    let data = tensor.data();

    let strip_len: usize = match order {
        MemoryOrder::ColumnMajor => shape[..axis].iter().product::<usize>().max(1),
        MemoryOrder::RowMajor => shape[axis + 1..].iter().product::<usize>().max(1),
    };
    let axis_dim = shape[axis];

    let result: Vec<T> = data
        .iter()
        .enumerate()
        .map(|(i, val)| {
            let w_idx = (i / strip_len) % axis_dim;
            val.clone() * weights[w_idx].clone()
        })
        .collect();

    Ok(Dense::new(result, shape.to_vec(), order))
}

#[cfg(test)]
mod diagonal_scale_tests {
    use super::*;
    use arnet_tensor::{MemoryOrder, reorder};

    /// RM/CM invariance: the same logical tensor, laid out in RM and CM,
    /// should produce logically identical results.
    #[test]
    fn rm_cm_invariance_axis0() {
        let rm_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cm_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let t_rm = Dense::new(rm_data, vec![2, 3], MemoryOrder::RowMajor);
        let t_cm = Dense::new(cm_data, vec![2, 3], MemoryOrder::ColumnMajor);
        let weights = [10.0, 20.0];

        let r_rm = diagonal_scale_inner(&t_rm, &weights, 0, MemoryOrder::RowMajor).unwrap();
        let r_cm = diagonal_scale_inner(&t_cm, &weights, 0, MemoryOrder::ColumnMajor).unwrap();

        let r_cm_as_rm = reorder(&r_cm, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);
        assert_eq!(r_rm.data(), r_cm_as_rm.data(), "axis0 RM/CM mismatch");
    }

    #[test]
    fn rm_cm_invariance_axis1() {
        let rm_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cm_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let t_rm = Dense::new(rm_data, vec![2, 3], MemoryOrder::RowMajor);
        let t_cm = Dense::new(cm_data, vec![2, 3], MemoryOrder::ColumnMajor);
        let weights = [10.0, 20.0, 30.0];

        let r_rm = diagonal_scale_inner(&t_rm, &weights, 1, MemoryOrder::RowMajor).unwrap();
        let r_cm = diagonal_scale_inner(&t_cm, &weights, 1, MemoryOrder::ColumnMajor).unwrap();

        let r_cm_as_rm = reorder(&r_cm, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);
        assert_eq!(r_rm.data(), r_cm_as_rm.data(), "axis1 RM/CM mismatch");
    }

    #[test]
    fn rm_cm_invariance_rank3() {
        let rm_data: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let t_rm = Dense::new(rm_data, vec![2, 2, 2], MemoryOrder::RowMajor);

        let cm_data = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
        let t_cm = Dense::new(cm_data, vec![2, 2, 2], MemoryOrder::ColumnMajor);

        let weights = [3.0, 7.0];

        let r_rm = diagonal_scale_inner(&t_rm, &weights, 1, MemoryOrder::RowMajor).unwrap();
        let r_cm = diagonal_scale_inner(&t_cm, &weights, 1, MemoryOrder::ColumnMajor).unwrap();

        let r_cm_as_rm = reorder(&r_cm, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);

        for (a, b) in r_rm.data().iter().zip(r_cm_as_rm.data()) {
            assert!(
                (a - b).abs() < 1e-10,
                "rank3 axis1 RM/CM mismatch: {a} vs {b}"
            );
        }
    }
}
