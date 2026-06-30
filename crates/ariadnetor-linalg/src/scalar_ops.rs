use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{ComputeBackend, MemoryOrder};
use ariadnetor_tensor::DenseTensorData;
use std::ops::Mul;

use crate::error::LinalgError;
use ariadnetor_tensor::{flat_index, normalize_to_data};

/// Validate trace pair indices and return the per-axis "is traced" mask.
///
/// Checks the layout-agnostic conditions shared by every partial trace —
/// indices in range, no self-pair, no index reused across pairs — and reports
/// which axes are traced. Dimension- or sector-level checks (dense
/// `shape[a] == shape[b]`, the block-sparse identical-block / opposite-direction
/// rules) are layout-specific and stay in the respective kernels. Shared
/// between the dense and block-sparse traces, mirroring how [`crate::perm`]'s
/// `validate_perm` serves both permutation kernels.
pub(crate) fn validate_trace_pairs(
    pairs: &[(usize, usize)],
    rank: usize,
) -> Result<Vec<bool>, LinalgError> {
    let mut used = vec![false; rank];
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
        used[a] = true;
        used[b] = true;
    }
    Ok(used)
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
/// Internal kernel for the partial trace operating on the joined
/// [`DenseTensorData<T>`] form. The public entry point is
/// [`crate::trace_with_backend`].
pub(crate) fn trace_dense<T: Scalar>(
    tensor: &DenseTensorData<T>,
    pairs: &[(usize, usize)],
) -> Result<DenseTensorData<T>, LinalgError> {
    let rank = tensor.rank();
    let shape = tensor.shape();

    // Empty pairs: return a clone
    if pairs.is_empty() {
        return Ok(tensor.clone());
    }

    // Normalize input to RowMajor for the direct-indexing implementation below.
    let rm_tensor = normalize_to_data(tensor, MemoryOrder::RowMajor);
    let tensor: &DenseTensorData<T> = &rm_tensor;

    // Shared range / self / reuse checks; the dense dimension-match check
    // stays here.
    let used = validate_trace_pairs(pairs, rank)?;
    let mut trace_dims = Vec::with_capacity(pairs.len());
    for &(a, b) in pairs {
        if shape[a] != shape[b] {
            return Err(LinalgError::InvalidArgument(format!(
                "Dimension mismatch for pair ({a}, {b}): {} vs {}",
                shape[a], shape[b]
            )));
        }
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

    Ok(DenseTensorData::from_raw_parts(
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
/// - **Matrix -> Vector**: If input has shape `[n, n]`, returns a vector of length `n`
///   containing the diagonal elements. Data is assumed to be in RowMajor layout.
/// - **Vector -> Matrix**: If input has shape `[n]`, returns an `n x n` matrix with the
///   input elements on the diagonal and zeros elsewhere (RowMajor layout).
///
/// # Errors
///
/// Returns an error if the input is a non-square matrix (rank 2 with mismatched dimensions)
/// or has rank > 2.
///
/// Internal kernel for the diagonal operation on the joined
/// [`DenseTensorData<T>`] form. The public entry point is
/// [`crate::diag_with_backend`].
pub(crate) fn diag_dense<T: Scalar>(
    tensor: &DenseTensorData<T>,
) -> Result<DenseTensorData<T>, LinalgError> {
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
            Ok(DenseTensorData::from_raw_parts(
                data,
                vec![n, n],
                tensor.order(),
            ))
        }
        2 => {
            // Matrix -> diagonal vector: normalize to RowMajor for direct indexing.
            let (m, n) = (shape[0], shape[1]);
            if m != n {
                return Err(LinalgError::InvalidArgument(format!(
                    "diag requires a square matrix, got {m}x{n}"
                )));
            }
            let rm = normalize_to_data(tensor, MemoryOrder::RowMajor);
            let raw = rm.data();
            let coords_rm = MemoryOrder::RowMajor;
            let data: Vec<T> = (0..n)
                .map(|i| raw[flat_index(&[i, i], shape, coords_rm)])
                .collect();
            // 1D output: layout is invariant; propagate the input's order.
            Ok(DenseTensorData::from_raw_parts(
                data,
                vec![n],
                tensor.order(),
            ))
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
/// Internal kernel for the diagonal-scale operation on the joined
/// [`DenseTensorData<T>`] form. The public entry point is
/// [`crate::diagonal_scale`].
pub(crate) fn diagonal_scale_dense<T, S>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    weights: &[S],
    axis: usize,
) -> Result<DenseTensorData<T>, LinalgError>
where
    T: Clone + Mul<S, Output = T> + 'static,
    S: Clone,
{
    diagonal_scale_inner(tensor, weights, axis, backend.preferred_order())
}

/// Inner implementation with explicit memory order (for internal use and testing).
fn diagonal_scale_inner<T, S>(
    tensor: &DenseTensorData<T>,
    weights: &[S],
    axis: usize,
    order: MemoryOrder,
) -> Result<DenseTensorData<T>, LinalgError>
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
        return Ok(DenseTensorData::from_raw_parts(
            Vec::new(),
            tensor.shape().to_vec(),
            order,
        ));
    }

    // The strip-length computation below assumes the input data is laid
    // out in `order`; normalize to that order if needed.
    let normalized = normalize_to_data(tensor, order);
    let tensor: &DenseTensorData<T> = &normalized;
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

    Ok(DenseTensorData::from_raw_parts(
        result,
        shape.to_vec(),
        order,
    ))
}

#[cfg(test)]
mod diagonal_scale_tests {
    use super::*;
    use ariadnetor_tensor::{MemoryOrder, reorder_data};

    /// RM/CM invariance: the same logical tensor, laid out in RM and CM,
    /// should produce logically identical results.
    #[test]
    fn rm_cm_invariance_axis0() {
        let rm_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cm_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let t_rm = DenseTensorData::from_raw_parts(rm_data, vec![2, 3], MemoryOrder::RowMajor);
        let t_cm = DenseTensorData::from_raw_parts(cm_data, vec![2, 3], MemoryOrder::ColumnMajor);
        let weights = [10.0, 20.0];

        let r_rm = diagonal_scale_inner(&t_rm, &weights, 0, MemoryOrder::RowMajor).unwrap();
        let r_cm = diagonal_scale_inner(&t_cm, &weights, 0, MemoryOrder::ColumnMajor).unwrap();

        let r_cm_as_rm = reorder_data(&r_cm, MemoryOrder::RowMajor);
        assert_eq!(r_rm.data(), r_cm_as_rm.data(), "axis0 RM/CM mismatch");
    }

    #[test]
    fn rm_cm_invariance_axis1() {
        let rm_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cm_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let t_rm = DenseTensorData::from_raw_parts(rm_data, vec![2, 3], MemoryOrder::RowMajor);
        let t_cm = DenseTensorData::from_raw_parts(cm_data, vec![2, 3], MemoryOrder::ColumnMajor);
        let weights = [10.0, 20.0, 30.0];

        let r_rm = diagonal_scale_inner(&t_rm, &weights, 1, MemoryOrder::RowMajor).unwrap();
        let r_cm = diagonal_scale_inner(&t_cm, &weights, 1, MemoryOrder::ColumnMajor).unwrap();

        let r_cm_as_rm = reorder_data(&r_cm, MemoryOrder::RowMajor);
        assert_eq!(r_rm.data(), r_cm_as_rm.data(), "axis1 RM/CM mismatch");
    }

    #[test]
    fn rm_cm_invariance_rank3() {
        let rm_data: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let t_rm = DenseTensorData::from_raw_parts(rm_data, vec![2, 2, 2], MemoryOrder::RowMajor);

        let cm_data = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
        let t_cm =
            DenseTensorData::from_raw_parts(cm_data, vec![2, 2, 2], MemoryOrder::ColumnMajor);

        let weights = [3.0, 7.0];

        let r_rm = diagonal_scale_inner(&t_rm, &weights, 1, MemoryOrder::RowMajor).unwrap();
        let r_cm = diagonal_scale_inner(&t_cm, &weights, 1, MemoryOrder::ColumnMajor).unwrap();

        let r_cm_as_rm = reorder_data(&r_cm, MemoryOrder::RowMajor);

        for (a, b) in r_rm.data().iter().zip(r_cm_as_rm.data()) {
            assert!(
                (a - b).abs() < 1e-10,
                "rank3 axis1 RM/CM mismatch: {a} vs {b}"
            );
        }
    }
}
