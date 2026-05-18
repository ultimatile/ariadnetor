//! Per-block layout primitives shared across block-sparse operations.
//!
//! Stride computation, identity-permutation order repack, and axis
//! permutation on a flat block buffer. These are reused by
//! `block_sparse_permute`, `block_sparse_contract`, and
//! `block_sparse_decomp`, and are kept in a single small module so the
//! per-op-family files stay under the 600-line per-file budget.

use arnet_core::backend::MemoryOrder;

/// Compute strides for the given memory order.
pub(crate) fn compute_strides(shape: &[usize], order: MemoryOrder) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    match order {
        MemoryOrder::RowMajor => {
            for i in (0..rank.saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        MemoryOrder::ColumnMajor => {
            for i in 1..rank {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
        }
    }
    strides
}

/// Transpose block data in the given memory order layout.
///
/// Convention: `perm[new_axis] = old_axis`.
pub(crate) fn transpose_block_data<T: Copy>(
    data: &[T],
    shape: &[usize],
    perm: &[usize],
    order: MemoryOrder,
) -> Vec<T> {
    let rank = shape.len();
    if rank <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let total = data.len();
    let old_strides = compute_strides(shape, order);
    let perm_strides: Vec<usize> = perm.iter().map(|&p| old_strides[p]).collect();

    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let new_strides = compute_strides(&new_shape, order);

    // Axis iteration order for flat→multi-index decomposition:
    // RowMajor strides are descending → process 0..rank
    // ColumnMajor strides are ascending → process (0..rank).rev()
    let decomp_order: Vec<usize> = match order {
        MemoryOrder::RowMajor => (0..rank).collect(),
        MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
    };

    let mut result = vec![data[0]; total];
    let mut idx = vec![0usize; rank];

    for (flat, out) in result.iter_mut().enumerate() {
        let mut rem = flat;
        for &ax in &decomp_order {
            idx[ax] = rem / new_strides[ax];
            rem %= new_strides[ax];
        }
        let old_flat: usize = idx
            .iter()
            .zip(perm_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();
        *out = data[old_flat];
    }

    result
}
