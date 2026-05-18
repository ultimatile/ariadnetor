//! Memory-order reordering for `DenseTensorData<T>`.

use crate::DenseTensorData;
use arnet_core::backend::MemoryOrder;
use std::borrow::Cow;

/// Reorder a dense tensor's flat data between memory layouts.
///
/// If `from == to`, returns a clone (Arc-shared, zero-copy). Otherwise
/// produces a new tensor whose `order()` matches the requested `to`.
pub fn reorder<T: Clone>(
    tensor: &DenseTensorData<T>,
    from: MemoryOrder,
    to: MemoryOrder,
) -> DenseTensorData<T> {
    if from == to {
        return tensor.clone();
    }
    let shape = tensor.shape();
    let rank = shape.len();
    let total: usize = shape.iter().product();
    if total == 0 {
        return DenseTensorData::from_raw_parts(Vec::new(), shape.to_vec(), to);
    }
    let raw = tensor.data();
    let mut new_data = Vec::with_capacity(total);
    let mut coords = vec![0usize; rank];

    // Target order determines iteration direction
    let axis_order: Vec<usize> = match to {
        MemoryOrder::RowMajor => (0..rank).collect(),
        MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
    };

    for _ in 0..total {
        // Compute source flat index in `from` order
        let src_idx = flat_index(&coords, shape, from);
        new_data.push(raw[src_idx].clone());

        // Advance coords in `to` order
        for &d in axis_order.iter().rev() {
            coords[d] += 1;
            if coords[d] < shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    DenseTensorData::from_raw_parts(new_data, shape.to_vec(), to)
}

/// Normalize a tensor's memory order to `target`, returning a borrow when
/// the tensor is already in the target order.
///
/// Use at the entry of any operation that requires a specific input
/// layout (typically backend kernels expecting `backend.preferred_order()`).
/// The returned `Cow` is `Borrowed` when no conversion is needed and
/// `Owned` when a `reorder` was performed.
pub fn normalize_to<T: Clone>(
    tensor: &DenseTensorData<T>,
    target: MemoryOrder,
) -> Cow<'_, DenseTensorData<T>> {
    if tensor.order() == target {
        Cow::Borrowed(tensor)
    } else {
        Cow::Owned(reorder(tensor, tensor.order(), target))
    }
}

/// Compute flat index for given coordinates in the specified memory order.
pub fn flat_index(coords: &[usize], shape: &[usize], order: MemoryOrder) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    match order {
        MemoryOrder::RowMajor => {
            for i in (0..shape.len()).rev() {
                idx += coords[i] * stride;
                stride *= shape[i];
            }
        }
        MemoryOrder::ColumnMajor => {
            for i in 0..shape.len() {
                idx += coords[i] * stride;
                stride *= shape[i];
            }
        }
    }
    idx
}
