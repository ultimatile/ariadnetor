//! Memory-order conversion for `DenseTensorData<T>`.
//!
//! Provides [`reorder_data`] for explicit conversion and
//! [`normalize_to_data`] for lazy `Cow`-borrowed normalization at the
//! entry of operations that require a specific input layout.

use crate::DenseTensorData;
use ariadnetor_core::backend::MemoryOrder;
use std::borrow::Cow;

/// Reorder a `DenseTensorData<T>` to the requested memory layout.
///
/// If `tensor.order() == to`, returns a clone (zero-copy via Arc).
/// Otherwise produces a new `DenseTensorData` whose layout `order()`
/// matches `to`. Primary callers are the linalg kernels at their
/// `&DenseTensorData<T>` entry points.
pub fn reorder_data<T: Clone>(tensor: &DenseTensorData<T>, to: MemoryOrder) -> DenseTensorData<T> {
    let from = tensor.order();
    let shape = tensor.shape();
    let rank = shape.len();
    let total = tensor.len();
    if from == to {
        return tensor.clone();
    }
    if total == 0 {
        return DenseTensorData::from_raw_parts(Vec::new(), shape.to_vec(), to);
    }
    let raw = tensor.storage().data();
    let mut new_data = Vec::with_capacity(total);
    let mut coords = vec![0usize; rank];

    let axis_order: Vec<usize> = match to {
        MemoryOrder::RowMajor => (0..rank).collect(),
        MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
    };

    for _ in 0..total {
        let src_idx = flat_index(&coords, shape, from);
        new_data.push(raw[src_idx].clone());
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

/// Normalize a `DenseTensorData<T>`'s memory order to `target`, with
/// `Cow::Borrowed` when the tensor is already in the target order.
///
/// Use at the entry of any operation that requires a specific input
/// layout (typically backend kernels expecting
/// `backend.preferred_order()`). The returned `Cow` is `Borrowed`
/// when no conversion is needed and `Owned` when a reorder was
/// performed.
pub fn normalize_to_data<T: Clone>(
    tensor: &DenseTensorData<T>,
    target: MemoryOrder,
) -> Cow<'_, DenseTensorData<T>> {
    if tensor.order() == target {
        Cow::Borrowed(tensor)
    } else {
        Cow::Owned(reorder_data(tensor, target))
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
