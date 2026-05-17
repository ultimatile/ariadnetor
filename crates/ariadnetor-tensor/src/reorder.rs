use crate::{Dense, DenseTensorData};
use arnet_core::backend::MemoryOrder;
use std::borrow::Cow;

/// Read-only abstraction over a dense flat buffer + its shape +
/// memory order, plus a constructor that returns the same flavor.
///
/// Implemented by both [`Dense<T>`] (legacy combined struct) and
/// [`DenseTensorData<T>`] (storage / layout split). Lets `reorder` and
/// `normalize_to` operate uniformly on either flavor during the
/// migration to `DenseTensorData<T>` (see issue #259) without forcing
/// every consumer to switch types in lockstep.
pub trait DenseView<T>: Sized {
    fn dense_data(&self) -> &[T];
    fn dense_shape(&self) -> &[usize];
    fn dense_order(&self) -> MemoryOrder;
    fn dense_len(&self) -> usize {
        self.dense_shape().iter().product()
    }
    /// Construct a new tensor of this flavor from flat data, shape,
    /// and the memory order the data is laid out in.
    fn dense_build(data: Vec<T>, shape: Vec<usize>, order: MemoryOrder) -> Self
    where
        T: Clone;
}

impl<T: Clone> DenseView<T> for Dense<T> {
    fn dense_data(&self) -> &[T] {
        self.data()
    }
    fn dense_shape(&self) -> &[usize] {
        self.shape()
    }
    fn dense_order(&self) -> MemoryOrder {
        self.order()
    }
    fn dense_build(data: Vec<T>, shape: Vec<usize>, order: MemoryOrder) -> Self
    where
        T: Clone,
    {
        Dense::new(data, shape, order)
    }
}

impl<T> DenseView<T> for DenseTensorData<T> {
    fn dense_data(&self) -> &[T] {
        self.data()
    }
    fn dense_shape(&self) -> &[usize] {
        self.shape()
    }
    fn dense_order(&self) -> MemoryOrder {
        self.order()
    }
    fn dense_build(data: Vec<T>, shape: Vec<usize>, order: MemoryOrder) -> Self
    where
        T: Clone,
    {
        DenseTensorData::from_raw_parts(data, shape, order)
    }
}

/// Reorder flat data between memory layouts.
///
/// If `from == to`, returns a clone (zero-copy via Arc). Otherwise
/// produces a new tensor whose `dense_order()` matches the requested
/// `to`.
pub fn reorder<T: Clone, D: DenseView<T> + Clone>(
    tensor: &D,
    from: MemoryOrder,
    to: MemoryOrder,
) -> D {
    if from == to {
        return tensor.clone();
    }
    let shape = tensor.dense_shape();
    let rank = shape.len();
    let total = tensor.dense_len();
    if total == 0 {
        return D::dense_build(Vec::new(), shape.to_vec(), to);
    }
    let raw = tensor.dense_data();
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

    D::dense_build(new_data, shape.to_vec(), to)
}

/// Normalize a tensor's memory order to `target`, returning a borrow when
/// the tensor is already in the target order.
///
/// Use at the entry of any operation that requires a specific input
/// layout (typically backend kernels expecting `backend.preferred_order()`).
/// The returned `Cow` is `Borrowed` when no conversion is needed and
/// `Owned` when a `reorder` was performed.
pub fn normalize_to<T: Clone, D: DenseView<T> + Clone>(
    tensor: &D,
    target: MemoryOrder,
) -> Cow<'_, D> {
    if tensor.dense_order() == target {
        Cow::Borrowed(tensor)
    } else {
        Cow::Owned(reorder(tensor, tensor.dense_order(), target))
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
