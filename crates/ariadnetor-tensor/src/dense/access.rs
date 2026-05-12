//! Data access and element-level operations for Dense.

use std::sync::Arc;

use crate::reorder::flat_index;

use super::Dense;

impl<T> Dense<T>
where
    T: Clone,
{
    /// Get a reference to the underlying contiguous data.
    ///
    /// The caller must know the memory order (from the compute backend)
    /// to interpret the layout correctly.
    pub fn data(&self) -> &[T] {
        &self.data[..]
    }

    /// Get a mutable reference to the underlying contiguous data
    /// (triggers CoW if shared).
    pub fn data_mut(&mut self) -> &mut [T] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// Fill tensor with a constant value (triggers CoW if shared).
    pub fn fill(&mut self, value: T) {
        Arc::make_mut(&mut self.data).as_mut_slice().fill(value);
    }

    /// Iterate over all elements in storage order.
    ///
    /// Element ordering depends on the memory layout (determined by
    /// the compute backend) and is not guaranteed to follow any
    /// particular index order. Use this for order-independent
    /// operations such as norm, scale, and element-wise maps.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data[..].iter()
    }

    /// Get element at multi-dimensional indices.
    ///
    /// The flat index is computed using `self.order()`, so a
    /// `RowMajor`-tagged and a `ColumnMajor`-tagged Dense holding
    /// the same logical matrix in their respective layouts return
    /// the same value at the same `[i, j, ...]`. This matches
    /// [`Tensor::get`](../../arnet/struct.Tensor.html#method.get)
    /// on the same storage; prefer `Tensor::get` only when
    /// backend-level concerns (Arc dereferencing, generic backend
    /// dispatch) are relevant — not for different indexing
    /// semantics.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn get(&self, indices: &[usize]) -> T {
        let shape = self.shape();
        assert_eq!(indices.len(), shape.len());
        for (axis, (&idx, &dim)) in indices.iter().zip(shape).enumerate() {
            assert!(
                idx < dim,
                "index {idx} out of bounds for axis {axis} with size {dim}"
            );
        }
        let idx = flat_index(indices, shape, self.order());
        self.data[idx].clone()
    }

    /// Set element at multi-dimensional indices.
    ///
    /// The flat index is computed using `self.order()`. See
    /// [`get`](Dense::get) for the indexing convention.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, indices: &[usize], value: T) {
        let shape = self.shape();
        assert_eq!(indices.len(), shape.len());
        for (axis, (&idx, &dim)) in indices.iter().zip(shape).enumerate() {
            assert!(
                idx < dim,
                "index {idx} out of bounds for axis {axis} with size {dim}"
            );
        }
        let order = self.order();
        let idx = flat_index(indices, shape, order);
        Arc::make_mut(&mut self.data)[idx] = value;
    }

    /// Get pointer to the underlying data for FFI.
    ///
    /// Returns a pointer to the start of the data buffer.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get mutable pointer to the underlying data for FFI (triggers CoW if shared).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        Arc::make_mut(&mut self.data).as_mut_ptr()
    }
}
