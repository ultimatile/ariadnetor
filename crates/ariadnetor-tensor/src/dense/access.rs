//! Data access and element-level operations for Dense.

use std::sync::Arc;

use super::Dense;

/// Compute row-major flat index from multi-dimensional indices.
fn rm_flat_index(indices: &[usize], shape: &[usize]) -> usize {
    debug_assert_eq!(indices.len(), shape.len());
    let mut idx = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        debug_assert!(
            indices[i] < shape[i],
            "index {} out of bounds for axis {} with size {}",
            indices[i],
            i,
            shape[i]
        );
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    idx
}

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

    /// Get element at multi-dimensional indices using row-major indexing.
    ///
    /// This is a convenience accessor for tests and simple use cases.
    /// The flat index is computed assuming the data is in row-major order.
    /// For backend-aware indexing, use `Tensor::get()` instead.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn get(&self, indices: &[usize]) -> T {
        let idx = rm_flat_index(indices, self.shape());
        self.data[idx].clone()
    }

    /// Set element at multi-dimensional indices using row-major indexing.
    ///
    /// This is a convenience accessor for tests and simple use cases.
    /// See [`get`](Dense::get) for the indexing convention.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, indices: &[usize], value: T) {
        let idx = rm_flat_index(indices, self.shape());
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
