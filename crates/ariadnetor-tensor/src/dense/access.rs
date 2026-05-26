//! Coordinate-based element access for `DenseTensorData<T>`.
//!
//! `get` / `set` resolve flat indices through the layout's memory
//! order, so a `RowMajor`-tagged and a `ColumnMajor`-tagged tensor
//! holding the same logical matrix return the same value at the same
//! `[i, j, ...]`.

use crate::DenseTensorData;
use crate::reorder::flat_index;

impl<T> DenseTensorData<T>
where
    T: Clone,
{
    /// Get element at multi-dimensional indices.
    ///
    /// The flat index is computed using `self.order()`.
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
        let order = self.order();
        let idx = flat_index(indices, shape, order);
        self.storage().data()[idx].clone()
    }

    /// Set element at multi-dimensional indices (triggers CoW on the
    /// storage half if shared).
    ///
    /// The flat index is computed using `self.order()`.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, indices: &[usize], value: T) {
        let idx = {
            let shape = self.shape();
            assert_eq!(indices.len(), shape.len());
            for (axis, (&i, &dim)) in indices.iter().zip(shape).enumerate() {
                assert!(
                    i < dim,
                    "index {i} out of bounds for axis {axis} with size {dim}"
                );
            }
            flat_index(indices, shape, self.order())
        };
        self.storage_mut().data_mut()[idx] = value;
    }

    /// Mutable reference to the underlying contiguous data buffer
    /// (triggers CoW on the storage half if shared).
    pub fn data_mut(&mut self) -> &mut [T] {
        self.storage_mut().data_mut()
    }

    /// Iterate over the flat storage in flat (storage) order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.storage().data().iter()
    }

    /// Fill every element with a constant value (triggers CoW if
    /// shared). Forwards to the storage half.
    pub fn fill(&mut self, value: T) {
        self.storage_mut().fill(value);
    }

    /// Scale every element by a scalar factor in place (triggers CoW
    /// if shared).
    pub fn scale<S>(&mut self, factor: S)
    where
        T: std::ops::Mul<S, Output = T>,
        S: Clone,
    {
        self.storage_mut().scale(factor);
    }

    /// Apply a function to each element in place (triggers CoW if
    /// shared).
    pub fn map_mut<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        self.storage_mut().map_mut(f);
    }
}
