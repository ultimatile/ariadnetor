//! Data access and element-level operations for DenseTensor.

use std::sync::Arc;

use super::DenseTensor;

impl<T> DenseTensor<T>
where
    T: Clone,
{
    /// Get a reference to the underlying data as a row-major contiguous slice.
    ///
    /// Existing callers (linalg, transpose, scalar_ops) index this slice
    /// assuming row-major layout. Returning non-row-major data would silently
    /// produce wrong results in those paths.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not row-major contiguous.
    pub fn data(&self) -> &[T] {
        assert!(
            self.is_row_major(),
            "data() requires row-major contiguous tensor; \
             call to_contiguous(MemoryOrder::RowMajor) first"
        );
        &self.data[self.offset..self.offset + self.len()]
    }

    /// Get a mutable reference to the underlying data (triggers CoW if shared).
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not row-major contiguous.
    pub fn data_mut(&mut self) -> &mut [T] {
        assert!(
            self.is_row_major(),
            "data_mut() requires row-major contiguous tensor; \
             call to_contiguous(MemoryOrder::RowMajor) first"
        );
        let len = self.len();
        let offset = self.offset;
        &mut Arc::make_mut(&mut self.data).as_mut_slice()[offset..offset + len]
    }

    /// Get a reference to the underlying data for any contiguous layout.
    ///
    /// Unlike [`data()`](Self::data) which requires row-major, this accepts
    /// any contiguous tensor (row-major or column-major). The caller must
    /// know the tensor's layout to interpret the data correctly.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous.
    pub fn data_contiguous(&self) -> &[T] {
        assert!(
            self.is_contiguous(),
            "data_contiguous() requires contiguous tensor; \
             call to_contiguous() first"
        );
        &self.data[self.offset..self.offset + self.len()]
    }

    /// Get a mutable reference to the underlying data for any contiguous layout
    /// (triggers CoW if shared).
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous.
    pub fn data_contiguous_mut(&mut self) -> &mut [T] {
        assert!(
            self.is_contiguous(),
            "data_contiguous_mut() requires contiguous tensor; \
             call to_contiguous() first"
        );
        let len = self.len();
        let offset = self.offset;
        &mut Arc::make_mut(&mut self.data).as_mut_slice()[offset..offset + len]
    }

    /// Get element at given indices
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn get(&self, indices: &[usize]) -> T {
        let flat_index = self.flat_index(indices);
        self.data[flat_index].clone()
    }

    /// Set element at given indices (triggers CoW if shared).
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, indices: &[usize], value: T) {
        let flat_index = self.flat_index(indices);
        Arc::make_mut(&mut self.data)[flat_index] = value;
    }

    /// Fill tensor with a constant value (triggers CoW if shared).
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous.
    pub fn fill(&mut self, value: T) {
        assert!(
            self.is_contiguous(),
            "fill() requires contiguous tensor; \
             call to_contiguous() first"
        );
        let len = self.len();
        let offset = self.offset;
        Arc::make_mut(&mut self.data).as_mut_slice()[offset..offset + len].fill(value);
    }

    /// Get pointer to the underlying data for FFI.
    ///
    /// Returns a pointer to the first logical element (accounting for offset).
    /// Callers rebuild a `len()`-element slice from this pointer, so
    /// non-contiguous tensors would drop stride information and feed wrong
    /// values to kernels.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous.
    pub fn as_ptr(&self) -> *const T {
        assert!(self.is_contiguous(), "as_ptr() requires contiguous tensor");
        unsafe { self.data.as_ptr().add(self.offset) }
    }

    /// Get mutable pointer to the underlying data for FFI (triggers CoW if shared).
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        assert!(
            self.is_contiguous(),
            "as_mut_ptr() requires contiguous tensor"
        );
        let offset = self.offset;
        unsafe { Arc::make_mut(&mut self.data).as_mut_ptr().add(offset) }
    }
}
