//! Data access and element-level operations for DenseTensor.

use std::sync::Arc;

use super::{DenseTensor, MemoryOrder};

impl<T> DenseTensor<T>
where
    T: Clone,
{
    /// Get a reference to the underlying contiguous data.
    ///
    /// The caller must check [`memory_order()`](super::DenseTensor::memory_order)
    /// to interpret the layout correctly.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous.
    pub fn data(&self) -> &[T] {
        assert!(
            self.is_contiguous(),
            "data() requires contiguous tensor; \
             call to_contiguous() first"
        );
        &self.data[self.offset..self.offset + self.len()]
    }

    /// Get a mutable reference to the underlying contiguous data
    /// (triggers CoW if shared).
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous.
    pub fn data_mut(&mut self) -> &mut [T] {
        assert!(
            self.is_contiguous(),
            "data_mut() requires contiguous tensor; \
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

    /// Iterate over all elements in storage order.
    ///
    /// If the tensor is contiguous, this iterates the underlying data slice
    /// directly (zero-cost). Otherwise, it walks elements via stride arithmetic.
    ///
    /// Element ordering depends on the memory layout and is not guaranteed to
    /// follow any particular index order. Use this for order-independent
    /// operations such as norm, scale, and element-wise maps.
    pub fn iter(&self) -> DenseTensorIter<'_, T> {
        if self.is_contiguous() {
            DenseTensorIter::Contiguous(self.data[self.offset..self.offset + self.len()].iter())
        } else {
            DenseTensorIter::Strided(StridedIter::new(self))
        }
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

/// Iterator over `DenseTensor` elements.
///
/// Dispatches between a direct slice iterator (contiguous case) and
/// a stride-walking iterator (non-contiguous case).
pub enum DenseTensorIter<'a, T> {
    Contiguous(std::slice::Iter<'a, T>),
    Strided(StridedIter<'a, T>),
}

impl<'a, T: Clone> Iterator for DenseTensorIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(it) => it.next(),
            Self::Strided(it) => it.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Contiguous(it) => it.size_hint(),
            Self::Strided(it) => it.size_hint(),
        }
    }
}

impl<T: Clone> ExactSizeIterator for DenseTensorIter<'_, T> {}

/// Stride-walking iterator for non-contiguous tensors.
///
/// Walks all logical elements by incrementing a multi-index counter
/// in the tensor's memory order and computing the flat offset via strides.
/// RowMajor: last axis fastest. ColumnMajor: first axis fastest.
pub struct StridedIter<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
    strides: &'a [isize],
    offset: usize,
    indices: Vec<usize>,
    remaining: usize,
    order: MemoryOrder,
}

impl<'a, T> StridedIter<'a, T> {
    fn new(tensor: &'a DenseTensor<T>) -> Self {
        let total = tensor.len();
        Self {
            data: &tensor.data,
            shape: &tensor.shape,
            strides: &tensor.strides,
            offset: tensor.offset,
            indices: vec![0; tensor.rank()],
            remaining: total,
            order: tensor.memory_order(),
        }
    }

    fn flat_offset(&self) -> usize {
        let raw: isize = self
            .indices
            .iter()
            .zip(self.strides)
            .map(|(&idx, &stride)| idx as isize * stride)
            .sum();
        (self.offset as isize + raw) as usize
    }

    /// Increment multi-index in the tensor's memory order.
    fn advance(&mut self) {
        match self.order {
            MemoryOrder::RowMajor => {
                for i in (0..self.indices.len()).rev() {
                    self.indices[i] += 1;
                    if self.indices[i] < self.shape[i] {
                        return;
                    }
                    self.indices[i] = 0;
                }
            }
            MemoryOrder::ColumnMajor => {
                for i in 0..self.indices.len() {
                    self.indices[i] += 1;
                    if self.indices[i] < self.shape[i] {
                        return;
                    }
                    self.indices[i] = 0;
                }
            }
        }
    }
}

impl<'a, T: Clone> Iterator for StridedIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let idx = self.flat_offset();
        self.advance();
        self.remaining -= 1;
        Some(&self.data[idx])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Clone> ExactSizeIterator for StridedIter<'_, T> {}
