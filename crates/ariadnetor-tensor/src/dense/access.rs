//! Data access and element-level operations for Dense.

use std::sync::Arc;

use super::Dense;

impl<T> Dense<T>
where
    T: Clone,
{
    /// Get a reference to the underlying contiguous data.
    ///
    /// The caller must check [`memory_order()`](super::Dense::memory_order)
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
    pub fn iter(&self) -> DenseIter<'_, T> {
        if self.is_contiguous() {
            DenseIter::Contiguous(self.data[self.offset..self.offset + self.len()].iter())
        } else {
            DenseIter::Strided(StridedIter::new(self))
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

/// Iterator over `Dense` elements.
///
/// Dispatches between a direct slice iterator (contiguous case) and
/// a stride-walking iterator (non-contiguous case).
pub enum DenseIter<'a, T> {
    Contiguous(std::slice::Iter<'a, T>),
    Strided(StridedIter<'a, T>),
}

impl<'a, T: Clone> Iterator for DenseIter<'a, T> {
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

impl<T: Clone> ExactSizeIterator for DenseIter<'_, T> {}

/// Stride-walking iterator for non-contiguous tensors.
///
/// Walks all logical elements in storage order by incrementing a multi-index
/// counter along axes sorted by ascending absolute stride (smallest stride =
/// fastest-varying axis). This matches the true memory layout regardless of
/// the tensor's `MemoryOrder` label.
pub struct StridedIter<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
    strides: &'a [isize],
    offset: usize,
    indices: Vec<usize>,
    remaining: usize,
    /// Axes sorted by ascending |stride| — fastest-varying first.
    axis_order: Vec<usize>,
}

impl<'a, T> StridedIter<'a, T> {
    fn new(tensor: &'a Dense<T>) -> Self {
        let total = tensor.len();
        let mut axis_order: Vec<usize> = (0..tensor.rank()).collect();
        let strides = tensor.strides();
        axis_order.sort_by_key(|&ax| strides[ax].unsigned_abs());
        Self {
            data: &tensor.data,
            shape: &tensor.shape,
            strides: &tensor.strides,
            offset: tensor.offset,
            indices: vec![0; tensor.rank()],
            remaining: total,
            axis_order,
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

    /// Increment multi-index along axes from fastest to slowest stride.
    fn advance(&mut self) {
        for &ax in &self.axis_order {
            self.indices[ax] += 1;
            if self.indices[ax] < self.shape[ax] {
                return;
            }
            self.indices[ax] = 0;
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
