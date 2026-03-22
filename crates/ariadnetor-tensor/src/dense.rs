//! Dense tensor storage with strides-based memory layout
//!
//! Provides a dense tensor with explicit strides and Arc-based shared ownership.
//! The tensor is self-describing regarding its memory layout — code should not
//! assume row-major or column-major without checking.

use aligned_vec::{AVec, ConstAlign};
use num_traits::{Float, One, Zero};
use std::fmt;
use std::ops::{Add, Mul};
use std::sync::Arc;

/// 64-byte alignment for SIMD (AVX-512)
type Align64 = ConstAlign<64>;

pub use arnet_core::MemoryOrder;

/// Dense tensor with shared ownership (Arc + Copy-on-Write)
///
/// # Memory Layout
///
/// Each tensor carries explicit `strides` and `offset` describing how logical
/// indices map to positions in the underlying data buffer. Constructors default
/// to row-major (C-contiguous) layout, but backends may produce other layouts.
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64)
#[derive(Clone)]
pub struct DenseTensor<T = f64> {
    /// Shared data buffer (64-byte aligned)
    data: Arc<AVec<T, Align64>>,
    /// Tensor shape
    shape: Vec<usize>,
    /// Strides for each axis (element count, signed for future negative strides)
    strides: Vec<isize>,
    /// Offset into the data buffer (element index of the first logical element)
    offset: usize,
    /// The memory order this tensor was created with.
    /// Needed to disambiguate layouts where strides alone are ambiguous
    /// (e.g., 1D tensors, tensors with size-1 dimensions).
    order: MemoryOrder,
}

// ============================================================================
// Strides computation helpers
// ============================================================================

/// Compute row-major (C-order) strides from shape.
/// Last axis has stride 1, each preceding axis has stride = product of subsequent dims.
pub fn row_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

/// Compute column-major (Fortran-order) strides from shape.
/// First axis has stride 1, each subsequent axis has stride = product of preceding dims.
pub fn column_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}

// ============================================================================
// Basic accessors
// ============================================================================

impl<T> DenseTensor<T> {
    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the rank (number of dimensions) of the tensor
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get shape as i64 slice for MLIR compatibility
    pub fn shape_i64(&self) -> Vec<i64> {
        self.shape.iter().map(|&s| s as i64).collect()
    }

    /// Get the total number of logical elements
    pub fn len(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the strides of the tensor
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Get the offset into the data buffer
    pub fn offset(&self) -> usize {
        self.offset
    }

    // ========================================================================
    // Layout queries
    // ========================================================================

    /// Check if the tensor is contiguous in any standard order.
    pub fn is_contiguous(&self) -> bool {
        self.is_row_major() || self.is_column_major()
    }

    /// Check if strides match row-major (C-order) layout.
    pub fn is_row_major(&self) -> bool {
        self.strides == row_major_strides(&self.shape)
    }

    /// Check if strides match column-major (Fortran-order) layout.
    pub fn is_column_major(&self) -> bool {
        self.strides == column_major_strides(&self.shape)
    }

    /// The memory order this tensor was created with.
    ///
    /// Unlike `is_row_major()` / `is_column_major()` which check strides,
    /// this returns the authoritative order that disambiguates cases where
    /// strides are ambiguous (e.g., 1D tensors or tensors with size-1 dims).
    pub fn memory_order(&self) -> MemoryOrder {
        self.order
    }

    /// Determine the memory order of this tensor, if contiguous.
    fn contiguous_order(&self) -> Option<MemoryOrder> {
        if !self.is_contiguous() {
            return None;
        }
        // Use the authoritative order field, not strides-based heuristic
        Some(self.order)
    }
}

// ============================================================================
// Constructors
// ============================================================================

impl<T> DenseTensor<T>
where
    T: Clone,
{
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero,
    {
        let total_elements: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total_elements);
        data.resize(total_elements, T::zero());
        let strides = row_major_strides(&shape);

        Self {
            data: Arc::new(data),
            strides,
            shape,
            offset: 0,
            order: MemoryOrder::RowMajor,
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Zero,
    {
        let total_elements: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total_elements);
        data.resize(total_elements, T::one());
        let strides = row_major_strides(&shape);

        Self {
            data: Arc::new(data),
            strides,
            shape,
            offset: 0,
            order: MemoryOrder::RowMajor,
        }
    }

    /// Create a tensor filled with a constant value
    pub fn constant(shape: Vec<usize>, value: T) -> Self {
        let total_elements: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total_elements);
        data.resize(total_elements, value);
        let strides = row_major_strides(&shape);

        Self {
            data: Arc::new(data),
            strides,
            shape,
            offset: 0,
            order: MemoryOrder::RowMajor,
        }
    }

    /// Create an n×n identity matrix in row-major order.
    pub fn eye(n: usize) -> Self
    where
        T: Zero + One,
    {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Self::from_data_with_order(data, vec![n, n], MemoryOrder::RowMajor)
    }

    /// Create a tensor from data with explicit strides and offset.
    ///
    /// Used by backends to produce tensors in non-row-major layouts.
    ///
    /// # Panics
    ///
    /// Panics if any logical index would address outside the data buffer,
    /// or if shape and strides ranks differ.
    pub fn from_data_with_strides(
        data: Vec<T>,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
        order: MemoryOrder,
    ) -> Self {
        assert_eq!(
            shape.len(),
            strides.len(),
            "Shape rank {} doesn't match strides rank {}",
            shape.len(),
            strides.len()
        );

        // Validate that all reachable indices are within bounds.
        let data_len = data.len();

        // Empty tensors still need offset within buffer (for data()/as_ptr() safety)
        assert!(
            offset <= data_len,
            "from_data_with_strides: offset {offset} exceeds data buffer of length {data_len}"
        );

        if !shape.contains(&0) {
            let mut min_offset: isize = offset as isize;
            let mut max_offset: isize = offset as isize;
            for (&dim, &stride) in shape.iter().zip(&strides) {
                let end = stride * (dim as isize - 1);
                if end >= 0 {
                    max_offset += end;
                } else {
                    min_offset += end;
                }
            }
            assert!(
                min_offset >= 0 && (max_offset as usize) < data_len,
                "from_data_with_strides: reachable index range [{min_offset}, {max_offset}] \
                 exceeds data buffer of length {data_len}"
            );
        }

        let mut aligned_data: AVec<T, Align64> = AVec::with_capacity(64, data_len);
        for elem in data {
            aligned_data.push(elem);
        }

        Self {
            data: Arc::new(aligned_data),
            shape,
            strides,
            offset,
            order,
        }
    }

    /// Create a tensor from existing data with the specified memory order.
    ///
    /// This is the primary order-explicit constructor. Strides are computed
    /// automatically based on the memory order.
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match the shape.
    pub fn from_data_with_order(data: Vec<T>, shape: Vec<usize>, order: MemoryOrder) -> Self {
        let strides = match order {
            MemoryOrder::RowMajor => row_major_strides(&shape),
            MemoryOrder::ColumnMajor => column_major_strides(&shape),
        };
        Self::from_data_with_strides(data, shape, strides, 0, order)
    }

    /// Create a tensor filled with random values from the standard distribution.
    #[cfg(feature = "random")]
    pub fn random<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let total: usize = shape.iter().product();
        let data: Vec<T> = (0..total).map(|_| rng.random()).collect();
        Self::from_data_with_order(data, shape, MemoryOrder::RowMajor)
    }

    // ========================================================================
    // Data access
    // ========================================================================

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

    // ========================================================================
    // Reshape
    // ========================================================================

    /// Reshape the tensor to a new shape.
    ///
    /// Zero-copy if strides are compatible with the new shape.
    /// Otherwise, copies to contiguous layout first.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different total number of elements.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_total: usize = new_shape.iter().product();
        assert_eq!(
            self.len(),
            new_total,
            "reshape: total elements must match ({} vs {new_total})",
            self.len()
        );

        if let Some(view) = self.reshape_view(new_shape.clone()) {
            return view;
        }

        // Non-contiguous: copy to contiguous first, then reshape
        let contiguous = self.to_contiguous(MemoryOrder::RowMajor);
        contiguous
            .reshape_view(new_shape)
            .expect("reshape_view failed on contiguous tensor")
    }

    /// Zero-copy reshape if strides are compatible with the new shape.
    ///
    /// Returns `None` if the tensor must be copied to support the new shape.
    pub fn reshape_view(&self, new_shape: Vec<usize>) -> Option<Self> {
        let new_total: usize = new_shape.iter().product();
        if self.len() != new_total {
            return None;
        }

        // For contiguous tensors, reshape is always zero-copy:
        // just compute new strides in the same memory order.
        if let Some(order) = self.contiguous_order() {
            let new_strides = match order {
                MemoryOrder::RowMajor => row_major_strides(&new_shape),
                MemoryOrder::ColumnMajor => column_major_strides(&new_shape),
            };
            return Some(Self {
                data: Arc::clone(&self.data),
                shape: new_shape,
                strides: new_strides,
                offset: self.offset,
                order,
            });
        }

        // Non-contiguous: cannot reshape without copying
        None
    }

    // ========================================================================
    // Contiguity conversion
    // ========================================================================

    /// Create a contiguous copy in the specified memory order.
    ///
    /// No-op (Arc clone) if already contiguous in the requested order.
    pub fn to_contiguous(&self, order: MemoryOrder) -> Self {
        let already_ok = match order {
            MemoryOrder::RowMajor => self.is_row_major() && self.offset == 0,
            MemoryOrder::ColumnMajor => self.is_column_major() && self.offset == 0,
        };

        if already_ok {
            return self.clone();
        }

        let total = self.len();
        let new_strides = match order {
            MemoryOrder::RowMajor => row_major_strides(&self.shape),
            MemoryOrder::ColumnMajor => column_major_strides(&self.shape),
        };

        // Iterate through all logical indices in the target order and copy
        let mut new_data = Vec::with_capacity(total);
        let rank = self.rank();
        let mut coords = vec![0usize; rank];

        // Iteration order depends on target layout
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        for _ in 0..total {
            new_data.push(self.get(&coords));

            // Increment coordinates in the target order
            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < self.shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data_with_strides(new_data, self.shape.clone(), new_strides, 0, order)
    }

    // ========================================================================
    // Element-wise operations
    // ========================================================================

    /// Apply a function to each element, preserving the tensor's memory order.
    ///
    /// Iterates over contiguous data directly for efficiency.
    pub fn map<U, F>(&self, f: F) -> DenseTensor<U>
    where
        F: Fn(&T) -> U,
        U: Clone + 'static,
    {
        let order = self.memory_order();
        let contiguous = self.to_contiguous(order);
        let result: Vec<U> = contiguous.data_contiguous().iter().map(f).collect();
        DenseTensor::from_data_with_order(result, self.shape().to_vec(), order)
    }

    /// Apply a function to each element in place (triggers CoW if shared).
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous.
    pub fn map_mut<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        assert!(
            self.is_contiguous(),
            "map_mut() requires contiguous tensor; \
             call to_contiguous() first"
        );
        let data = self.data_contiguous_mut();
        for x in data.iter_mut() {
            *x = f(x);
        }
    }

    /// Apply a function with multi-dimensional coordinates to each element,
    /// preserving the tensor's memory order.
    pub fn map_with_index<U, F>(&self, f: F) -> DenseTensor<U>
    where
        F: Fn(&[usize], &T) -> U,
        U: Clone + 'static,
    {
        let shape = self.shape();
        let rank = shape.len();
        let total = self.len();
        let order = self.memory_order();
        let mut coords = vec![0usize; rank];
        let mut result = Vec::with_capacity(total);

        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        for _ in 0..total {
            let val = self.get(&coords);
            result.push(f(&coords, &val));

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        DenseTensor::from_data_with_order(result, shape.to_vec(), order)
    }

    // ========================================================================
    // Arithmetic operations
    // ========================================================================

    /// Scale all elements by a scalar factor (in-place).
    ///
    /// Preserves the tensor's memory order.
    pub fn scale<S>(&mut self, factor: S)
    where
        T: Mul<S, Output = T>,
        S: Clone,
    {
        *self = self.to_contiguous(self.memory_order());
        let data = self.data_contiguous_mut();
        for elem in data.iter_mut() {
            *elem = elem.clone() * factor.clone();
        }
    }

    /// Linear combination: Σ coefs\[i\] * tensors\[i\].
    ///
    /// Output memory order matches the first tensor's order.
    ///
    /// # Errors
    ///
    /// Returns an error if tensors have different shapes, the list is empty,
    /// or tensors and coefficients have different lengths.
    pub fn linear_combine(
        tensors: &[&DenseTensor<T>],
        coefs: &[T],
    ) -> Result<DenseTensor<T>, String>
    where
        T: Zero + Add<Output = T> + Mul<Output = T>,
    {
        if tensors.is_empty() {
            return Err("Cannot combine empty tensor list".to_string());
        }
        if tensors.len() != coefs.len() {
            return Err(format!(
                "Mismatched lengths: {} tensors vs {} coefficients",
                tensors.len(),
                coefs.len()
            ));
        }
        let shape = tensors[0].shape();
        for t in &tensors[1..] {
            if t.shape() != shape {
                return Err("All tensors must have the same shape".to_string());
            }
        }
        let order = tensors[0].memory_order();
        let len = tensors[0].len();
        let mut result = vec![T::zero(); len];
        for (tensor, coef) in tensors.iter().zip(coefs) {
            let c = tensor.to_contiguous(order);
            for (r, val) in result.iter_mut().zip(c.data_contiguous()) {
                *r = r.clone() + coef.clone() * val.clone();
            }
        }
        Ok(DenseTensor::from_data_with_order(
            result,
            shape.to_vec(),
            order,
        ))
    }

    // ========================================================================
    // Slice / expand / replace
    // ========================================================================

    /// Extract a sub-tensor by specifying a range for each axis.
    ///
    /// Each range is `(start, end)` with exclusive end.
    ///
    /// # Panics
    ///
    /// Panics if `ranges` length doesn't match rank, or any range is out of bounds.
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Self {
        let shape = self.shape();
        assert_eq!(
            ranges.len(),
            shape.len(),
            "slice: ranges length {} doesn't match rank {}",
            ranges.len(),
            shape.len()
        );
        for (i, &(start, end)) in ranges.iter().enumerate() {
            assert!(
                start <= end && end <= shape[i],
                "slice: range ({start}, {end}) out of bounds for axis {i} with size {}",
                shape[i]
            );
        }

        let new_shape: Vec<usize> = ranges.iter().map(|&(s, e)| e - s).collect();
        let new_total: usize = new_shape.iter().product();
        let order = self.memory_order();
        let mut data = Vec::with_capacity(new_total);

        let rank = shape.len();
        let mut coords = vec![0usize; rank];
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        for _ in 0..new_total {
            let src_coords: Vec<usize> = coords
                .iter()
                .zip(ranges)
                .map(|(&c, &(s, _))| c + s)
                .collect();
            data.push(self.get(&src_coords));

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < new_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data_with_order(data, new_shape, order)
    }

    /// Expand tensor by adding zero-padding at the boundaries.
    pub fn expand(&self, padding: &[(usize, usize)]) -> Self
    where
        T: Zero,
    {
        let shape = self.shape();
        assert_eq!(
            padding.len(),
            shape.len(),
            "expand: padding length {} doesn't match rank {}",
            padding.len(),
            shape.len()
        );

        let new_shape: Vec<usize> = shape
            .iter()
            .zip(padding)
            .map(|(&s, &(before, after))| s + before + after)
            .collect();
        let new_total: usize = new_shape.iter().product();
        let order = self.memory_order();
        let dst_strides = match order {
            MemoryOrder::RowMajor => compute_strides_usize(&new_shape),
            MemoryOrder::ColumnMajor => compute_strides_column_usize(&new_shape),
        };
        let rank = shape.len();
        let mut data = vec![T::zero(); new_total];
        let mut coords = vec![0usize; rank];
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        let src_total = self.len();
        for _ in 0..src_total {
            let val = self.get(&coords);
            let dst_flat: usize = (0..rank)
                .map(|d| (coords[d] + padding[d].0) * dst_strides[d])
                .sum();
            data[dst_flat] = val;

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data_with_order(data, new_shape, order)
    }

    /// Write a sub-tensor into this tensor starting at the given position.
    pub fn replace_slice(&mut self, sub: &Self, begin: &[usize]) {
        let shape = self.shape().to_vec();
        let sub_shape = sub.shape();
        assert_eq!(
            begin.len(),
            shape.len(),
            "replace_slice: begin length {} doesn't match rank {}",
            begin.len(),
            shape.len()
        );
        for (d, (&b, &ss)) in begin.iter().zip(sub_shape).enumerate() {
            assert!(
                b + ss <= shape[d],
                "replace_slice: sub-tensor exceeds boundary on axis {d} ({b} + {ss} > {})",
                shape[d]
            );
        }

        let rank = shape.len();
        let sub_total = sub.len();
        let mut coords = vec![0usize; rank];

        for _ in 0..sub_total {
            let val = sub.get(&coords);
            let dst_coords: Vec<usize> = coords.iter().zip(begin).map(|(&c, &b)| c + b).collect();
            self.set(&dst_coords, val);

            for d in (0..rank).rev() {
                coords[d] += 1;
                if coords[d] < sub_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }
    }

    // ========================================================================
    // Multi-tensor operations
    // ========================================================================

    /// Concatenate tensors along an existing axis.
    ///
    /// Output memory order matches the first tensor's order.
    /// Inputs may be any layout.
    pub fn concatenate(tensors: &[&DenseTensor<T>], axis: usize) -> Self {
        assert!(!tensors.is_empty(), "concatenate: empty tensor list");
        let rank = tensors[0].rank();
        assert!(
            axis < rank,
            "concatenate: axis {axis} out of range for rank {rank}"
        );

        let base_shape = tensors[0].shape();
        let order = tensors[0].memory_order();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            assert_eq!(
                t.rank(),
                rank,
                "concatenate: tensor {i} has rank {} but expected {rank}",
                t.rank()
            );
            for (d, (&ts, &bs)) in t.shape().iter().zip(base_shape).enumerate() {
                if d != axis {
                    assert_eq!(
                        ts, bs,
                        "concatenate: tensor {i} has size {ts} on axis {d} but expected {bs}",
                    );
                }
            }
        }

        let mut out_shape: Vec<usize> = base_shape.to_vec();
        out_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();

        let out_total: usize = out_shape.iter().product();
        let mut data = Vec::with_capacity(out_total);
        let mut coords = vec![0usize; rank];
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        for _ in 0..out_total {
            // Determine which input tensor and local coordinate for the concat axis
            let mut axis_pos = coords[axis];
            let mut src_tensor = None;
            for t in tensors {
                let t_size = t.shape()[axis];
                if axis_pos < t_size {
                    src_tensor = Some(t);
                    break;
                }
                axis_pos -= t_size;
            }
            let t = src_tensor.expect("concatenate: axis position out of range");
            let mut src_coords = coords.clone();
            src_coords[axis] = axis_pos;
            data.push(t.get(&src_coords));

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < out_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data_with_order(data, out_shape, order)
    }

    /// Stack tensors along a new axis.
    ///
    /// Output memory order matches the first tensor's order.
    /// Inputs may be any layout.
    pub fn stack(tensors: &[&DenseTensor<T>], axis: usize) -> Self {
        assert!(!tensors.is_empty(), "stack: empty tensor list");
        let base_shape = tensors[0].shape();
        let rank = tensors[0].rank();
        let order = tensors[0].memory_order();
        assert!(
            axis <= rank,
            "stack: axis {axis} out of range for rank {rank} (max {rank})"
        );

        for (i, t) in tensors.iter().enumerate().skip(1) {
            assert_eq!(
                t.shape(),
                base_shape,
                "stack: tensor {i} has shape {:?} but expected {base_shape:?}",
                t.shape()
            );
        }

        let n = tensors.len();
        let mut out_shape = Vec::with_capacity(rank + 1);
        out_shape.extend_from_slice(&base_shape[..axis]);
        out_shape.push(n);
        out_shape.extend_from_slice(&base_shape[axis..]);

        let out_total: usize = out_shape.iter().product();
        let out_rank = out_shape.len();
        let mut data = Vec::with_capacity(out_total);
        let mut coords = vec![0usize; out_rank];
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..out_rank).collect(),
            MemoryOrder::ColumnMajor => (0..out_rank).rev().collect(),
        };

        for _ in 0..out_total {
            // The stacked axis at position `axis` indexes into tensors
            let t_idx = coords[axis];
            let mut src_coords: Vec<usize> = coords[..axis].to_vec();
            src_coords.extend_from_slice(&coords[axis + 1..]);
            data.push(tensors[t_idx].get(&src_coords));

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < out_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data_with_order(data, out_shape, order)
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Convert multi-dimensional indices to flat index using strides and offset.
    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "Number of indices {} doesn't match tensor rank {}",
            indices.len(),
            self.shape.len()
        );

        indices.iter().zip(&self.shape).for_each(|(&idx, &dim)| {
            assert!(
                idx < dim,
                "Index {} out of bounds for dimension {}",
                idx,
                dim
            )
        });

        let raw: isize = indices
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx as isize * stride)
            .sum();
        (self.offset as isize + raw) as usize
    }
}

// ============================================================================
// Scalar-dependent operations
// ============================================================================

impl<T> DenseTensor<T>
where
    T: arnet_core::scalar::Scalar,
{
    /// Element-wise complex conjugate.
    pub fn conj(&self) -> Self {
        self.map(|x| x.conj())
    }

    /// Convert each element to its complex representation.
    pub fn to_complex(&self) -> DenseTensor<T::Complex> {
        self.map(|x| x.into_complex())
    }

    /// Extract the real part of each element.
    pub fn real(&self) -> DenseTensor<T::Real> {
        self.map(|x| x.re())
    }

    /// Extract the imaginary part of each element.
    pub fn imag(&self) -> DenseTensor<T::Real> {
        self.map(|x| x.im())
    }

    /// Compute squared Frobenius norm: Σ |element|².
    fn norm_squared(&self) -> T::Real {
        let c = self.to_contiguous(self.memory_order());
        c.data_contiguous()
            .iter()
            .map(|&x| {
                let a = x.abs();
                a * a
            })
            .fold(T::Real::zero(), |acc, x| acc + x)
    }

    /// Compute Frobenius norm: √(Σ |element|²).
    pub fn norm_frobenius(&self) -> T::Real {
        self.norm_squared().sqrt()
    }

    /// Normalize to unit Frobenius norm (in-place).
    ///
    /// Returns the norm before normalization.
    /// Panics if the tensor has zero norm.
    pub fn normalize_in_place(&mut self) -> T::Real {
        let norm = self.norm_frobenius();
        assert!(norm != T::Real::zero(), "Cannot normalize zero tensor");
        let inv_norm = T::Real::one() / norm;
        *self = self.to_contiguous(self.memory_order());
        let data = self.data_contiguous_mut();
        for elem in data.iter_mut() {
            *elem = elem.scale_real(inv_norm);
        }
        norm
    }
}

// ============================================================================
// Display / Debug
// ============================================================================

impl<T> fmt::Debug for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DenseTensor(shape={:?}, strides={:?}, offset={}, elements={})",
            self.shape,
            self.strides,
            self.offset,
            self.len()
        )
    }
}

impl<T> fmt::Display for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenseTensor{:?}", self.shape)
    }
}

/// Compute row-major strides as usize (for internal use in expand/replace_slice).
fn compute_strides_usize(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute column-major strides as usize (for internal use in expand).
fn compute_strides_column_usize(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    strides
}
