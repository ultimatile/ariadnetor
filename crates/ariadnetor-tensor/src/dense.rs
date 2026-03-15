//! Dense tensor storage
//!
//! Provides a dense tensor with row-major layout and Arc-based shared ownership.

use aligned_vec::{AVec, ConstAlign};
use num_traits::{One, Zero};
use std::fmt;
use std::sync::Arc;

/// 64-byte alignment for SIMD (AVX-512)
type Align64 = ConstAlign<64>;

/// Dense tensor with shared ownership (Arc + Copy-on-Write)
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64). Commonly used types:
///   - `f64`: Double precision floating point
///   - `f32`: Single precision floating point
///   - `Complex<f64>`: Double precision complex numbers
///   - `Complex<f32>`: Single precision complex numbers
///
/// # Memory Management
///
/// Uses `Arc<Vec<T>>` for efficient cloning:
/// - Cloning is O(1) (only increments reference count)
/// - Mutation triggers Copy-on-Write if reference count > 1
/// - Ideal for read-heavy workloads
///
/// # Layout
///
/// Data is stored in row-major order (C-contiguous).
/// For a 2D tensor `[2, 3]`:
/// ```text
/// [[a, b, c],
///  [d, e, f]]
/// → [a, b, c, d, e, f]
/// ```
#[derive(Clone)]
pub struct DenseTensor<T = f64> {
    /// Shared data buffer (row-major order, 64-byte aligned)
    data: Arc<AVec<T, Align64>>,
    /// Tensor shape
    shape: Vec<usize>,
}

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
    ///
    /// MLIR uses i64 for tensor dimensions, so we need conversion from usize.
    pub fn shape_i64(&self) -> Vec<i64> {
        self.shape.iter().map(|&s| s as i64).collect()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reshape the tensor to a new shape without copying data.
    ///
    /// The total number of elements must remain the same.
    /// The returned tensor shares the underlying data via `Arc` (zero-copy).
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
        Self {
            data: Arc::clone(&self.data),
            shape: new_shape,
        }
    }
}

impl<T> DenseTensor<T>
where
    T: Clone,
{
    /// Create a new tensor filled with zeros
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use arnet_tensor::DenseTensor;
    ///
    /// let tensor = DenseTensor::zeros(vec![10, 20]);
    /// assert_eq!(tensor.shape(), &[10, 20]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero,
    {
        let total_elements: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total_elements);
        data.resize(total_elements, T::zero());

        Self {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create a tensor filled with ones
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tensor = DenseTensor::ones(vec![5, 5]);
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Zero,
    {
        let total_elements: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total_elements);
        data.resize(total_elements, T::one());

        Self {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create a tensor filled with a constant value
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tensor = DenseTensor::constant(vec![3, 3], 3.14);
    /// ```
    pub fn constant(shape: Vec<usize>, value: T) -> Self {
        let total_elements: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total_elements);
        data.resize(total_elements, value);

        Self {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create an n×n identity matrix.
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix dimension
    pub fn eye(n: usize) -> Self
    where
        T: Zero + One,
    {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Self::from_data(data, vec![n, n])
    }

    /// Create a tensor from existing data
    ///
    /// # Arguments
    ///
    /// * `data` - Tensor data in row-major order
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match the shape
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            total_elements,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            total_elements
        );

        let mut aligned_data: AVec<T, Align64> = AVec::with_capacity(64, total_elements);
        for elem in data {
            aligned_data.push(elem);
        }

        Self {
            data: Arc::new(aligned_data),
            shape,
        }
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable reference to the underlying data (triggers CoW if shared)
    ///
    /// # Copy-on-Write
    ///
    /// If the data is shared (Arc reference count > 1), this will clone the data
    /// before returning a mutable reference.
    pub fn data_mut(&mut self) -> &mut [T] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// Get element at given indices
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn get(&self, indices: &[usize]) -> T {
        let flat_index = self.flat_index(indices);
        self.data[flat_index].clone()
    }

    /// Set element at given indices (triggers CoW if shared)
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn set(&mut self, indices: &[usize], value: T) {
        let flat_index = self.flat_index(indices);
        Arc::make_mut(&mut self.data)[flat_index] = value;
    }

    /// Fill tensor with a constant value (triggers CoW if shared)
    pub fn fill(&mut self, value: T) {
        Arc::make_mut(&mut self.data).fill(value);
    }

    /// Get pointer to the underlying data for FFI
    ///
    /// Returns a pointer that can be passed to JIT-compiled functions.
    /// The pointer remains valid as long as the DenseTensor is not moved or dropped.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get mutable pointer to the underlying data for FFI (triggers CoW if shared)
    ///
    /// Returns a mutable pointer for writing results from JIT-compiled functions.
    /// The pointer remains valid as long as the DenseTensor is not moved or dropped.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        Arc::make_mut(&mut self.data).as_mut_ptr()
    }

    /// Apply a function to each element, producing a new tensor.
    ///
    /// Supports type-changing transforms (e.g., `DenseTensor<f64>` → `DenseTensor<Complex<f64>>`).
    pub fn map<U, F>(&self, f: F) -> DenseTensor<U>
    where
        F: Fn(&T) -> U,
        U: Clone + 'static,
    {
        let data: Vec<U> = self.data().iter().map(&f).collect();
        DenseTensor::from_data(data, self.shape().to_vec())
    }

    /// Apply a function to each element in place (triggers CoW if shared).
    pub fn map_mut<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        let data = self.data_mut();
        for x in data.iter_mut() {
            *x = f(x);
        }
    }

    /// Apply a function with multi-dimensional coordinates to each element.
    ///
    /// The closure receives `(&[usize], &T)` where the first argument is the
    /// multi-dimensional index.
    pub fn map_with_index<U, F>(&self, f: F) -> DenseTensor<U>
    where
        F: Fn(&[usize], &T) -> U,
        U: Clone + 'static,
    {
        let shape = self.shape();
        let mut coords = vec![0usize; shape.len()];
        let data: Vec<U> = self
            .data()
            .iter()
            .enumerate()
            .map(|(flat, x)| {
                // Decode flat index to coordinates
                let mut rem = flat;
                for i in (0..shape.len()).rev() {
                    coords[i] = rem % shape[i];
                    rem /= shape[i];
                }
                f(&coords, x)
            })
            .collect();
        DenseTensor::from_data(data, self.shape().to_vec())
    }

    /// Extract a sub-tensor by specifying a range for each axis.
    ///
    /// Each range is `(start, end)` with exclusive end, similar to Rust's `start..end`.
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
        let mut data = Vec::with_capacity(new_total);

        let src_strides = Self::compute_strides(shape);
        let rank = shape.len();
        let mut coords = vec![0usize; rank];

        for _ in 0..new_total {
            let flat: usize = (0..rank)
                .map(|d| (coords[d] + ranges[d].0) * src_strides[d])
                .sum();
            data.push(self.data[flat].clone());

            // Increment coordinates (row-major order)
            for d in (0..rank).rev() {
                coords[d] += 1;
                if coords[d] < new_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data(data, new_shape)
    }

    /// Expand tensor by adding zero-padding at the boundaries.
    ///
    /// `padding` specifies `(before, after)` padding for each axis.
    ///
    /// # Panics
    ///
    /// Panics if `padding` length doesn't match rank.
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
        let mut data = vec![T::zero(); new_total];

        let src_strides = Self::compute_strides(shape);
        let dst_strides = Self::compute_strides(&new_shape);
        let rank = shape.len();
        let mut coords = vec![0usize; rank];

        for _ in 0..self.len() {
            let src_flat: usize = (0..rank).map(|d| coords[d] * src_strides[d]).sum();
            let dst_flat: usize = (0..rank)
                .map(|d| (coords[d] + padding[d].0) * dst_strides[d])
                .sum();
            data[dst_flat] = self.data[src_flat].clone();

            for d in (0..rank).rev() {
                coords[d] += 1;
                if coords[d] < shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data(data, new_shape)
    }

    /// Write a sub-tensor into this tensor starting at the given position.
    ///
    /// # Panics
    ///
    /// Panics if `begin` length doesn't match rank, or the sub-tensor
    /// would extend beyond the boundaries of this tensor.
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

        let dst_strides = Self::compute_strides(&shape);
        let src_strides = Self::compute_strides(sub_shape);
        let rank = shape.len();
        let mut coords = vec![0usize; rank];
        let data = self.data_mut();

        for _ in 0..sub.len() {
            let src_flat: usize = (0..rank).map(|d| coords[d] * src_strides[d]).sum();
            let dst_flat: usize = (0..rank)
                .map(|d| (coords[d] + begin[d]) * dst_strides[d])
                .sum();
            data[dst_flat] = sub.data()[src_flat].clone();

            for d in (0..rank).rev() {
                coords[d] += 1;
                if coords[d] < sub_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }
    }

    /// Concatenate tensors along an existing axis.
    ///
    /// All tensors must have the same shape except along the concatenation axis.
    ///
    /// # Panics
    ///
    /// Panics if `tensors` is empty, `axis` is out of range, or shapes are
    /// incompatible on non-concatenation axes.
    pub fn concatenate(tensors: &[&DenseTensor<T>], axis: usize) -> Self {
        assert!(!tensors.is_empty(), "concatenate: empty tensor list");
        let rank = tensors[0].rank();
        assert!(axis < rank, "concatenate: axis {axis} out of range for rank {rank}");

        let base_shape = tensors[0].shape();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            assert_eq!(
                t.rank(), rank,
                "concatenate: tensor {i} has rank {} but expected {rank}", t.rank()
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

        // Size of a "block" along and after the concat axis
        let outer: usize = out_shape[..axis].iter().product();
        let inner: usize = if axis + 1 < rank {
            out_shape[axis + 1..].iter().product()
        } else {
            1
        };

        for o in 0..outer {
            for t in tensors {
                let t_axis_size = t.shape()[axis];
                let t_inner_stride = t_axis_size * inner;
                let src_offset = o * t_inner_stride;
                data.extend_from_slice(&t.data()[src_offset..src_offset + t_inner_stride]);
            }
        }

        Self::from_data(data, out_shape)
    }

    /// Stack tensors along a new axis.
    ///
    /// All tensors must have identical shapes. A new axis is inserted at position `axis`.
    ///
    /// # Panics
    ///
    /// Panics if `tensors` is empty, `axis > rank`, or shapes differ.
    pub fn stack(tensors: &[&DenseTensor<T>], axis: usize) -> Self {
        assert!(!tensors.is_empty(), "stack: empty tensor list");
        let base_shape = tensors[0].shape();
        let rank = tensors[0].rank();
        assert!(
            axis <= rank,
            "stack: axis {axis} out of range for rank {rank} (max {rank})"
        );

        for (i, t) in tensors.iter().enumerate().skip(1) {
            assert_eq!(
                t.shape(), base_shape,
                "stack: tensor {i} has shape {:?} but expected {base_shape:?}",
                t.shape()
            );
        }

        // Insert new axis: e.g., shape [2, 3] stacked at axis 0 with n=3 → [3, 2, 3]
        let n = tensors.len();
        let mut out_shape = Vec::with_capacity(rank + 1);
        out_shape.extend_from_slice(&base_shape[..axis]);
        out_shape.push(n);
        out_shape.extend_from_slice(&base_shape[axis..]);

        let out_total: usize = out_shape.iter().product();
        let mut data = Vec::with_capacity(out_total);

        let outer: usize = base_shape[..axis].iter().product();
        let inner: usize = base_shape[axis..].iter().product();

        for o in 0..outer {
            for t in tensors {
                let src_offset = o * inner;
                data.extend_from_slice(&t.data()[src_offset..src_offset + inner]);
            }
        }

        Self::from_data(data, out_shape)
    }

    /// Permute tensor axes (pure-Rust naive implementation)
    ///
    /// For optimized transpose using HPTT or other backends, use
    /// `arnet_linalg::transpose()` with a `ComputeBackend`.
    ///
    /// # Arguments
    ///
    /// * `perm` - Permutation of axes (e.g., `[1, 0]` transposes a 2D tensor)
    ///
    /// # Panics
    ///
    /// Panics if the permutation is invalid (wrong length or duplicate indices)
    pub fn permute(&self, perm: &[usize]) -> Self
    where
        T: Zero,
    {
        self.permute_naive(perm)
    }

    /// Naive tensor permutation implementation
    ///
    /// This is the fallback implementation that works for all types and devices.
    /// For f32/f64, prefer using `permute()` which automatically selects HPTT.
    pub fn permute_naive(&self, perm: &[usize]) -> Self
    where
        T: Zero,
    {
        self.validate_permutation(perm);

        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let new_strides = Self::compute_strides(&new_shape);
        let old_strides = Self::compute_strides(&self.shape);
        let total_elements = self.len();
        let rank = self.rank();

        let mut result_data = vec![T::zero(); total_elements];

        // Iterate over all elements and reorder
        for old_idx in 0..total_elements {
            let old_coords = Self::linear_to_coords_with_strides(old_idx, &old_strides, rank);
            let new_coords: Vec<usize> = perm.iter().map(|&i| old_coords[i]).collect();
            let new_idx = Self::coords_to_linear(&new_coords, &new_strides);

            result_data[new_idx] = self.data[old_idx].clone();
        }

        Self::from_data(result_data, new_shape)
    }

    /// Validate permutation
    fn validate_permutation(&self, perm: &[usize]) {
        assert_eq!(
            perm.len(),
            self.rank(),
            "Permutation length {} doesn't match tensor rank {}",
            perm.len(),
            self.rank()
        );

        let mut seen = vec![false; self.rank()];
        for &i in perm {
            assert!(
                i < self.rank(),
                "Permutation index {} out of range [0, {})",
                i,
                self.rank()
            );
            assert!(!seen[i], "Duplicate index {} in permutation", i);
            seen[i] = true;
        }
    }

    /// Convert linear index to multi-dimensional coordinates using precomputed strides
    fn linear_to_coords_with_strides(idx: usize, strides: &[usize], rank: usize) -> Vec<usize> {
        let mut coords = vec![0; rank];
        let mut remaining = idx;

        for i in 0..rank {
            coords[i] = remaining / strides[i];
            remaining %= strides[i];
        }

        coords
    }

    /// Convert multi-dimensional coordinates to linear index
    fn coords_to_linear(coords: &[usize], strides: &[usize]) -> usize {
        coords
            .iter()
            .zip(strides.iter())
            .map(|(&c, &s)| c * s)
            .sum()
    }

    /// Compute strides for row-major layout
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Convert multi-dimensional indices to flat index
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

        let strides = Self::compute_strides(&self.shape);
        indices
            .iter()
            .zip(&strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

}

impl<T> DenseTensor<T>
where
    T: arnet_core::scalar::Scalar,
{
    /// Element-wise complex conjugate.
    pub fn conj(&self) -> Self {
        let data: Vec<T> = self.data().iter().map(|&x| x.conj()).collect();
        Self::from_data(data, self.shape().to_vec())
    }

    /// Convert each element to its complex representation.
    ///
    /// For real types (f64, f32), wraps each value as `Complex::new(x, 0)`.
    /// For complex types, this is the identity operation.
    pub fn to_complex(&self) -> DenseTensor<T::Complex> {
        let data: Vec<T::Complex> = self.data().iter().map(|&x| x.into_complex()).collect();
        DenseTensor::from_data(data, self.shape().to_vec())
    }

    /// Extract the real part of each element.
    ///
    /// For real types, returns a copy of the tensor.
    /// For complex types, extracts the real component.
    pub fn real(&self) -> DenseTensor<T::Real> {
        let data: Vec<T::Real> = self.data().iter().map(|&x| x.re()).collect();
        DenseTensor::from_data(data, self.shape().to_vec())
    }

    /// Extract the imaginary part of each element.
    ///
    /// For real types, returns a tensor of zeros.
    /// For complex types, extracts the imaginary component.
    pub fn imag(&self) -> DenseTensor<T::Real> {
        let data: Vec<T::Real> = self.data().iter().map(|&x| x.im()).collect();
        DenseTensor::from_data(data, self.shape().to_vec())
    }
}

impl<T> fmt::Debug for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DenseTensor(shape={:?}, elements={})",
            self.shape,
            self.len()
        )
    }
}

impl<T> fmt::Display for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenseTensor{:?}", self.shape)
    }
}
