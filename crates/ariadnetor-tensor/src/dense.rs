//! Dense tensor storage
//!
//! Provides a dense tensor with row-major layout and Arc-based shared ownership.

use std::fmt;
use std::sync::Arc;

/// Dense tensor with shared ownership (Arc + Copy-on-Write)
///
/// # Memory Management
///
/// Uses `Arc<Vec<f64>>` for efficient cloning:
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
pub struct DenseTensor {
    /// Shared data buffer (row-major order)
    data: Arc<Vec<f64>>,
    /// Tensor shape
    shape: Vec<usize>,
    /// Strides for indexing (row-major)
    strides: Vec<usize>,
}

impl DenseTensor {
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
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = Arc::new(vec![0.0; total_elements]);
        let strides = Self::compute_strides(&shape);

        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create a tensor filled with ones
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tensor = DenseTensor::ones(vec![5, 5]);
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self {
        let mut tensor = Self::zeros(shape);
        tensor.fill(1.0);
        tensor
    }

    /// Create a tensor filled with a constant value
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tensor = DenseTensor::constant(vec![3, 3], 3.14);
    /// ```
    pub fn constant(shape: Vec<usize>, value: f64) -> Self {
        let mut tensor = Self::zeros(shape);
        tensor.fill(value);
        tensor
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
    pub fn from_data(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            total_elements,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            total_elements
        );

        let strides = Self::compute_strides(&shape);

        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the rank (number of dimensions) of the tensor
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get a mutable reference to the underlying data (triggers CoW if shared)
    ///
    /// # Copy-on-Write
    ///
    /// If the data is shared (Arc reference count > 1), this will clone the data
    /// before returning a mutable reference.
    pub fn data_mut(&mut self) -> &mut [f64] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// Get element at given indices
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn get(&self, indices: &[usize]) -> f64 {
        let flat_index = self.flat_index(indices);
        self.data[flat_index]
    }

    /// Set element at given indices (triggers CoW if shared)
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn set(&mut self, indices: &[usize], value: f64) {
        let flat_index = self.flat_index(indices);
        Arc::make_mut(&mut self.data)[flat_index] = value;
    }

    /// Fill tensor with a constant value (triggers CoW if shared)
    pub fn fill(&mut self, value: f64) {
        Arc::make_mut(&mut self.data).fill(value);
    }

    /// Get pointer to the underlying data for FFI
    ///
    /// Returns a pointer that can be passed to JIT-compiled functions.
    /// The pointer remains valid as long as the DenseTensor is not moved or dropped.
    pub fn as_ptr(&self) -> *const f64 {
        self.data.as_ptr()
    }

    /// Get mutable pointer to the underlying data for FFI (triggers CoW if shared)
    ///
    /// Returns a mutable pointer for writing results from JIT-compiled functions.
    /// The pointer remains valid as long as the DenseTensor is not moved or dropped.
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        Arc::make_mut(&mut self.data).as_mut_ptr()
    }

    /// Get shape as i64 slice for MLIR compatibility
    ///
    /// MLIR uses i64 for tensor dimensions, so we need conversion from usize.
    pub fn shape_i64(&self) -> Vec<i64> {
        self.shape.iter().map(|&s| s as i64).collect()
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

        indices
            .iter()
            .zip(&self.shape)
            .for_each(|(&idx, &dim)| {
                assert!(
                    idx < dim,
                    "Index {} out of bounds for dimension {}",
                    idx,
                    dim
                )
            });

        indices
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }
}

impl fmt::Debug for DenseTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DenseTensor(shape={:?}, elements={})",
            self.shape,
            self.len()
        )
    }
}

impl fmt::Display for DenseTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenseTensor{:?}", self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = DenseTensor::zeros(vec![3, 4]);
        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.len(), 12);
    }

    #[test]
    fn test_tensor_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DenseTensor::from_data(data.clone(), vec![2, 2]);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &data[..]);
    }

    #[test]
    fn test_tensor_indexing() {
        let mut tensor = DenseTensor::zeros(vec![3, 4]);
        tensor.set(&[1, 2], 42.0);
        assert_eq!(tensor.get(&[1, 2]), 42.0);
    }

    #[test]
    fn test_tensor_fill() {
        let mut tensor = DenseTensor::zeros(vec![2, 3]);
        tensor.fill(3.14);
        for &val in tensor.data() {
            assert_eq!(val, 3.14);
        }
    }

    #[test]
    fn test_ones() {
        let tensor = DenseTensor::ones(vec![2, 3]);
        for &val in tensor.data() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_copy_on_write() {
        let tensor1 = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut tensor2 = tensor1.clone(); // Share data

        // Modification triggers CoW
        tensor2.set(&[0, 0], 999.0);

        // tensor1 should be unchanged
        assert_eq!(tensor1.get(&[0, 0]), 1.0);
        assert_eq!(tensor2.get(&[0, 0]), 999.0);
    }
}
