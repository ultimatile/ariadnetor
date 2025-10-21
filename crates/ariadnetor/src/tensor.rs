//! Tensor representation and operations
//!
//! Provides the core Tensor type for the library.

use std::fmt;

/// A multi-dimensional tensor
#[derive(Clone)]
pub struct Tensor {
    /// Tensor data in row-major order
    data: Vec<f64>,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Strides for indexing
    strides: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with zeros
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Example
    ///
    /// ```
    /// use tn_mlir::Tensor;
    ///
    /// let tensor = Tensor::new(vec![10, 20]);
    /// assert_eq!(tensor.shape(), &[10, 20]);
    /// ```
    pub fn new(shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![0.0; total_elements];
        let strides = Self::compute_strides(&shape);

        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create a tensor from existing data
    ///
    /// # Arguments
    ///
    /// * `data` - Tensor data in row-major order
    /// * `shape` - Dimensions of the tensor
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
            data,
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

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
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

    /// Set element at given indices
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn set(&mut self, indices: &[usize], value: f64) {
        let flat_index = self.flat_index(indices);
        self.data[flat_index] = value;
    }

    /// Compute strides for row-major layout
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
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

    /// Fill tensor with a constant value
    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self {
        let mut tensor = Self::new(shape);
        tensor.fill(1.0);
        tensor
    }

    /// Create a tensor filled with a specific value
    pub fn constant(shape: Vec<usize>, value: f64) -> Self {
        let mut tensor = Self::new(shape);
        tensor.fill(value);
        tensor
    }

    /// Get pointer to the underlying data for FFI
    ///
    /// Returns a pointer that can be passed to JIT-compiled functions.
    /// The pointer remains valid as long as the Tensor is not moved or dropped.
    pub fn as_ptr(&self) -> *const f64 {
        self.data.as_ptr()
    }

    /// Get mutable pointer to the underlying data for FFI
    ///
    /// Returns a mutable pointer for writing results from JIT-compiled functions.
    /// The pointer remains valid as long as the Tensor is not moved or dropped.
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.data.as_mut_ptr()
    }

    /// Get shape as i64 slice for MLIR compatibility
    ///
    /// MLIR uses i64 for tensor dimensions, so we need conversion from usize.
    pub fn shape_i64(&self) -> Vec<i64> {
        self.shape.iter().map(|&s| s as i64).collect()
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, elements={})", self.shape, self.len())
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{:?}", self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(vec![3, 4]);
        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.len(), 12);
    }

    #[test]
    fn test_tensor_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data.clone(), vec![2, 2]);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &data[..]);
    }

    #[test]
    fn test_tensor_indexing() {
        let mut tensor = Tensor::new(vec![3, 4]);
        tensor.set(&[1, 2], 42.0);
        assert_eq!(tensor.get(&[1, 2]), 42.0);
    }

    #[test]
    fn test_tensor_fill() {
        let mut tensor = Tensor::new(vec![2, 3]);
        tensor.fill(3.14);
        for &val in tensor.data() {
            assert_eq!(val, 3.14);
        }
    }

    #[test]
    fn test_ones() {
        let tensor = Tensor::ones(vec![2, 3]);
        for &val in tensor.data() {
            assert_eq!(val, 1.0);
        }
    }
}
