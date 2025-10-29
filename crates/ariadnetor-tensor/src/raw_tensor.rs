//! Raw tensor storage format enum
//!
//! RawTensor represents the storage layer without metadata (labels, indices, etc.).

use crate::dense::DenseTensor;
use std::fmt;

/// Raw tensor storage format (low-level, no metadata)
///
/// This enum represents different tensor storage strategies:
/// - **Dense**: Contiguous array (all elements stored)
/// - **Sparse**: Coordinate (COO) format (only non-zero elements)
/// - **BlockSparse**: Block-wise storage with symmetry sectors
///
#[derive(Clone)]
pub enum RawTensor {
    /// Dense tensor with contiguous storage
    Dense(DenseTensor),

    // TODO: Phase 1+ - Sparse tensor support
    // Sparse(SparseTensor),
    //
    // TODO: Phase 2+ - Block-sparse tensor support
    // BlockSparse(BlockSparseVariant),
}

impl RawTensor {
    /// Create a dense tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::Dense(DenseTensor::zeros(shape))
    }

    /// Create a dense tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self {
        Self::Dense(DenseTensor::ones(shape))
    }

    /// Create a dense tensor filled with a constant value
    pub fn constant(shape: Vec<usize>, value: f64) -> Self {
        Self::Dense(DenseTensor::constant(shape, value))
    }

    /// Create a dense tensor from existing data
    pub fn from_data(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::Dense(DenseTensor::from_data(data, shape))
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(t) => t.shape(),
        }
    }

    /// Get the rank (number of dimensions) of the tensor
    pub fn rank(&self) -> usize {
        match self {
            Self::Dense(t) => t.rank(),
        }
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(t) => t.len(),
        }
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Dense(t) => t.is_empty(),
        }
    }

    /// Get a reference to the underlying data (only for Dense)
    ///
    /// Returns None for non-dense storage formats
    pub fn data(&self) -> Option<&[f64]> {
        match self {
            Self::Dense(t) => Some(t.data()),
        }
    }

    /// Get a mutable reference to the underlying data (only for Dense)
    ///
    /// Returns None for non-dense storage formats
    pub fn data_mut(&mut self) -> Option<&mut [f64]> {
        match self {
            Self::Dense(t) => Some(t.data_mut()),
        }
    }

    /// Get element at given indices
    pub fn get(&self, indices: &[usize]) -> f64 {
        match self {
            Self::Dense(t) => t.get(indices),
        }
    }

    /// Set element at given indices
    pub fn set(&mut self, indices: &[usize], value: f64) {
        match self {
            Self::Dense(t) => t.set(indices, value),
        }
    }

    /// Fill tensor with a constant value
    pub fn fill(&mut self, value: f64) {
        match self {
            Self::Dense(t) => t.fill(value),
        }
    }

    /// Get pointer to the underlying data for FFI (only for Dense)
    pub fn as_ptr(&self) -> Option<*const f64> {
        match self {
            Self::Dense(t) => Some(t.as_ptr()),
        }
    }

    /// Get mutable pointer to the underlying data for FFI (only for Dense)
    pub fn as_mut_ptr(&mut self) -> Option<*mut f64> {
        match self {
            Self::Dense(t) => Some(t.as_mut_ptr()),
        }
    }

    /// Get shape as i64 slice for MLIR compatibility
    pub fn shape_i64(&self) -> Vec<i64> {
        match self {
            Self::Dense(t) => t.shape_i64(),
        }
    }
}

impl fmt::Debug for RawTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(t) => write!(f, "RawTensor::Dense({:?})", t),
        }
    }
}

impl fmt::Display for RawTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(t) => write!(f, "RawTensor::Dense({})", t),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_tensor_zeros() {
        let tensor = RawTensor::zeros(vec![3, 4]);
        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.len(), 12);
    }

    #[test]
    fn test_raw_tensor_ones() {
        let tensor = RawTensor::ones(vec![2, 3]);
        if let Some(data) = tensor.data() {
            for &val in data {
                assert_eq!(val, 1.0);
            }
        }
    }

    #[test]
    fn test_raw_tensor_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = RawTensor::from_data(data.clone(), vec![2, 2]);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data().unwrap(), &data[..]);
    }

    #[test]
    fn test_raw_tensor_indexing() {
        let mut tensor = RawTensor::zeros(vec![3, 4]);
        tensor.set(&[1, 2], 42.0);
        assert_eq!(tensor.get(&[1, 2]), 42.0);
    }
}
