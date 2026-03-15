//! Tensor storage format enum
//!
//! TensorStorage represents the storage layer without metadata.

use crate::dense::DenseTensor;
use num_traits::{One, Zero};
use std::fmt;

/// Tensor storage format (low-level, no metadata)
///
/// This enum represents different tensor storage strategies:
/// - **Dense**: Contiguous array (all elements stored)
/// - **Sparse**: Coordinate (COO) format (only non-zero elements)
/// - **BlockSparse**: Block-wise storage with symmetry sectors
///
#[derive(Clone)]
pub enum TensorStorage<T = f64> {
    /// Dense tensor with contiguous storage
    Dense(DenseTensor<T>),
    // TODO: Phase 1+ - Sparse tensor support
    // Sparse(SparseTensor<T>),
    //
    // TODO: Phase 2+ - Block-sparse tensor support
    // BlockSparse(BlockSparseVariant<T>),
}

impl<T> TensorStorage<T> {
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

    /// Get shape as i64 slice for MLIR compatibility
    pub fn shape_i64(&self) -> Vec<i64> {
        match self {
            Self::Dense(t) => t.shape_i64(),
        }
    }
}

impl<T> TensorStorage<T>
where
    T: Clone,
{
    /// Create a dense tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero,
    {
        Self::Dense(DenseTensor::zeros(shape))
    }

    /// Create a dense tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Zero,
    {
        Self::Dense(DenseTensor::ones(shape))
    }

    /// Create a dense tensor filled with a constant value
    pub fn constant(shape: Vec<usize>, value: T) -> Self {
        Self::Dense(DenseTensor::constant(shape, value))
    }

    /// Create a dense tensor from existing data
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::Dense(DenseTensor::from_data(data, shape))
    }

    /// Get a reference to the underlying data (only for Dense)
    ///
    /// Returns None for non-dense storage formats
    pub fn data(&self) -> Option<&[T]> {
        match self {
            Self::Dense(t) => Some(t.data()),
        }
    }

    /// Get a mutable reference to the underlying data (only for Dense)
    ///
    /// Returns None for non-dense storage formats
    pub fn data_mut(&mut self) -> Option<&mut [T]> {
        match self {
            Self::Dense(t) => Some(t.data_mut()),
        }
    }

    /// Get element at given indices
    pub fn get(&self, indices: &[usize]) -> T {
        match self {
            Self::Dense(t) => t.get(indices),
        }
    }

    /// Set element at given indices
    pub fn set(&mut self, indices: &[usize], value: T) {
        match self {
            Self::Dense(t) => t.set(indices, value),
        }
    }

    /// Fill tensor with a constant value
    pub fn fill(&mut self, value: T) {
        match self {
            Self::Dense(t) => t.fill(value),
        }
    }

    /// Get pointer to the underlying data for FFI (only for Dense)
    pub fn as_ptr(&self) -> Option<*const T> {
        match self {
            Self::Dense(t) => Some(t.as_ptr()),
        }
    }

    /// Get mutable pointer to the underlying data for FFI (only for Dense)
    pub fn as_mut_ptr(&mut self) -> Option<*mut T> {
        match self {
            Self::Dense(t) => Some(t.as_mut_ptr()),
        }
    }
}

impl<T> fmt::Debug for TensorStorage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(t) => write!(f, "TensorStorage::Dense({:?})", t),
        }
    }
}

impl<T> fmt::Display for TensorStorage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(t) => write!(f, "TensorStorage::Dense({})", t),
        }
    }
}
