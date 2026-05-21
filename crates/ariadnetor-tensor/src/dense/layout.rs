//! `DenseLayout`: interpretation half of the dense tensor split.
//!
//! Carries shape and memory order. Data lives on
//! [`DenseStorage<T>`](crate::DenseStorage); the wrapper
//! [`DenseTensorData<T>`](crate::DenseTensorData) joins the two with a
//! length-consistency check.

use arnet_core::backend::MemoryOrder;

use crate::TensorLayout;

/// Interpretation half of the dense tensor split.
///
/// Holds the logical shape and the memory order the paired
/// [`DenseStorage`](crate::DenseStorage) is laid out in. Operations
/// consuming a [`DenseTensorData`](crate::DenseTensorData) consult
/// `order()` to decide whether to repack at their boundary.
#[derive(Clone, Debug)]
pub struct DenseLayout {
    shape: Vec<usize>,
    order: MemoryOrder,
}

impl DenseLayout {
    /// Construct a `DenseLayout` from shape and memory order.
    pub fn new(shape: Vec<usize>, order: MemoryOrder) -> Self {
        Self { shape, order }
    }

    /// Logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Memory order the paired storage is laid out in.
    pub fn order(&self) -> MemoryOrder {
        self.order
    }
}

impl TensorLayout for DenseLayout {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn storage_extent(&self) -> usize {
        self.shape.iter().product()
    }
}
