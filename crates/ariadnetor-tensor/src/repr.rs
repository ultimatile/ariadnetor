//! Tensor representation trait
//!
//! Defines the common interface that all tensor storage types must implement.

use crate::dense::Dense;

/// Common interface for tensor storage representations.
///
/// Implemented by [`Dense<T>`] and future storage types (e.g., `BlockSparse<T, S>`).
pub trait TensorRepr: Clone {
    /// Element type stored in the tensor
    type Elem;

    /// Get the shape of the tensor
    fn shape(&self) -> &[usize];

    /// Get the rank (number of dimensions) of the tensor
    fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of logical elements
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if the tensor has zero elements
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> TensorRepr for Dense<T> {
    type Elem = T;

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}
