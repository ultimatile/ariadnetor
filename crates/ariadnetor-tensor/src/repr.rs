//! Tensor representation trait
//!
//! Defines the common interface that all tensor storage types must implement.

use arnet_core::Scalar;

use crate::block_sparse::BlockSparse;
use crate::dense::Dense;
use crate::sector::Sector;

/// Common interface for tensor storage representations.
///
/// Implemented by [`Dense<T>`] and [`BlockSparse<T, S>`].
pub trait TensorRepr: Clone {
    /// Scalar element type stored in the tensor.
    type Elem: Scalar;

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

impl<T: Scalar> TensorRepr for Dense<T> {
    type Elem = T;

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}

impl<T: Scalar, S: Sector> TensorRepr for BlockSparse<T, S> {
    type Elem = T;

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}
