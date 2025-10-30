//! Fat tensor with metadata (storage + indices)

use crate::index::IndexSet;
use crate::raw_tensor::RawTensor;

/// Fat tensor: RawTensor + Index metadata
///
/// This is the main tensor type for tensor network computations.
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64). See [`DenseTensor`](crate::DenseTensor) for details.
#[derive(Debug, Clone)]
pub struct FatTensor<T = f64> {
    pub tensor: RawTensor<T>,
    pub indices: IndexSet,
}

impl<T> FatTensor<T> {
    /// Create a new FatTensor
    pub fn new(tensor: RawTensor<T>, indices: IndexSet) -> Self {
        // TODO: Validate that tensor rank matches number of indices
        Self { tensor, indices }
    }

    /// Get the shape of the underlying tensor
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.tensor.rank()
    }
}
