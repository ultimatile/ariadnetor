//! Fat tensor with metadata (storage + indices)

use crate::raw_tensor::RawTensor;
use crate::index::IndexSet;

/// Fat tensor: RawTensor + Index metadata
///
/// This is the main tensor type for tensor network computations.
#[derive(Debug, Clone)]
pub struct FatTensor {
    pub tensor: RawTensor,
    pub indices: IndexSet,
}

impl FatTensor {
    /// Create a new FatTensor
    pub fn new(tensor: RawTensor, indices: IndexSet) -> Self {
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
