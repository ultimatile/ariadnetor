//! Tensor library with two-layer architecture
//!
//! - **RawTensor**: Storage layer (Dense, Sparse, BlockSparse)
//! - **FatTensor**: Metadata layer (storage + indices)

pub mod dense;
pub mod raw_tensor;
pub mod fat_tensor;
pub mod index;
pub mod sector;

pub use dense::DenseTensor;
pub use raw_tensor::RawTensor;
pub use fat_tensor::FatTensor;
pub use index::{Index, IndexSet};

/// Public API alias for FatTensor
pub type Tensor = FatTensor;
