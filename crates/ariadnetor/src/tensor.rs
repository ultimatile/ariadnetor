//! Tensor representation and operations
//!
//! Re-exports from ariadnetor-tensor crate.
//!
//! # Migration Note
//!
//! This module now re-exports from the ariadnetor-tensor crate.
//! - Old `Tensor` is now `DenseTensor` wrapped in `RawTensor::Dense`
//! - New public API `Tensor` (= `FatTensor`) with Label metadata
//!
//! For backward compatibility, existing code uses `DenseTensor` directly.

pub use arnet_tensor::{DenseTensor, FatTensor, LabelId, RawTensor};

// Temporary backward compatibility: Old code expects bare `Tensor` = DenseTensor
// TODO: Migrate to RawTensor/FatTensor distinction
pub type Tensor = DenseTensor;
