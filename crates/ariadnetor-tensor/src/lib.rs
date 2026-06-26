//! Tensor storage library
//!
//! Provides backend-agnostic data structures for tensor storage and
//! the user-facing [`Tensor<St, L>`] type that joins them. The tensor
//! carries no backend; operations take it explicitly at the call site.

#![deny(missing_docs)]

mod block_sparse;
mod capability;
mod dense;
mod error;
mod layout;
mod ops;
mod reorder;
mod sector;
mod storage;
mod tensor;
mod tensor_data;

#[cfg(test)]
mod test_fixtures;

// Re-export from ariadnetor-core
pub use arnet_core::{
    Complex, ComputeBackend, ContractionError, ContractionPlan, EinsumExpr, MemoryOrder, Scalar,
    compute_permutation,
};

pub use block_sparse::{
    BlockCoord, BlockMeta, BlockSparseLayout, BlockSparseStorage, BlockSparseTensorData, Direction,
    QNIndex,
};
pub use capability::{Host, OpsFor};
pub use dense::{DenseLayout, DenseStorage, DenseTensorData};
pub use error::TensorError;
pub use layout::{StorageFor, TensorLayout};
pub use ops::{add_all, linear_combine};
pub use reorder::{flat_index, normalize_to_data, reorder_data};
pub use sector::{Sector, U1Sector, Z2Sector};
pub use storage::Storage;
pub use tensor::{BlockSparseTensor, DenseTensor, Tensor};
pub use tensor_data::TensorData;

// Re-export the native backend so users importing arnet_tensor for
// `DenseTensor` / `BlockSparseTensor` constructors get its default
// backend without a separate crate import.
pub use arnet_native::NativeBackend;

/// Extension trait for backend-aware tensor construction.
///
/// Provides tensor constructors on any `ComputeBackend`. The constructed
/// tensors have `order()` matching the backend's `preferred_order()`, so
/// downstream linalg operations driven by that backend find the storage
/// already in the layout they expect. Most constructors return the
/// Mid-layer `DenseTensorData<T>` for kernel-output paths; `dense`
/// returns the wrapped `DenseTensor<T>` for the input-fabrication case.
pub trait ComputeBackendTensorExt: ComputeBackend {
    /// Construct a `DenseTensorData` from data in this backend's
    /// preferred memory order.
    ///
    /// The caller must arrange `data` in this backend's
    /// `preferred_order()`. The resulting tensor data has
    /// `order() == self.preferred_order()`.
    fn make_tensor<T: Clone>(&self, data: Vec<T>, shape: Vec<usize>) -> DenseTensorData<T> {
        DenseTensorData::from_raw_parts(data, shape, self.preferred_order())
    }

    /// Construct a `DenseTensor` from data in this backend's preferred
    /// memory order.
    ///
    /// The caller must arrange `data` in this backend's
    /// `preferred_order()`. The resulting tensor has
    /// `order() == self.preferred_order()`.
    ///
    /// One-call entry for fabricating an input tensor from flat parts:
    /// fuses [`make_tensor`](Self::make_tensor) — which yields the
    /// Mid-layer `DenseTensorData`, kept for kernel-output paths — with
    /// the `DenseTensor` wrap, so a caller that just wants a tensor need
    /// not reach across the Data layer. The backend stays explicit at
    /// the call site, so this is not a host-hardcoded constructor.
    fn dense<T: Clone>(&self, data: Vec<T>, shape: Vec<usize>) -> DenseTensor<T> {
        DenseTensor::from_data(self.make_tensor(data, shape))
    }

    /// Create a zero-filled tensor whose `order()` matches this backend.
    fn zeros<T: Clone + num_traits::Zero>(&self, shape: Vec<usize>) -> DenseTensorData<T> {
        DenseTensorData::zeros_in_order(shape, self.preferred_order())
    }

    /// Create a ones-filled tensor whose `order()` matches this backend.
    fn ones<T: Clone + num_traits::Zero + num_traits::One>(
        &self,
        shape: Vec<usize>,
    ) -> DenseTensorData<T> {
        DenseTensorData::ones_in_order(shape, self.preferred_order())
    }

    /// Create a tensor filled with `value` whose `order()` matches this backend.
    fn filled<T: Clone>(&self, shape: Vec<usize>, value: T) -> DenseTensorData<T> {
        DenseTensorData::filled_in_order(shape, value, self.preferred_order())
    }

    /// Create an identity matrix whose `order()` matches this backend.
    ///
    /// The identity matrix is symmetric, so its flat data layout is the same
    /// regardless of memory order; only the `order()` field differs.
    fn eye<T: Clone + num_traits::Zero + num_traits::One>(&self, n: usize) -> DenseTensorData<T> {
        DenseTensorData::eye_in_order(n, self.preferred_order())
    }
}

impl<B: ComputeBackend> ComputeBackendTensorExt for B {}
