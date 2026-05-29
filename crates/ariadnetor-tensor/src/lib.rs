//! Tensor storage library
//!
//! Provides backend-agnostic data structures for tensor storage and
//! the user-facing [`Tensor<St, L, B>`] type that pairs them with a
//! compute backend.

mod block_sparse;
mod dense;
mod error;
mod layout;
mod ops;
mod reorder;
mod sector;
mod storage;
mod tensor;
mod tensor_data;

// Re-export from ariadnetor-core
pub use arnet_core::{
    Complex, ComputeBackend, ContractionError, ContractionPlan, EinsumExpr, MemoryOrder, Scalar,
    compute_permutation,
};

pub use block_sparse::{
    BlockCoord, BlockMeta, BlockSparseLayout, BlockSparseStorage, BlockSparseTensorData, Direction,
    QNIndex,
};
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
/// Provides tensor constructors on any `ComputeBackend` that produce
/// `DenseTensorData<T>` whose `order()` matches the backend's
/// `preferred_order()`, so downstream linalg operations on the
/// joined `Tensor<DenseStorage, DenseLayout, B>` find the storage
/// already in the layout they expect.
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
