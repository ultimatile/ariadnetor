//! Tensor storage library
//!
//! Provides backend-agnostic data structures for tensor storage.
//!
//! - [`Dense<T>`]: Dense storage with Arc-based Copy-on-Write
//! - [`TensorRepr`]: Common trait for tensor storage representations
//!
//! For the main `Tensor` type (storage + backend), see the `arnet` crate.

mod block_sparse;
mod dense;
mod layout;
mod reorder;
mod repr;
mod sector;
mod storage;
mod tensor_data;

// Re-export from ariadnetor-core
pub use arnet_core::{
    Complex, ComputeBackend, ContractionError, ContractionPlan, EinsumExpr, MemoryOrder, Scalar,
    compute_permutation,
};

pub use block_sparse::{
    BlockCoord, BlockMeta, BlockSparse, BlockSparseLayout, BlockSparseStorage,
    BlockSparseTensorData, Direction, QNIndex,
};
pub use dense::{Dense, DenseLayout, DenseStorage, DenseTensorData};
pub use layout::{StorageFor, TensorLayout};
pub use reorder::{DenseView, flat_index, normalize_to, reorder};
pub use repr::TensorRepr;
pub use sector::{Sector, U1Sector, Z2Sector};
pub use storage::Storage;
pub use tensor_data::TensorData;

/// Extension trait for backend-aware tensor construction.
///
/// Provides tensor constructors on any `ComputeBackend` that produce
/// `Dense<T>` whose `order()` matches the backend's `preferred_order()`,
/// so downstream linalg operations on `Tensor<Dense, B>` find the
/// storage already in the layout they expect.
pub trait ComputeBackendTensorExt: ComputeBackend {
    /// Construct a `Dense` from data in this backend's preferred memory order.
    ///
    /// The caller must arrange `data` in this backend's `preferred_order()`.
    /// The resulting Dense has `order() == self.preferred_order()`.
    fn make_tensor<T: Clone>(&self, data: Vec<T>, shape: Vec<usize>) -> Dense<T> {
        Dense::new(data, shape, self.preferred_order())
    }

    /// Create a zero-filled tensor whose `order()` matches this backend.
    fn zeros<T: Clone + num_traits::Zero>(&self, shape: Vec<usize>) -> Dense<T> {
        let total: usize = shape.iter().product();
        Dense::new(vec![T::zero(); total], shape, self.preferred_order())
    }

    /// Create a ones-filled tensor whose `order()` matches this backend.
    fn ones<T: Clone + num_traits::Zero + num_traits::One>(&self, shape: Vec<usize>) -> Dense<T> {
        let total: usize = shape.iter().product();
        Dense::new(vec![T::one(); total], shape, self.preferred_order())
    }

    /// Create a constant-filled tensor whose `order()` matches this backend.
    fn constant<T: Clone>(&self, shape: Vec<usize>, value: T) -> Dense<T> {
        let total: usize = shape.iter().product();
        Dense::new(vec![value; total], shape, self.preferred_order())
    }

    /// Create an identity matrix whose `order()` matches this backend.
    ///
    /// The identity matrix is symmetric, so its flat data layout is the same
    /// regardless of memory order; only the `order()` field differs from
    /// `Dense::eye(n)`.
    fn eye<T: Clone + num_traits::Zero + num_traits::One>(&self, n: usize) -> Dense<T> {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Dense::new(data, vec![n, n], self.preferred_order())
    }

    /// Construct a `DenseTensorData` from data in this backend's
    /// preferred memory order. Canonical counterpart of
    /// [`make_tensor`](Self::make_tensor).
    fn make_tensor_data<T: Clone>(&self, data: Vec<T>, shape: Vec<usize>) -> DenseTensorData<T> {
        DenseTensorData::from_raw_parts(data, shape, self.preferred_order())
    }

    /// Zero-filled `DenseTensorData` whose `order()` matches this
    /// backend.
    fn zeros_data<T: Clone + num_traits::Zero>(&self, shape: Vec<usize>) -> DenseTensorData<T> {
        let total: usize = shape.iter().product();
        DenseTensorData::from_raw_parts(vec![T::zero(); total], shape, self.preferred_order())
    }

    /// Ones-filled `DenseTensorData` whose `order()` matches this
    /// backend.
    fn ones_data<T: Clone + num_traits::Zero + num_traits::One>(
        &self,
        shape: Vec<usize>,
    ) -> DenseTensorData<T> {
        let total: usize = shape.iter().product();
        DenseTensorData::from_raw_parts(vec![T::one(); total], shape, self.preferred_order())
    }

    /// Constant-filled `DenseTensorData` whose `order()` matches this
    /// backend.
    fn constant_data<T: Clone>(&self, shape: Vec<usize>, value: T) -> DenseTensorData<T> {
        let total: usize = shape.iter().product();
        DenseTensorData::from_raw_parts(vec![value; total], shape, self.preferred_order())
    }

    /// `n×n` identity `DenseTensorData` whose `order()` matches this
    /// backend.
    fn eye_data<T: Clone + num_traits::Zero + num_traits::One>(
        &self,
        n: usize,
    ) -> DenseTensorData<T> {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        DenseTensorData::from_raw_parts(data, vec![n, n], self.preferred_order())
    }
}

impl<B: ComputeBackend> ComputeBackendTensorExt for B {}
