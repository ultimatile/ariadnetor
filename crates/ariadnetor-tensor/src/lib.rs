//! Tensor storage library
//!
//! Provides backend-agnostic data structures for tensor storage.
//!
//! - [`DenseTensor<T>`]: Dense storage with Arc-based Copy-on-Write
//! - [`TensorStorage<T>`]: Storage format enum (Dense; future: Sparse, BlockSparse)
//!
//! For the main `Tensor` type (storage + backend), see the `arnet` crate.

pub mod arithmetic;
pub mod dense;
pub mod sector;
pub mod tensor_storage;

// Re-export from ariadnetor-core
pub use arnet_core::{
    Complex, ComputeBackend, ContractionError, ContractionPlan, EinsumExpr, FloatCompute, Scalar,
    compute_permutation, contraction_error, einsum, scalar,
};

pub use dense::{DenseTensor, MemoryOrder, column_major_strides, row_major_strides};
pub use tensor_storage::TensorStorage;

/// Extension trait for backend-aware tensor construction.
///
/// Provides `make_tensor` on any `ComputeBackend`, hiding `MemoryOrder`
/// from callers and allowing future backend properties to influence
/// tensor construction without changing call sites.
pub trait ComputeBackendTensorExt: ComputeBackend {
    /// Construct a `DenseTensor` in this backend's preferred memory order.
    fn make_tensor<T: Clone>(&self, data: Vec<T>, shape: Vec<usize>) -> DenseTensor<T> {
        DenseTensor::from_data_with_order(data, shape, self.preferred_order())
    }
}

impl<B: ComputeBackend> ComputeBackendTensorExt for B {}

// Convenient type aliases for common numeric types
pub type DenseTensor64 = DenseTensor<f64>;
pub type DenseTensor32 = DenseTensor<f32>;
pub type DenseTensorC64 = DenseTensor<Complex<f64>>;
pub type DenseTensorC32 = DenseTensor<Complex<f32>>;

pub type TensorStorage64 = TensorStorage<f64>;
pub type TensorStorage32 = TensorStorage<f32>;
pub type TensorStorageC64 = TensorStorage<Complex<f64>>;
pub type TensorStorageC32 = TensorStorage<Complex<f32>>;
