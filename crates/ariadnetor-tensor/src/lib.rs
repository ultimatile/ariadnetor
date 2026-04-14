//! Tensor storage library
//!
//! Provides backend-agnostic data structures for tensor storage.
//!
//! - [`Dense<T>`]: Dense storage with Arc-based Copy-on-Write
//! - [`TensorRepr`]: Common trait for tensor storage representations
//!
//! For the main `Tensor` type (storage + backend), see the `arnet` crate.

pub mod block_sparse;
pub mod dense;
pub mod repr;
pub mod sector;

// Re-export from ariadnetor-core
pub use arnet_core::{
    Complex, ComputeBackend, ContractionError, ContractionPlan, EinsumExpr, FloatCompute,
    MemoryOrder, Scalar, compute_permutation, contraction_error, einsum, scalar,
};

pub use dense::{Dense, column_major_strides, row_major_strides};
pub use repr::TensorRepr;

/// Extension trait for backend-aware tensor construction.
///
/// Provides tensor constructors on any `ComputeBackend`, hiding `MemoryOrder`
/// from callers and allowing future backend properties to influence
/// tensor construction without changing call sites.
pub trait ComputeBackendTensorExt: ComputeBackend {
    /// Construct a `Dense` from data in this backend's preferred memory order.
    ///
    /// The caller must arrange `data` in this backend's `preferred_order()`.
    fn make_tensor<T: Clone>(&self, data: Vec<T>, shape: Vec<usize>) -> Dense<T> {
        Dense::new(data, shape)
    }

    /// Create a zero-filled tensor.
    fn zeros<T: Clone + num_traits::Zero>(&self, shape: Vec<usize>) -> Dense<T> {
        Dense::zeros(shape)
    }

    /// Create a ones-filled tensor.
    fn ones<T: Clone + num_traits::Zero + num_traits::One>(&self, shape: Vec<usize>) -> Dense<T> {
        Dense::ones(shape)
    }

    /// Create a constant-filled tensor.
    fn constant<T: Clone>(&self, shape: Vec<usize>, value: T) -> Dense<T> {
        Dense::constant(shape, value)
    }

    /// Create an identity matrix.
    ///
    /// The identity matrix is symmetric, so its flat data layout is the same
    /// regardless of memory order.
    fn eye<T: Clone + num_traits::Zero + num_traits::One>(&self, n: usize) -> Dense<T> {
        Dense::eye(n)
    }
}

impl<B: ComputeBackend> ComputeBackendTensorExt for B {}
