//! Tensor storage library
//!
//! Provides backend-agnostic data structures for tensor storage.
//!
//! - [`Dense<T>`]: Dense storage with Arc-based Copy-on-Write
//! - [`TensorRepr`]: Common trait for tensor storage representations
//!
//! For the main `Tensor` type (storage + backend), see the `arnet` crate.

pub mod dense;
pub mod repr;
pub mod sector;

// Re-export from ariadnetor-core
pub use arnet_core::{
    Complex, ComputeBackend, ContractionError, ContractionPlan, EinsumExpr, FloatCompute, Scalar,
    compute_permutation, contraction_error, einsum, scalar,
};

pub use dense::{Dense, DenseIter, MemoryOrder, column_major_strides, row_major_strides};
pub use repr::TensorRepr;

/// Extension trait for backend-aware tensor construction.
///
/// Provides tensor constructors on any `ComputeBackend`, hiding `MemoryOrder`
/// from callers and allowing future backend properties to influence
/// tensor construction without changing call sites.
pub trait ComputeBackendTensorExt: ComputeBackend {
    /// Construct a `Dense` from data in this backend's preferred memory order.
    fn make_tensor<T: Clone>(&self, data: Vec<T>, shape: Vec<usize>) -> Dense<T> {
        Dense::from_data_with_order(data, shape, self.preferred_order())
    }

    /// Create a zero-filled tensor in this backend's preferred memory order.
    fn zeros<T: Clone + num_traits::Zero>(&self, shape: Vec<usize>) -> Dense<T> {
        let total: usize = shape.iter().product();
        Dense::from_data_with_order(vec![T::zero(); total], shape, self.preferred_order())
    }

    /// Create a ones-filled tensor in this backend's preferred memory order.
    fn ones<T: Clone + num_traits::Zero + num_traits::One>(&self, shape: Vec<usize>) -> Dense<T> {
        let total: usize = shape.iter().product();
        Dense::from_data_with_order(vec![T::one(); total], shape, self.preferred_order())
    }

    /// Create a constant-filled tensor in this backend's preferred memory order.
    fn constant<T: Clone>(&self, shape: Vec<usize>, value: T) -> Dense<T> {
        let total: usize = shape.iter().product();
        Dense::from_data_with_order(vec![value; total], shape, self.preferred_order())
    }

    /// Create an identity matrix in this backend's preferred memory order.
    ///
    /// The identity matrix is symmetric, so its flat data layout is the same
    /// regardless of memory order.
    fn eye<T: Clone + num_traits::Zero + num_traits::One>(&self, n: usize) -> Dense<T> {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * (n + 1)] = T::one();
        }
        Dense::from_data_with_order(data, vec![n, n], self.preferred_order())
    }
}

impl<B: ComputeBackend> ComputeBackendTensorExt for B {}
