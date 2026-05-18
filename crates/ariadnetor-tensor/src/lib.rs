//! Tensor storage library
//!
//! Provides backend-agnostic data structures for tensor storage under
//! the storage / layout split:
//!
//! - [`DenseStorage<T>`] / [`DenseLayout`] / [`DenseTensorData<T>`] —
//!   dense case
//! - [`BlockSparseStorage<T>`] / [`BlockSparseLayout<S>`] /
//!   [`BlockSparseTensorData<T, S>`] — block-sparse case
//!
//! For the main `Tensor` type (storage + backend), see the `arnet` crate.

mod block_sparse;
mod dense;
mod layout;
mod reorder;
mod sector;
mod storage;
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
pub use layout::{StorageFor, TensorLayout};
pub use reorder::{flat_index, normalize_to, reorder};
pub use sector::{Sector, U1Sector, Z2Sector};
pub use storage::Storage;
pub use tensor_data::TensorData;

/// Extension trait for backend-aware tensor construction.
///
/// Provides tensor constructors on any `ComputeBackend` that produce
/// `DenseTensorData<T>` whose `order()` matches the backend's
/// `preferred_order()`, so downstream linalg operations on
/// `Tensor<DenseStorage, _, B>` find the storage already in the layout
/// they expect.
pub trait ComputeBackendTensorExt: ComputeBackend {
    /// Construct a `DenseTensorData` from data in this backend's
    /// preferred memory order. The caller must arrange `data` in this
    /// backend's `preferred_order()`. The resulting tensor has
    /// `order() == self.preferred_order()`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use arnet_core::Scalar;
    use arnet_core::backend::{
        BackendError, ComputeBackend, DeviceType, GemmDescriptor, MemoryOrder, TransposeDescriptor,
    };

    /// RowMajor-preferring stub backend used by the
    /// `ComputeBackendTensorExt` tests below.
    struct RmBackend;

    impl ComputeBackend for RmBackend {
        fn name(&self) -> &'static str {
            "rm-test-stub"
        }
        fn device_type(&self) -> DeviceType {
            DeviceType::Cpu
        }
        fn preferred_order(&self) -> MemoryOrder {
            MemoryOrder::RowMajor
        }
        fn gemm<T: Scalar>(&self, _desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
            Err(BackendError::NotSupported("stub".into()))
        }
        fn transpose<T: Scalar>(
            &self,
            _desc: TransposeDescriptor<'_, T>,
        ) -> Result<(), BackendError> {
            Err(BackendError::NotSupported("stub".into()))
        }
    }

    #[test]
    fn make_tensor_data_tags_backend_order() {
        let b = RmBackend;
        let t: DenseTensorData<f64> = b.make_tensor_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.order(), MemoryOrder::RowMajor);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn zeros_data_tags_backend_order() {
        let b = RmBackend;
        let t: DenseTensorData<f64> = b.zeros_data(vec![3, 2]);
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.order(), MemoryOrder::RowMajor);
        assert_eq!(t.data(), &[0.0; 6]);
    }

    #[test]
    fn ones_data_tags_backend_order() {
        let b = RmBackend;
        let t: DenseTensorData<f64> = b.ones_data(vec![2, 4]);
        assert_eq!(t.shape(), &[2, 4]);
        assert_eq!(t.order(), MemoryOrder::RowMajor);
        assert_eq!(t.data(), &[1.0; 8]);
    }

    #[test]
    fn constant_data_tags_backend_order() {
        let b = RmBackend;
        let t: DenseTensorData<f64> = b.constant_data(vec![2, 3], 9.0);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.order(), MemoryOrder::RowMajor);
        assert_eq!(t.data(), &[9.0; 6]);
    }

    #[test]
    fn eye_data_is_identity_in_backend_order() {
        let b = RmBackend;
        let t: DenseTensorData<f64> = b.eye_data(3);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(t.order(), MemoryOrder::RowMajor);
        // Identity is symmetric: storage matches under either order.
        assert_eq!(t.data(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(t.get(&[i, j]), expected);
            }
        }
    }
}
