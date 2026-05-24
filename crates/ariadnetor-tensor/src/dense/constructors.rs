//! Factory methods and data initialization for Dense.

use aligned_vec::{AVec, ConstAlign};
use arnet_core::backend::MemoryOrder;
use num_traits::{One, Zero};
use std::sync::Arc;

use super::{Align64, Dense};

impl<T> Dense<T> {
    /// Construct a `Dense<T>` from an already-aligned storage Arc, the
    /// logical shape, and the memory order the data is laid out in.
    ///
    /// Internal kernel constructor: only callers that already hold a
    /// 64-byte-aligned `Arc<AVec<T, Align64>>` (notably
    /// [`DenseTensorData::as_dense`](crate::DenseTensorData::as_dense))
    /// can satisfy the alignment invariant without a copy. Pub for
    /// cross-crate access from `arnet-linalg`; not user-facing.
    #[doc(hidden)]
    pub fn from_storage_arc(
        data: Arc<AVec<T, Align64>>,
        shape: Vec<usize>,
        order: MemoryOrder,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Dense::from_storage_arc: data length {} doesn't match shape product {:?}",
            data.len(),
            shape,
        );
        Self { data, shape, order }
    }

    /// Move into a [`DenseTensorData<T>`](crate::DenseTensorData) by
    /// stealing the storage Arc.
    ///
    /// Inverse of [`DenseTensorData::as_dense`](crate::DenseTensorData::as_dense):
    /// `dense.into_tensor_data()` produces a `DenseTensorData` that
    /// shares the same aligned buffer, no copy. Pub for cross-crate
    /// kernel-output wrapping; not user-facing.
    #[doc(hidden)]
    pub fn into_tensor_data(self) -> crate::DenseTensorData<T> {
        let storage = crate::DenseStorage::from_arc(self.data);
        let layout = crate::DenseLayout::new(self.shape, self.order);
        crate::TensorData::new(storage, layout)
    }
}

impl<T> Dense<T>
where
    T: Clone,
{
    /// Create a Dense tensor from flat data and shape, declaring the
    /// memory order the data is laid out in.
    ///
    /// The `source_order` argument is the storage layout of `data`
    /// at the call site; the resulting Dense's `order()` matches.
    /// Operations consuming this tensor will reorder at the boundary
    /// if their expected layout differs.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal the product of `shape`.
    pub fn new(data: Vec<T>, shape: Vec<usize>, source_order: MemoryOrder) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            total,
            "Dense::new: data length {} doesn't match shape {:?} (total {})",
            data.len(),
            shape,
            total
        );

        let mut aligned: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, total);
        for elem in data {
            aligned.push(elem);
        }

        Self {
            data: Arc::new(aligned),
            shape,
            order: source_order,
        }
    }

    /// Create a new tensor filled with zeros.
    ///
    /// Uniform-value data is layout-invariant; the resulting Dense's
    /// `order()` is set to ColumnMajor as the project default.
    /// Backend-aware callers should prefer `backend.zeros(shape)` so
    /// the order matches the active backend's preferred layout.
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero,
    {
        let total: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total);
        data.resize(total, T::zero());

        Self {
            data: Arc::new(data),
            shape,
            order: MemoryOrder::ColumnMajor,
        }
    }

    /// Create a tensor filled with ones.
    ///
    /// Uniform-value data is layout-invariant; see [`Dense::zeros`]
    /// for the order convention.
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Zero,
    {
        let total: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total);
        data.resize(total, T::one());

        Self {
            data: Arc::new(data),
            shape,
            order: MemoryOrder::ColumnMajor,
        }
    }

    /// Create a tensor filled with a constant value.
    ///
    /// Uniform-value data is layout-invariant; see [`Dense::zeros`]
    /// for the order convention.
    pub fn constant(shape: Vec<usize>, value: T) -> Self {
        let total: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total);
        data.resize(total, value);

        Self {
            data: Arc::new(data),
            shape,
            order: MemoryOrder::ColumnMajor,
        }
    }

    /// Create an n×n identity matrix.
    ///
    /// The identity matrix is symmetric, so its flat data layout is
    /// the same regardless of memory order. The resulting Dense's
    /// `order()` is set to ColumnMajor as the project default.
    pub fn eye(n: usize) -> Self
    where
        T: Zero + One,
    {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Self::new(data, vec![n, n], MemoryOrder::ColumnMajor)
    }

    /// Create a tensor filled with random values from the standard distribution.
    ///
    /// Random-valued data is layout-invariant; the resulting Dense's
    /// `order()` is set to ColumnMajor as the project default.
    pub fn random<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let total: usize = shape.iter().product();
        let data: Vec<T> = (0..total).map(|_| rng.random()).collect();
        Self::new(data, shape, MemoryOrder::ColumnMajor)
    }
}
