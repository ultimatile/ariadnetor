//! Convenience constructors and accessors for `DenseTensorData<T>`.

use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};
use arnet_core::backend::MemoryOrder;

use crate::{DenseLayout, DenseStorage, TensorData};

/// Backend-less Dense tensor bundle = `TensorData<DenseStorage<T>, DenseLayout>`.
pub type DenseTensorData<T = f64> = TensorData<DenseStorage<T>, DenseLayout>;

impl<T> DenseTensorData<T> {
    /// Construct from flat data, shape, and the memory order the
    /// data is laid out in.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal `shape.iter().product()`.
    pub fn from_raw_parts(data: Vec<T>, shape: Vec<usize>, order: MemoryOrder) -> Self
    where
        T: Clone,
    {
        let storage = DenseStorage::new(data);
        let layout = DenseLayout::new(shape, order);
        Self::new(storage, layout)
    }

    /// Reference to the flat data buffer.
    pub fn data(&self) -> &[T] {
        self.storage().data()
    }

    /// Logical shape.
    pub fn shape(&self) -> &[usize] {
        self.layout().shape()
    }

    /// Memory order the flat data is laid out in.
    pub fn order(&self) -> MemoryOrder {
        self.layout().order()
    }

    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.layout().rank()
    }

    /// Total number of logical elements (`shape().iter().product()`).
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Whether the tensor has zero logical elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Zero-filled tensor in `ColumnMajor` order. Uniform-value data
    /// is layout-invariant; backend-aware callers should construct
    /// directly via [`from_raw_parts`](Self::from_raw_parts) with
    /// `backend.preferred_order()` when matching is required.
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Clone + num_traits::Zero,
    {
        let total: usize = shape.iter().product();
        Self::from_raw_parts(vec![T::zero(); total], shape, MemoryOrder::ColumnMajor)
    }

    /// Ones-filled tensor in `ColumnMajor` order. See
    /// [`zeros`](Self::zeros) for the order convention.
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: Clone + num_traits::One + num_traits::Zero,
    {
        let total: usize = shape.iter().product();
        Self::from_raw_parts(vec![T::one(); total], shape, MemoryOrder::ColumnMajor)
    }

    /// Constant-filled tensor in `ColumnMajor` order.
    pub fn constant(shape: Vec<usize>, value: T) -> Self
    where
        T: Clone,
    {
        let total: usize = shape.iter().product();
        Self::from_raw_parts(vec![value; total], shape, MemoryOrder::ColumnMajor)
    }

    /// `n×n` identity matrix in `ColumnMajor` order. The identity
    /// matrix is symmetric, so the flat layout is the same under
    /// either memory order; only the layout's tag differs from
    /// `RowMajor`.
    pub fn eye(n: usize) -> Self
    where
        T: Clone + num_traits::Zero + num_traits::One,
    {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Self::from_raw_parts(data, vec![n, n], MemoryOrder::ColumnMajor)
    }

    /// Random tensor with elements drawn from the standard
    /// distribution. Random-valued data is layout-invariant; the
    /// resulting tensor's order is `ColumnMajor`.
    pub fn random<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Self
    where
        T: Clone,
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let total: usize = shape.iter().product();
        let data: Vec<T> = (0..total).map(|_| rng.random()).collect();
        Self::from_raw_parts(data, shape, MemoryOrder::ColumnMajor)
    }

    /// Reshape the tensor to a new shape (zero-copy on storage).
    ///
    /// The flat data is not rearranged — only the layout changes.
    /// The output preserves `self.order()`. Reshape semantics depend
    /// on the order: adjacent-axis fusion is zero-copy under both
    /// row-major and column-major for contiguous tensors, but
    /// non-adjacent fusion produces a different logical mapping
    /// under each order.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different total element count
    /// than the current shape.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self
    where
        T: Clone,
    {
        let new_total: usize = new_shape.iter().product();
        assert_eq!(
            self.len(),
            new_total,
            "reshape: total elements must match ({} vs {new_total})",
            self.len()
        );
        let storage = self.storage().clone();
        let layout = DenseLayout::new(new_shape, self.order());
        Self::new(storage, layout)
    }
}

impl<T> DenseTensorData<T>
where
    T: arnet_core::Scalar,
{
    /// Element-wise complex conjugate. Layout is preserved.
    pub fn conj(&self) -> Self {
        let new_data: AVec<T, ConstAlign<64>> =
            AVec::from_iter(64, self.data().iter().copied().map(|x| x.conj()));
        let storage = DenseStorage::from_arc(Arc::new(new_data));
        Self::new(storage, self.layout().clone())
    }
}
