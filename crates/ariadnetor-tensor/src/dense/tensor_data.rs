//! Convenience constructors and accessors for `DenseTensorData<T>`.

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

    /// Cheap O(1) `Dense<T>` view that shares the underlying storage Arc.
    ///
    /// Bridges the joined-form TensorData into the legacy
    /// [`Dense<T>`](crate::Dense) representation that internal linalg
    /// kernels still operate on. Cloning the Arc avoids the bulk copy
    /// the umbrella crate's earlier `bridge_in` used.
    pub fn as_dense(&self) -> crate::Dense<T> {
        crate::Dense::from_storage_arc(
            self.storage().arc_clone(),
            self.shape().to_vec(),
            self.order(),
        )
    }
}
