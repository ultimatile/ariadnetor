//! Storage abstraction: pure-data tensor element container.
//!
//! `Storage` is the data-only half of the tensor split. It exposes only
//! what is needed to validate buffer length against a [`TensorLayout`]
//! and to access the element scalar type. Interpretation metadata
//! (shape, memory order, symmetry sector) lives on the paired
//! [`TensorLayout`](crate::TensorLayout), not here.

/// Pure-data tensor element container.
///
/// Implementors expose a flat element count and the scalar element
/// type. They are intentionally unaware of shape, memory order, and
/// symmetry sector — those properties live on the paired
/// [`TensorLayout`](crate::TensorLayout). The
/// [`StorageFor<L>`](crate::StorageFor) marker trait declares which
/// `Storage` types are valid partners for a given layout.
pub trait Storage {
    /// Scalar element type stored in this container.
    type Element;

    /// Number of element slots actually held by the storage buffer.
    ///
    /// For [`TensorData`](crate::TensorData) construction, this is
    /// compared against [`TensorLayout::storage_extent`](crate::TensorLayout::storage_extent)
    /// to detect storage-layout boundary violations.
    fn flat_len(&self) -> usize;
}
