//! Layout abstraction: tensor interpretation metadata.
//!
//! `TensorLayout` is the interpretation half of the tensor split. It
//! carries shape and storage-extent information sufficient to validate
//! a paired [`Storage`](crate::Storage) buffer. Element data lives on
//! the storage; metadata such as memory order, axis labels, and
//! symmetry sector lives here.

use crate::Storage;

/// Interpretation metadata for a paired [`Storage`](crate::Storage) buffer.
///
/// A `TensorLayout` describes how to read the flat buffer: the logical
/// shape, and the number of element slots the buffer is expected to
/// hold (see [`storage_extent`](Self::storage_extent)). Concrete
/// implementors carry additional flavor-specific metadata (memory
/// order on `DenseLayout`; block structure, indices, and flux on
/// `BlockSparseLayout`).
pub trait TensorLayout {
    /// Logical shape of the tensor.
    fn shape(&self) -> &[usize];

    /// Expected flat storage length for any compatible
    /// [`Storage`](crate::Storage).
    ///
    /// For [`DenseLayout`](crate::DenseLayout) this equals
    /// `shape().iter().product()`; for
    /// [`BlockSparseLayout`](crate::BlockSparseLayout) it equals the
    /// sum of allowed block sizes (strictly less than
    /// `product(shape)` when symmetry forbids blocks). This is the
    /// quantity checked by
    /// [`TensorData::new`](crate::TensorData::new) at the
    /// storage-layout boundary.
    ///
    /// Logical / dense extent (always `product(shape)`) is computed
    /// at the call site from [`shape`](Self::shape); it is
    /// intentionally not a method here so the two quantities are not
    /// silently conflated for `BlockSparseLayout`.
    fn storage_extent(&self) -> usize;
}

/// Compatibility marker between a [`Storage`] and a [`TensorLayout`].
///
/// `St: StorageFor<L>` declares that `St` is a valid storage for
/// layout `L`. The marker has no methods; it serves as the type-level
/// gate enforcing that
/// [`TensorData<St, L>`](crate::TensorData) is only constructed from
/// flavor-matched pairs (`DenseStorage` ⇔ `DenseLayout`,
/// `BlockSparseStorage` ⇔ `BlockSparseLayout`).
pub trait StorageFor<L: TensorLayout>: Storage {}
