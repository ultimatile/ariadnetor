//! `TensorData<St, L>`: the storage + layout bundle.
//!
//! Joins a [`Storage`] half with a paired [`TensorLayout`] half. The
//! `new` constructor enforces the storage-layout boundary
//! (length-equality check); layout-internal invariants are validated
//! by the layout's own constructor.
//!
//! Flavor-specific aliases [`DenseTensorData<T>`](crate::DenseTensorData)
//! and [`BlockSparseTensorData<T, S>`](crate::BlockSparseTensorData)
//! carry the convenience constructors and joined accessors that need
//! to touch both halves simultaneously (e.g. block-data slicing for
//! block-sparse tensors).

use crate::{Storage, StorageFor, TensorLayout};

/// Joined storage + layout bundle.
///
/// Construction goes through [`new`](Self::new), which asserts
/// `storage.flat_len() == layout.storage_extent()`. The bound
/// `St: StorageFor<L>` enforces flavor compatibility at the type
/// level (only `DenseStorage` ⇔ `DenseLayout`,
/// `BlockSparseStorage` ⇔ `BlockSparseLayout`).
pub struct TensorData<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    storage: St,
    layout: L,
}

impl<St, L> TensorData<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Construct from a `Storage` half and a paired `TensorLayout`
    /// half. Asserts the storage-layout boundary: the storage's flat
    /// length must match the layout's expected storage extent.
    pub fn new(storage: St, layout: L) -> Self {
        assert_eq!(
            storage.flat_len(),
            layout.storage_extent(),
            "TensorData::new: storage.flat_len() = {} but layout.storage_extent() = {}",
            storage.flat_len(),
            layout.storage_extent(),
        );
        Self { storage, layout }
    }

    /// Reference to the storage half.
    pub fn storage(&self) -> &St {
        &self.storage
    }

    /// Mutable reference to the storage half.
    pub fn storage_mut(&mut self) -> &mut St {
        &mut self.storage
    }

    /// Reference to the layout half.
    pub fn layout(&self) -> &L {
        &self.layout
    }

    /// Mutable reference to the layout half.
    pub fn layout_mut(&mut self) -> &mut L {
        &mut self.layout
    }

    /// Consume and return both halves.
    pub fn into_parts(self) -> (St, L) {
        (self.storage, self.layout)
    }
}

impl<St, L> Clone for TensorData<St, L>
where
    St: Storage + StorageFor<L> + Clone,
    L: TensorLayout + Clone,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            layout: self.layout.clone(),
        }
    }
}
