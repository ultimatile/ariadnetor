//! Arc-move conversion to/from `BlockSparseTensorData<T, S>`.
//!
//! `BlockSparse<T, S>` and `BlockSparseTensorData<T, S>` share the
//! same flat data buffer representation (`Arc<AVec<T,
//! ConstAlign<64>>>`) and the same block-structure components.
//! During the storage / layout split migration (issue #259) the
//! linalg pub fns cross this boundary at every call; the converters
//! below move the `Arc` and reuse the block tables directly so the
//! boundary is O(1) and allocation-free.
//!
//! Legacy `BlockSparse<T, S>` has no `order` field, so
//! [`BlockSparse::into_tensor_data`] takes a `MemoryOrder` parameter
//! (typically `backend.preferred_order()`) and
//! [`BlockSparse::from_tensor_data`] discards `order`. The pair is
//! removed in Unit 5 when `BlockSparse<T, S>` is deleted.

use arnet_core::backend::MemoryOrder;

use super::{BlockSparse, BlockSparseLayout, BlockSparseStorage, BlockSparseTensorData, Sector};

impl<T, S: Sector> BlockSparse<T, S> {
    /// Consume `self` and return a `BlockSparseTensorData<T, S>`
    /// sharing the same underlying buffer. The caller supplies the
    /// `order` since `BlockSparse<T, S>` does not carry one.
    pub fn into_tensor_data(self, order: MemoryOrder) -> BlockSparseTensorData<T, S> {
        let storage = BlockSparseStorage::from_arc(self.data);
        let layout = BlockSparseLayout::from_parts(
            self.blocks,
            self.block_index,
            self.indices,
            self.flux,
            self.shape,
            order,
        );
        BlockSparseTensorData::new(storage, layout)
    }

    /// Build a `BlockSparse<T, S>` from an existing
    /// `BlockSparseTensorData<T, S>`, reusing the `Arc` and block
    /// tables. The layout's `order` is dropped.
    pub fn from_tensor_data(td: BlockSparseTensorData<T, S>) -> Self {
        let (storage, layout) = td.into_parts();
        let (blocks, block_index, indices, flux, shape, _order) = layout.into_parts();
        Self {
            data: storage.into_arc(),
            blocks,
            block_index,
            indices,
            flux,
            shape,
        }
    }
}
