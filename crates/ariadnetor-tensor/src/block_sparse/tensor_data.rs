//! Convenience constructors and joined accessors for `BlockSparseTensorData<T, S>`.

use std::collections::HashMap;
use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};
use arnet_core::backend::MemoryOrder;
use num_traits::Zero;

use super::{BlockCoord, BlockMeta, BlockSparseLayout, BlockSparseStorage, QNIndex};
use crate::{Sector, TensorData};

/// Backend-less BlockSparse tensor bundle =
/// `TensorData<BlockSparseStorage<T>, BlockSparseLayout<S>>`.
pub type BlockSparseTensorData<T, S> = TensorData<BlockSparseStorage<T>, BlockSparseLayout<S>>;

impl<T, S: Sector> BlockSparseTensorData<T, S> {
    /// Construct a zero-filled `BlockSparseTensorData` with all
    /// flux-allowed blocks.
    pub fn zeros(indices: Vec<QNIndex<S>>, flux: S, order: MemoryOrder) -> Self
    where
        T: Clone + Zero,
    {
        let layout = BlockSparseLayout::new(indices, flux, order);
        let extent = <BlockSparseLayout<S> as crate::TensorLayout>::storage_extent(&layout);
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, extent);
        data.resize(extent, T::zero());
        let storage = BlockSparseStorage::from_aligned(data);
        Self::new(storage, layout)
    }

    /// Construct with all flux-allowed blocks filled with random
    /// values from the standard distribution.
    pub fn random<R: rand::Rng>(
        indices: Vec<QNIndex<S>>,
        flux: S,
        order: MemoryOrder,
        rng: &mut R,
    ) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let layout = BlockSparseLayout::new(indices, flux, order);
        let extent = <BlockSparseLayout<S> as crate::TensorLayout>::storage_extent(&layout);
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, extent);
        for _ in 0..extent {
            data.push(rng.random());
        }
        let storage = BlockSparseStorage::from_aligned(data);
        Self::new(storage, layout)
    }

    /// Construct from pre-validated raw parts.
    ///
    /// Caller is responsible for the invariants enforced by
    /// [`BlockSparseLayout::new`]: sector conservation per block,
    /// coord uniqueness, packed offsets without gap or overlap,
    /// blocks sorted by coordinate. The `TensorData::new` assertion
    /// will additionally check `data.len() == sum(blocks.size)`.
    pub fn from_raw_parts(
        data: Vec<T>,
        blocks: Vec<BlockMeta>,
        block_index: HashMap<BlockCoord, usize>,
        indices: Vec<QNIndex<S>>,
        flux: S,
        shape: Vec<usize>,
        order: MemoryOrder,
    ) -> Self
    where
        T: Clone,
    {
        let layout =
            BlockSparseLayout::from_parts(blocks, block_index, indices, flux, shape, order);
        let storage = BlockSparseStorage::new(data);
        Self::new(storage, layout)
    }

    /// Data slice for a block identified by coordinate.
    ///
    /// Returns `None` if the block is not stored (zero by symmetry).
    pub fn block_data(&self, coord: &BlockCoord) -> Option<&[T]> {
        let &idx = self.layout().block_index().get(coord)?;
        let meta = &self.layout().block_metas()[idx];
        Some(&self.storage().data()[meta.offset..meta.offset + meta.size])
    }

    /// Mutable data slice for a block identified by coordinate
    /// (triggers CoW on the storage half if shared).
    pub fn block_data_mut(&mut self, coord: &BlockCoord) -> Option<&mut [T]>
    where
        T: Clone,
    {
        let &idx = self.layout().block_index().get(coord)?;
        let meta = &self.layout().block_metas()[idx];
        let offset = meta.offset;
        let size = meta.size;
        let arc = self.storage_mut().arc_mut();
        let data = Arc::make_mut(arc);
        Some(&mut data[offset..offset + size])
    }
}
