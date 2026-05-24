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
    ///
    /// Crate-internal: the user-facing constructor is
    /// [`BlockSparseTensor::zeros`](crate::BlockSparseTensor::zeros),
    /// which pins memory order to the active backend. Direct callers
    /// that need an explicit `order` go through this helper or build
    /// `TensorData::new(storage, layout)` directly.
    pub(crate) fn zeros(indices: Vec<QNIndex<S>>, flux: S, order: MemoryOrder) -> Self
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

    /// Construct by populating each flux-allowed block from a closure.
    ///
    /// The closure receives the block coordinate and its dense block
    /// shape (one entry per leg) and must return the block's flat data
    /// in the layout's memory `order`. Forbidden blocks are not
    /// queried. Block coordinates are visited in the layout's
    /// lexicographic enumeration order.
    ///
    /// Crate-internal: the user-facing constructor is
    /// [`BlockSparseTensor::from_block_fn`](crate::BlockSparseTensor::from_block_fn),
    /// which pins memory order to the active backend.
    ///
    /// # Panics
    ///
    /// Panics if the closure returns a `Vec<T>` whose length differs
    /// from `product(block_shape)` (the per-block element count).
    pub(crate) fn from_block_fn<F>(
        indices: Vec<QNIndex<S>>,
        flux: S,
        order: MemoryOrder,
        mut f: F,
    ) -> Self
    where
        T: Clone + Zero,
        F: FnMut(&BlockCoord, &[usize]) -> Vec<T>,
    {
        let layout = BlockSparseLayout::new(indices, flux, order);
        let extent = <BlockSparseLayout<S> as crate::TensorLayout>::storage_extent(&layout);
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, extent);
        data.resize(extent, T::zero());
        for meta in layout.block_metas() {
            let block_shape = layout
                .block_shape(&meta.coord)
                .expect("BlockSparseLayout enumerated coord must resolve to a block shape");
            let block = f(&meta.coord, &block_shape);
            assert_eq!(
                block.len(),
                meta.size,
                "from_block_fn: closure returned {} elements for block {:?}, expected {}",
                block.len(),
                meta.coord,
                meta.size,
            );
            for (dst, src) in data[meta.offset..meta.offset + meta.size]
                .iter_mut()
                .zip(block)
            {
                *dst = src;
            }
        }
        let storage = BlockSparseStorage::from_aligned(data);
        Self::new(storage, layout)
    }

    /// Construct with all flux-allowed blocks filled with random
    /// values from the standard distribution.
    ///
    /// Crate-internal: the user-facing constructor is
    /// [`BlockSparseTensor::random`](crate::BlockSparseTensor::random),
    /// which pins memory order to the active backend.
    pub(crate) fn random<R: rand::Rng>(
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
    ///
    /// Internal kernel-output bridge. The joined-surface
    /// `BlockSparseTensor::from_raw_parts` wraps this with an
    /// explicit backend; direct callers stay inside `arnet-tensor`.
    pub(crate) fn from_raw_parts(
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

impl<T, S: Sector> BlockSparseTensorData<T, S> {
    /// Cheap O(1) `BlockSparse<T, S>` view that shares the underlying
    /// storage Arc and clones the block metadata.
    ///
    /// Bridges the joined-form `BlockSparseTensorData` into the legacy
    /// [`BlockSparse<T, S>`](crate::BlockSparse) representation that
    /// internal linalg kernels still operate on. The memory `order`
    /// recorded on the layout is dropped because the legacy
    /// `BlockSparse` type predates the per-tensor order field.
    pub fn as_block_sparse(&self) -> super::BlockSparse<T, S> {
        super::BlockSparse::from_storage_arc(
            self.storage().arc_clone(),
            self.layout().block_metas().to_vec(),
            self.layout().block_index().clone(),
            self.layout().indices().to_vec(),
            self.layout().flux().clone(),
            self.layout().shape().to_vec(),
        )
    }
}

impl<T, S: Sector> BlockSparseTensorData<T, S>
where
    T: arnet_core::Scalar,
{
    /// Hermitian adjoint: element-wise conjugation of the data, flip
    /// of every QNIndex direction (Out↔In), and dualization of the
    /// flux.
    ///
    /// Block coordinates and packed offsets are preserved
    /// ([`BlockSparseLayout::dagger_layout`] reuses them). Involution:
    /// `x.dagger().dagger() == x`.
    pub fn dagger(&self) -> Self {
        let new_layout = self.layout().dagger_layout();
        let new_data: AVec<T, ConstAlign<64>> =
            AVec::from_iter(64, self.storage().data().iter().copied().map(|x| x.conj()));
        let storage = BlockSparseStorage::from_aligned(new_data);
        Self::new(storage, new_layout)
    }

    /// Element-wise complex conjugate. Layout (including directions
    /// and flux) is preserved; use [`dagger`](Self::dagger) when the
    /// adjoint structure is required for inner products.
    pub fn conj(&self) -> Self {
        let new_data: AVec<T, ConstAlign<64>> =
            AVec::from_iter(64, self.storage().data().iter().copied().map(|x| x.conj()));
        let storage = BlockSparseStorage::from_aligned(new_data);
        Self::new(storage, self.layout().clone())
    }
}
