//! Convenience constructors and joined accessors for `BlockSparseTensorData<T, S>`.

#[cfg(test)]
use std::collections::HashMap;
use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};
use arnet_core::backend::MemoryOrder;
use num_traits::Zero;

#[cfg(test)]
use super::BlockMeta;
use super::{BlockCoord, BlockSparseLayout, BlockSparseStorage, QNIndex};
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

    /// Construct by populating each flux-allowed block from a closure.
    ///
    /// The closure receives the block coordinate and its dense block
    /// shape (one entry per leg) and must return the block's flat data
    /// in the layout's memory `order`. Forbidden blocks are not
    /// queried. Block coordinates are visited in the layout's
    /// lexicographic enumeration order.
    ///
    /// # Panics
    ///
    /// Panics if the closure returns a `Vec<T>` whose length differs
    /// from `product(block_shape)` (the per-block element count).
    pub fn from_block_fn<F>(indices: Vec<QNIndex<S>>, flux: S, order: MemoryOrder, mut f: F) -> Self
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
    ///
    /// Test-only: no in-workspace consumer needs to bypass the
    /// enumerating constructors at runtime; publicize when a real
    /// deserialization or migration path materializes.
    #[cfg(test)]
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

    /// Total dimension per axis (read from layout).
    pub fn shape(&self) -> Vec<usize> {
        self.layout().shape().to_vec()
    }

    /// Rank (number of axes).
    pub fn rank(&self) -> usize {
        self.layout().rank()
    }

    /// QNIndex list (one per axis).
    pub fn indices(&self) -> &[QNIndex<S>] {
        self.layout().indices()
    }

    /// Symmetry flux.
    pub fn flux(&self) -> &S {
        self.layout().flux()
    }

    /// Number of allocated (flux-allowed) blocks.
    pub fn num_blocks(&self) -> usize {
        self.layout().num_blocks()
    }

    /// Block metadata for every allocated block.
    pub fn block_metas(&self) -> &[super::BlockMeta] {
        self.layout().block_metas()
    }

    /// Per-axis block dimensions for a given block coordinate.
    pub fn block_shape(&self, coord: &BlockCoord) -> Option<Vec<usize>> {
        self.layout().block_shape(coord)
    }

    /// Whether the given block coordinate is allowed by the flux
    /// conservation law: the per-axis directed sectors fuse to
    /// `flux` under the sector's group operation.
    pub fn is_allowed_block(&self, coord: &BlockCoord) -> bool {
        self.layout().is_allowed_block(coord)
    }

    /// Memory order of the per-block stored data.
    pub fn order(&self) -> MemoryOrder {
        self.layout().order()
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

    /// Repack per-block data to a different memory order.
    ///
    /// Returns a clone (Arc-shared) when the tensor is already in
    /// `target`. Otherwise allocates a fresh buffer, rearranges each
    /// block's flat data from the current order to `target`, and tags
    /// the new layout at `target`. Blocks of rank ≤ 1 are layout-
    /// invariant and copy through unchanged.
    pub fn to_order(&self, target: MemoryOrder) -> Self
    where
        T: Clone + Zero,
    {
        let current = self.layout().order();
        if current == target {
            return self.clone();
        }
        let layout = self.layout();
        let indices: Vec<_> = layout.indices().to_vec();
        let flux = layout.flux().clone();
        let mut out = Self::zeros(indices, flux, target);
        for meta in layout.block_metas() {
            let src = self
                .block_data(&meta.coord)
                .expect("layout-enumerated block must have storage");
            let block_shape: Vec<usize> = (0..layout.rank())
                .map(|a| layout.indices()[a].block_dim(meta.coord.0[a]))
                .collect();
            let dst = out
                .block_data_mut(&meta.coord)
                .expect("zero-initialized output must have matching block");
            if block_shape.len() <= 1 || src.is_empty() {
                dst.clone_from_slice(src);
                continue;
            }
            let axis_order: Vec<usize> = match target {
                MemoryOrder::RowMajor => (0..block_shape.len()).collect(),
                MemoryOrder::ColumnMajor => (0..block_shape.len()).rev().collect(),
            };
            let mut coords = vec![0usize; block_shape.len()];
            for dst_slot in dst.iter_mut() {
                let src_flat = crate::flat_index(&coords, &block_shape, current);
                *dst_slot = src[src_flat].clone();
                for &d in axis_order.iter().rev() {
                    coords[d] += 1;
                    if coords[d] < block_shape[d] {
                        break;
                    }
                    coords[d] = 0;
                }
            }
        }
        out
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
    /// Block coordinates and packed offsets are preserved by the
    /// per-layout transform. Involution: `x.dagger().dagger() == x`.
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

    /// Frobenius norm: √(Σ |element|²) summed over all stored blocks.
    pub fn norm(&self) -> T::Real {
        self.storage().norm()
    }
}
