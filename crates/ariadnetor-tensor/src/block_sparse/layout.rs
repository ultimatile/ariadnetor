//! `BlockSparseLayout<S>`: interpretation half of the block-sparse tensor split.
//!
//! Carries block metadata (allowed-block enumeration with offsets and
//! sizes), per-leg sector indices, flux, logical shape, and memory
//! order. Data lives on
//! [`BlockSparseStorage<T>`](crate::BlockSparseStorage); the wrapper
//! [`BlockSparseTensorData<T, S>`](crate::BlockSparseTensorData) joins
//! the two with a length-consistency check.

use std::collections::HashMap;

use arnet_core::backend::MemoryOrder;

use super::{BlockCoord, BlockMeta, QNIndex};
use crate::{Sector, TensorLayout};

/// Interpretation half of the block-sparse tensor split.
///
/// Holds the allowed-block enumeration (sorted by coordinate, packed
/// offsets), the per-leg sector indices, the conserved flux, the
/// cached logical shape, and the memory order the paired
/// [`BlockSparseStorage`](crate::BlockSparseStorage) is laid out in.
///
/// Construction goes through [`new`](Self::new), which enumerates
/// flux-allowed blocks and produces a packed layout. Layout-internal
/// invariants (sector conservation, coord uniqueness, no-gap
/// packing) hold by construction; the storage-layout boundary check
/// happens in
/// [`TensorData::new`](crate::TensorData::new).
#[derive(Clone)]
pub struct BlockSparseLayout<S: Sector> {
    blocks: Vec<BlockMeta>,
    block_index: HashMap<BlockCoord, usize>,
    indices: Vec<QNIndex<S>>,
    flux: S,
    shape: Vec<usize>,
    order: MemoryOrder,
    /// Cached sum of allowed block sizes; equals expected
    /// [`BlockSparseStorage::flat_len`](crate::BlockSparseStorage::flat_len).
    storage_extent: usize,
}

impl<S: Sector> BlockSparseLayout<S> {
    /// Construct a layout by enumerating flux-allowed blocks.
    ///
    /// The resulting layout has blocks sorted by coordinate
    /// (lexicographic) with packed offsets (no gaps or overlaps),
    /// every block satisfying the flux-conservation law, and a
    /// cached `storage_extent` equal to the sum of allowed block
    /// sizes.
    pub fn new(indices: Vec<QNIndex<S>>, flux: S, order: MemoryOrder) -> Self {
        let (blocks, block_index, shape, storage_extent) =
            enumerate_allowed_blocks(&indices, &flux);
        Self {
            blocks,
            block_index,
            indices,
            flux,
            shape,
            order,
            storage_extent,
        }
    }

    /// Construct directly from pre-validated components.
    ///
    /// Used by joined-level constructors that already have the
    /// structure on hand; caller is responsible for the same
    /// invariants enforced by [`new`](Self::new).
    pub(crate) fn from_parts(
        blocks: Vec<BlockMeta>,
        block_index: HashMap<BlockCoord, usize>,
        indices: Vec<QNIndex<S>>,
        flux: S,
        shape: Vec<usize>,
        order: MemoryOrder,
    ) -> Self {
        let storage_extent = blocks.iter().map(|b| b.size).sum();
        Self {
            blocks,
            block_index,
            indices,
            flux,
            shape,
            order,
            storage_extent,
        }
    }

    /// Conserved flux (total quantum number).
    pub fn flux(&self) -> &S {
        &self.flux
    }

    /// Per-leg QN indices.
    pub fn indices(&self) -> &[QNIndex<S>] {
        &self.indices
    }

    /// Number of stored (non-zero) blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Block metadata (sorted by coordinate).
    pub fn block_metas(&self) -> &[BlockMeta] {
        &self.blocks
    }

    /// Block-coordinate → blocks index lookup.
    pub(crate) fn block_index(&self) -> &HashMap<BlockCoord, usize> {
        &self.block_index
    }

    /// Logical shape (total dimension per leg).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Rank (number of legs).
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Memory order the paired storage is laid out in.
    pub fn order(&self) -> MemoryOrder {
        self.order
    }

    /// Shape of a specific block, or `None` if the coordinate is out of range.
    pub fn block_shape(&self, coord: &BlockCoord) -> Option<Vec<usize>> {
        if coord.0.len() != self.indices.len() {
            return None;
        }
        let mut shape = Vec::with_capacity(coord.0.len());
        for (axis, &block_idx) in coord.0.iter().enumerate() {
            if block_idx >= self.indices[axis].num_blocks() {
                return None;
            }
            shape.push(self.indices[axis].block_dim(block_idx));
        }
        Some(shape)
    }

    /// Check whether a block coordinate satisfies the flux conservation law.
    pub fn is_allowed_block(&self, coord: &BlockCoord) -> bool {
        if coord.0.len() != self.indices.len() {
            return false;
        }
        let mut fused = S::identity();
        for (axis, &block_idx) in coord.0.iter().enumerate() {
            let idx = &self.indices[axis];
            if block_idx >= idx.num_blocks() {
                return false;
            }
            let sector = idx.sector(block_idx);
            let directed = idx.direction().apply(sector);
            fused = fused.fuse(&directed);
        }
        fused == self.flux
    }
}

impl<S: Sector> TensorLayout for BlockSparseLayout<S> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn storage_extent(&self) -> usize {
        self.storage_extent
    }
}

/// Enumerate flux-allowed blocks for given indices and flux.
///
/// Returns `(blocks, block_index, shape, total_size)`. Blocks are
/// emitted in lexicographic coordinate order with packed offsets.
fn enumerate_allowed_blocks<S: Sector>(
    indices: &[QNIndex<S>],
    flux: &S,
) -> (
    Vec<BlockMeta>,
    HashMap<BlockCoord, usize>,
    Vec<usize>,
    usize,
) {
    let shape: Vec<usize> = indices.iter().map(|idx| idx.total_dim()).collect();
    let rank = indices.len();
    let num_blocks_per_leg: Vec<usize> = indices.iter().map(|idx| idx.num_blocks()).collect();

    let mut blocks = Vec::new();
    let mut total_size = 0usize;

    if rank == 0 || num_blocks_per_leg.iter().all(|&n| n > 0) {
        let mut current = vec![0usize; rank];
        loop {
            let mut fused = S::identity();
            for (axis, &bi) in current.iter().enumerate() {
                let sector = indices[axis].sector(bi);
                let directed = indices[axis].direction().apply(sector);
                fused = fused.fuse(&directed);
            }

            if fused == *flux {
                let size: usize = current
                    .iter()
                    .enumerate()
                    .map(|(axis, &bi)| indices[axis].block_dim(bi))
                    .product();
                blocks.push(BlockMeta {
                    coord: BlockCoord(current.clone()),
                    offset: total_size,
                    size,
                });
                total_size += size;
            }

            let mut carry = true;
            for axis in (0..rank).rev() {
                current[axis] += 1;
                if current[axis] < num_blocks_per_leg[axis] {
                    carry = false;
                    break;
                }
                current[axis] = 0;
            }
            if carry {
                break;
            }
        }
    }

    let mut block_index = HashMap::with_capacity(blocks.len());
    for (i, meta) in blocks.iter().enumerate() {
        block_index.insert(meta.coord.clone(), i);
    }

    (blocks, block_index, shape, total_size)
}
