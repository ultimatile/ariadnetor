//! `BlockSparseLayout<S>`: interpretation half of the block-sparse tensor split.
//!
//! Carries block metadata (allowed-block enumeration with offsets and
//! sizes), per-leg sector indices, flux, logical shape, and memory
//! order. Data lives on
//! [`BlockSparseStorage<T>`](crate::BlockSparseStorage); the wrapper
//! [`BlockSparseTensorData<T, S>`](crate::BlockSparseTensorData) joins
//! the two with a length-consistency check.

use std::collections::HashMap;

use ariadnetor_core::backend::MemoryOrder;
use thiserror::Error;

use super::{BlockCoord, BlockMeta, Direction, QNIndex};
use crate::serialize::SerializableSector;
use crate::{Sector, TensorLayout};

/// Overflow encountered while enumerating flux-allowed blocks in the
/// panic-free [`BlockSparseLayout::try_new`] path.
///
/// The infallible [`BlockSparseLayout::new`] would panic on these; `try_new`
/// exists so a load reconstructing a layout from crafted input reports a
/// typed error instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BlockLayoutError {
    /// Sector fusion (or its dual) overflowed while testing a block.
    #[error("sector fusion overflow while enumerating blocks")]
    FusionOverflow,
    /// A block's element count (product of per-leg dims) overflowed `usize`.
    #[error("block size overflow while enumerating blocks")]
    SizeOverflow,
    /// A running offset, the cached storage extent, or a leg's total
    /// dimension overflowed `usize`.
    #[error("storage extent overflow while enumerating blocks")]
    ExtentOverflow,
}

/// Interpretation half of the block-sparse tensor split.
///
/// Holds the allowed-block enumeration (sorted by coordinate, packed
/// offsets), the per-leg sector indices, the conserved flux, the
/// cached logical shape, and the memory order the paired
/// [`BlockSparseStorage`](crate::BlockSparseStorage) is laid out in.
///
/// Construction goes through [`new`](Self::new), which enumerates
/// flux-allowed blocks and produces a packed layout, or its panic-free
/// counterpart [`try_new`](Self::try_new) for reconstruction from
/// untrusted input. Layout-internal invariants (sector conservation,
/// coord uniqueness, no-gap packing) hold by construction; the
/// storage-layout boundary check happens in
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

    /// Hermitian-adjoint layout: flip every QNIndex direction (Out↔In)
    /// and dual the flux.
    ///
    /// The allowed-block set is preserved: each block's flux
    /// contribution becomes `dual(direction.apply(sector))`, whose sum
    /// equals the dualed flux exactly when the original sum equalled
    /// the original flux (abelian dual is a group homomorphism).
    /// `blocks`, `block_index`, `shape`, `order`, and `storage_extent`
    /// are reused as-is.
    pub(crate) fn dagger_layout(&self) -> Self {
        let flipped_indices: Vec<QNIndex<S>> = self
            .indices
            .iter()
            .map(|idx| {
                let new_dir = match idx.direction() {
                    Direction::Out => Direction::In,
                    Direction::In => Direction::Out,
                };
                QNIndex::new(idx.blocks().to_vec(), new_dir)
            })
            .collect();
        Self {
            blocks: self.blocks.clone(),
            block_index: self.block_index.clone(),
            indices: flipped_indices,
            flux: self.flux.dual(),
            shape: self.shape.clone(),
            order: self.order,
            storage_extent: self.storage_extent,
        }
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

impl<S: SerializableSector> BlockSparseLayout<S> {
    /// Panic-free counterpart to [`new`](Self::new) for reconstruction from
    /// untrusted input.
    ///
    /// Enumerates the same flux-allowed blocks in the same order as `new`, but
    /// uses [`checked_fuse`](SerializableSector::checked_fuse) /
    /// [`checked_dual`](SerializableSector::checked_dual) and checked size /
    /// offset / extent arithmetic, returning [`BlockLayoutError`] where `new`
    /// would panic. It does not delegate to `new`, and `new` does not delegate
    /// to it — the two coexist so `new`'s hot path keeps its unchecked
    /// arithmetic.
    pub fn try_new(
        indices: Vec<QNIndex<S>>,
        flux: S,
        order: MemoryOrder,
    ) -> Result<Self, BlockLayoutError> {
        let rank = indices.len();

        // Leg total dimensions via checked sums: crafted per-block dims could
        // otherwise overflow the logical shape.
        let mut shape = Vec::with_capacity(rank);
        for idx in &indices {
            let mut total = 0usize;
            for &(_, dim) in idx.blocks() {
                total = total
                    .checked_add(dim)
                    .ok_or(BlockLayoutError::ExtentOverflow)?;
            }
            shape.push(total);
        }

        let num_blocks_per_leg: Vec<usize> = indices.iter().map(|idx| idx.num_blocks()).collect();
        let mut blocks = Vec::new();
        let mut storage_extent = 0usize;

        if rank == 0 || num_blocks_per_leg.iter().all(|&n| n > 0) {
            let mut current = vec![0usize; rank];
            loop {
                let mut fused = S::identity();
                let mut fusion_ok = true;
                for (axis, &bi) in current.iter().enumerate() {
                    let sector = indices[axis].sector(bi);
                    let directed = match indices[axis].direction() {
                        Direction::Out => sector.clone(),
                        Direction::In => match sector.checked_dual() {
                            Some(dual) => dual,
                            None => {
                                fusion_ok = false;
                                break;
                            }
                        },
                    };
                    match fused.checked_fuse(&directed) {
                        Some(next) => fused = next,
                        None => {
                            fusion_ok = false;
                            break;
                        }
                    }
                }
                if !fusion_ok {
                    return Err(BlockLayoutError::FusionOverflow);
                }

                if fused == flux {
                    let mut size = 1usize;
                    for (axis, &bi) in current.iter().enumerate() {
                        size = size
                            .checked_mul(indices[axis].block_dim(bi))
                            .ok_or(BlockLayoutError::SizeOverflow)?;
                    }
                    blocks.push(BlockMeta {
                        coord: BlockCoord(current.clone()),
                        offset: storage_extent,
                        size,
                    });
                    storage_extent = storage_extent
                        .checked_add(size)
                        .ok_or(BlockLayoutError::ExtentOverflow)?;
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

        Ok(Self {
            blocks,
            block_index,
            indices,
            flux,
            shape,
            order,
            storage_extent,
        })
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
