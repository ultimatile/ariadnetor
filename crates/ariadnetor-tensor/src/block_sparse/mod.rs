//! Block-sparse tensor storage for abelian symmetries.
//!
//! Provides [`BlockSparse<T, S>`] — a tensor storage type where only blocks
//! satisfying a conservation law (flux) are allocated.
//!
//! # Key types
//!
//! - [`Direction`] — leg direction (Out/In) for flux computation
//! - [`QNIndex<S>`] — quantum-number index mapping sectors to block dimensions
//! - [`BlockCoord`] — N-dimensional block coordinate
//! - [`BlockMeta`] — per-block metadata (coordinate, offset, size)
//! - [`BlockSparse<T, S>`] — the main storage struct

use std::collections::HashMap;
use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};

use crate::sector::Sector;

mod scalar_ops;

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

/// Leg direction for flux computation.
///
/// - `Out` (ket / row index): sector contributes as-is to flux
/// - `In` (bra / column index): sector contributes via `dual()` to flux
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    Out,
    In,
}

impl Direction {
    /// Apply direction to a sector: identity for `Out`, `dual()` for `In`.
    pub fn apply<S: Sector>(&self, sector: &S) -> S {
        match self {
            Direction::Out => sector.clone(),
            Direction::In => sector.dual(),
        }
    }
}

// ---------------------------------------------------------------------------
// QNIndex
// ---------------------------------------------------------------------------

/// Quantum-number index for one tensor leg.
///
/// Maps each sector to a block dimension, with a direction for flux computation.
///
/// # Invariants (enforced by constructor)
///
/// - `blocks` is sorted by sector (`Ord`)
/// - No duplicate sectors
/// - Every block dimension is > 0
#[derive(Clone, Debug)]
pub struct QNIndex<S: Sector> {
    /// Sector → block dimension pairs, sorted by sector, no duplicates.
    blocks: Vec<(S, usize)>,
    /// Leg direction.
    direction: Direction,
}

impl<S: Sector> QNIndex<S> {
    /// Create a new QN index.
    ///
    /// `blocks` is sorted by sector. Panics if any block dimension is zero
    /// or if duplicate sectors are present.
    pub fn new(mut blocks: Vec<(S, usize)>, direction: Direction) -> Self {
        blocks.sort_by(|a, b| a.0.cmp(&b.0));

        for (i, (sector, dim)) in blocks.iter().enumerate() {
            assert!(
                *dim > 0,
                "QNIndex: block dimension must be > 0 for sector {sector:?}"
            );
            if i > 0 {
                assert!(
                    blocks[i - 1].0 != *sector,
                    "QNIndex: duplicate sector {sector:?}"
                );
            }
        }

        Self { blocks, direction }
    }

    /// Sector–dimension pairs (sorted by sector).
    pub fn blocks(&self) -> &[(S, usize)] {
        &self.blocks
    }

    /// Leg direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Number of distinct sectors (blocks) in this index.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Total dimension (sum of all block dimensions).
    pub fn total_dim(&self) -> usize {
        self.blocks.iter().map(|(_, d)| d).sum()
    }

    /// Block dimension for a given block index.
    ///
    /// Panics if `idx >= self.num_blocks()`.
    pub fn block_dim(&self, idx: usize) -> usize {
        self.blocks[idx].1
    }

    /// Sector for a given block index.
    ///
    /// Panics if `idx >= self.num_blocks()`.
    pub fn sector(&self, idx: usize) -> &S {
        &self.blocks[idx].0
    }
}

// ---------------------------------------------------------------------------
// BlockCoord
// ---------------------------------------------------------------------------

/// N-dimensional block coordinate.
///
/// Each element is an index into the corresponding `QNIndex.blocks`.
/// `Ord` is derived (lexicographic) to define a deterministic sort order
/// for `Vec<BlockMeta>`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockCoord(pub Vec<usize>);

// ---------------------------------------------------------------------------
// BlockMeta
// ---------------------------------------------------------------------------

/// Metadata for a single block within [`BlockSparse`].
#[derive(Clone, Debug)]
pub struct BlockMeta {
    /// Block coordinate (index into each leg's QNIndex).
    pub coord: BlockCoord,
    /// Element offset into the flat data buffer.
    pub offset: usize,
    /// Number of elements in this block.
    pub size: usize,
}

// ---------------------------------------------------------------------------
// BlockSparse
// ---------------------------------------------------------------------------

/// Block-sparse tensor storage for abelian symmetries.
///
/// Only blocks whose sectors satisfy the flux conservation law are allocated.
/// Data is stored in a single flat buffer for cache-friendly access and
/// CoW semantics via `Arc`.
pub struct BlockSparse<T, S: Sector> {
    /// Flat data buffer holding all block data contiguously (64-byte aligned, Arc for CoW).
    data: Arc<AVec<T, ConstAlign<64>>>,

    /// Block metadata, sorted by `BlockCoord` (deterministic order).
    blocks: Vec<BlockMeta>,

    /// Auxiliary index: block coordinate → index into `blocks`.
    block_index: HashMap<BlockCoord, usize>,

    /// QN index for each tensor leg.
    indices: Vec<QNIndex<S>>,

    /// Tensor flux (conserved total quantum number).
    flux: S,

    /// Cached logical shape (total dimension per leg).
    shape: Vec<usize>,
}

impl<T, S: Sector> BlockSparse<T, S> {
    /// Get the flux (conserved quantum number) of this tensor.
    pub fn flux(&self) -> &S {
        &self.flux
    }

    /// Get the QN indices for all legs.
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

    /// Logical shape (total dimension per leg).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Rank (number of legs).
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Total number of stored elements across all blocks.
    pub fn stored_len(&self) -> usize {
        self.data.len()
    }

    /// Data slice for a block identified by coordinate.
    ///
    /// Returns `None` if the block is not stored (zero by symmetry).
    pub fn block_data(&self, coord: &BlockCoord) -> Option<&[T]> {
        let &idx = self.block_index.get(coord)?;
        let meta = &self.blocks[idx];
        Some(&self.data[meta.offset..meta.offset + meta.size])
    }

    /// Shape of a specific block.
    ///
    /// Each element is the block dimension of the corresponding leg at the
    /// given block index.
    ///
    /// Returns `None` if the coordinate is invalid (out-of-range block index).
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
    ///
    /// A block is allowed iff:
    /// `fuse(direction_applied(sector(b_0)), ..., direction_applied(sector(b_{n-1}))) == flux`
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

// Manual Clone: Arc::clone does not require T: Clone (same pattern as Dense<T>).
impl<T, S: Sector> Clone for BlockSparse<T, S> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            blocks: self.blocks.clone(),
            block_index: self.block_index.clone(),
            indices: self.indices.clone(),
            flux: self.flux.clone(),
            shape: self.shape.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Construction API
// ---------------------------------------------------------------------------

impl<T: Clone, S: Sector> BlockSparse<T, S> {
    /// Construct a `BlockSparse` from pre-validated components.
    ///
    /// Performs consistency checks (coord validity, flux conservation,
    /// offset/size bounds, contiguous packing) but does NOT enumerate
    /// allowed blocks — the caller is responsible for providing correct blocks.
    ///
    /// # Panics
    ///
    /// Panics if any invariant is violated.
    #[cfg(test)]
    pub(crate) fn from_raw_parts(
        data: Vec<T>,
        blocks: Vec<BlockMeta>,
        indices: Vec<QNIndex<S>>,
        flux: S,
    ) -> Self {
        let rank = indices.len();
        let shape: Vec<usize> = indices.iter().map(|idx| idx.total_dim()).collect();

        // Validate blocks
        let mut block_index = HashMap::with_capacity(blocks.len());
        for (i, meta) in blocks.iter().enumerate() {
            assert_eq!(
                meta.coord.0.len(),
                rank,
                "BlockMeta coord rank mismatch: expected {rank}, got {}",
                meta.coord.0.len()
            );

            // Validate coord bounds
            for (axis, &block_idx) in meta.coord.0.iter().enumerate() {
                assert!(
                    block_idx < indices[axis].num_blocks(),
                    "Block index {block_idx} out of range for axis {axis} (max {})",
                    indices[axis].num_blocks()
                );
            }

            // Validate size matches block shape
            let expected_size: usize = meta
                .coord
                .0
                .iter()
                .enumerate()
                .map(|(axis, &bi)| indices[axis].block_dim(bi))
                .product();
            assert_eq!(
                meta.size, expected_size,
                "BlockMeta size mismatch at coord {:?}: expected {expected_size}, got {}",
                meta.coord, meta.size
            );

            // Validate offset + size within data bounds
            assert!(
                meta.offset + meta.size <= data.len(),
                "BlockMeta at coord {:?} exceeds data buffer: offset {} + size {} > len {}",
                meta.coord,
                meta.offset,
                meta.size,
                data.len()
            );

            // Validate flux conservation
            let mut fused = S::identity();
            for (axis, &bi) in meta.coord.0.iter().enumerate() {
                let sector = indices[axis].sector(bi);
                let directed = indices[axis].direction().apply(sector);
                fused = fused.fuse(&directed);
            }
            assert_eq!(
                fused, flux,
                "Block {:?} violates flux conservation: fused {:?} != flux {:?}",
                meta.coord, fused, flux
            );

            let prev = block_index.insert(meta.coord.clone(), i);
            assert!(
                prev.is_none(),
                "Duplicate block coordinate {:?}",
                meta.coord
            );
        }

        // Verify blocks are sorted by coord
        for i in 1..blocks.len() {
            assert!(
                blocks[i - 1].coord < blocks[i].coord,
                "Blocks not sorted: {:?} >= {:?}",
                blocks[i - 1].coord,
                blocks[i].coord
            );
        }

        // Verify blocks tile the data buffer contiguously without gaps or overlaps
        let mut expected_offset = 0;
        // Sort by offset to verify packing (blocks are sorted by coord, not offset)
        let mut offset_order: Vec<usize> = (0..blocks.len()).collect();
        offset_order.sort_by_key(|&i| blocks[i].offset);
        for &i in &offset_order {
            assert_eq!(
                blocks[i].offset, expected_offset,
                "Block {:?} has offset {} but expected {} (gap or overlap)",
                blocks[i].coord, blocks[i].offset, expected_offset
            );
            expected_offset += blocks[i].size;
        }
        assert_eq!(
            expected_offset,
            data.len(),
            "Data buffer has {} elements but blocks cover only {}",
            data.len(),
            expected_offset
        );

        let aligned_data = AVec::from_slice(64, &data);

        Self {
            data: Arc::new(aligned_data),
            blocks,
            block_index,
            indices,
            flux,
            shape,
        }
    }
}

impl<T, S: Sector> BlockSparse<T, S> {
    /// Construct a zero-filled `BlockSparse` with all flux-allowed blocks.
    ///
    /// Enumerates every block coordinate satisfying the flux conservation law
    /// and allocates a contiguous zero-filled data buffer.
    pub fn zeros(indices: Vec<QNIndex<S>>, flux: S) -> Self
    where
        T: Clone + num_traits::Zero,
    {
        let shape: Vec<usize> = indices.iter().map(|idx| idx.total_dim()).collect();
        let rank = indices.len();
        let num_blocks_per_leg: Vec<usize> = indices.iter().map(|idx| idx.num_blocks()).collect();

        let mut blocks = Vec::new();
        let mut total_size = 0usize;

        // Enumerate all block coordinate combinations in lexicographic order.
        // Skip if any leg has no sectors (tensor is empty).
        if rank == 0 || num_blocks_per_leg.iter().all(|&n| n > 0) {
            let mut current = vec![0usize; rank];
            loop {
                // Check flux conservation for this coordinate
                let mut fused = S::identity();
                for (axis, &bi) in current.iter().enumerate() {
                    let sector = indices[axis].sector(bi);
                    let directed = indices[axis].direction().apply(sector);
                    fused = fused.fuse(&directed);
                }

                if fused == flux {
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

                // Increment multi-index (lexicographic order)
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

        let mut data = AVec::<T, ConstAlign<64>>::with_capacity(64, total_size);
        data.resize(total_size, T::zero());

        Self {
            data: Arc::new(data),
            blocks,
            block_index,
            indices,
            flux,
            shape,
        }
    }

    /// Construct a `BlockSparse` with all flux-allowed blocks filled with
    /// random values from the standard distribution.
    ///
    /// The tensor structure (shape, blocks, flux) is identical to [`Self::zeros`];
    /// only the data differs.
    pub fn random<R: rand::Rng>(indices: Vec<QNIndex<S>>, flux: S, rng: &mut R) -> Self
    where
        T: Clone + num_traits::Zero,
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let mut tensor = Self::zeros(indices, flux);
        for meta in tensor.block_metas().to_vec() {
            let block = tensor.block_data_mut(&meta.coord).unwrap();
            for elem in block.iter_mut() {
                *elem = rng.random();
            }
        }
        tensor
    }

    /// Mutable data slice for a block identified by coordinate (CoW).
    ///
    /// Triggers a copy of the entire data buffer if other clones exist.
    /// Returns `None` if the block is not stored.
    pub fn block_data_mut(&mut self, coord: &BlockCoord) -> Option<&mut [T]>
    where
        T: Clone,
    {
        let &idx = self.block_index.get(coord)?;
        let meta = &self.blocks[idx];
        let offset = meta.offset;
        let size = meta.size;
        let data = Arc::make_mut(&mut self.data);
        Some(&mut data[offset..offset + size])
    }
}

#[cfg(test)]
mod tests;
