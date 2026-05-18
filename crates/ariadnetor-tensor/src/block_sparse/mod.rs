//! Block-sparse tensor storage for abelian symmetries.
//!
//! Provides the storage / layout split for the block-sparse case:
//!
//! - [`BlockSparseStorage<T>`] — pure-data half (packed 64-byte-aligned
//!   buffer with Arc-based Copy-on-Write).
//! - [`BlockSparseLayout<S>`] — interpretation half (allowed-block
//!   enumeration, per-leg sector indices, flux, shape, memory order).
//! - [`BlockSparseTensorData<T, S>`] — joined bundle
//!   = `TensorData<BlockSparseStorage<T>, BlockSparseLayout<S>>`.
//!
//! # Key types
//!
//! - [`Direction`] — leg direction (Out/In) for flux computation
//! - [`QNIndex<S>`] — quantum-number index mapping sectors to block dimensions
//! - [`BlockCoord`] — N-dimensional block coordinate
//! - [`BlockMeta`] — per-block metadata (coordinate, offset, size)

use crate::sector::Sector;

mod layout;
mod storage;
mod tensor_data;

pub use layout::BlockSparseLayout;
pub use storage::BlockSparseStorage;
pub use tensor_data::BlockSparseTensorData;

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

/// Metadata for a single block.
#[derive(Clone, Debug)]
pub struct BlockMeta {
    /// Block coordinate (index into each leg's QNIndex).
    pub coord: BlockCoord,
    /// Element offset into the flat data buffer.
    pub offset: usize,
    /// Number of elements in this block.
    pub size: usize,
}

#[cfg(test)]
mod tests;
