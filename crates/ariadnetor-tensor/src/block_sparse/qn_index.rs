//! `QNIndex<S>`: per-leg quantum-number index with sorted sector blocks.

use super::Direction;
use crate::sector::Sector;

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
