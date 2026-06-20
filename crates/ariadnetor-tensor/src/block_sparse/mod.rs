//! Block-sparse tensor storage for abelian symmetries.
//!
//! Provides [`BlockSparseTensorData<T, S>`] — the joined storage +
//! layout bundle for tensors whose blocks are constrained by an
//! abelian conservation law (flux). Only blocks satisfying the
//! conservation law are allocated; the packed flat buffer lives on
//! [`BlockSparseStorage<T>`] and the per-leg sector indices /
//! allowed-block metadata live on [`BlockSparseLayout<S>`].
//!
//! # Key types
//!
//! - [`Direction`] — leg direction (Out/In) for flux computation
//! - [`QNIndex<S>`] — quantum-number index mapping sectors to block dimensions
//! - [`BlockCoord`] — N-dimensional block coordinate
//! - [`BlockMeta`] — per-block metadata (coordinate, offset, size)
//! - [`BlockSparseStorage<T>`] / [`BlockSparseLayout<S>`] /
//!   [`BlockSparseTensorData<T, S>`] — the storage / layout / joined
//!   bundle

use crate::sector::Sector;

mod layout;
mod qn_index;
mod storage;
mod tensor_data;

pub use layout::BlockSparseLayout;
pub use qn_index::QNIndex;
pub use storage::BlockSparseStorage;
pub use tensor_data::BlockSparseTensorData;

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

/// Leg direction for flux computation (see each variant).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Ket / row index: the sector contributes as-is to the flux.
    Out,
    /// Bra / column index: the sector contributes via `dual()` to the flux.
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

/// Metadata for a single block within a block-sparse tensor.
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
