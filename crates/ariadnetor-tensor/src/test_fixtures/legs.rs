//! BlockSparse `QNIndex` leg-construction builders.
//!
//! The workspace's BlockSparse tests hand-roll `QNIndex` leg pairs at many
//! sites. These builders centralize that construction: the leg-shape intent
//! (square / non-square / general) reads at the call site, and a `QNIndex::new`
//! signature change is absorbed here rather than at every test call site. They
//! are generic over the sector type so both `U1Sector` and `Z2Sector` fixtures
//! share them.

use crate::block_sparse::{Direction, QNIndex};
use crate::sector::Sector;

/// Build one `QNIndex` leg per `(sectors, direction)` spec, in order.
///
/// The general builder underpinning [`out_in_legs`] and [`square_legs`]; use it
/// directly for the irregular shapes — `Out`/`Out` pairs and rank-N legs.
pub fn legs<S: Sector>(
    specs: impl IntoIterator<Item = (Vec<(S, usize)>, Direction)>,
) -> Vec<QNIndex<S>> {
    specs
        .into_iter()
        .map(|(sectors, direction)| QNIndex::new(sectors, direction))
        .collect()
}

/// An `Out` row leg and an `In` column leg with independent sector lists.
pub fn out_in_legs<S: Sector>(row: Vec<(S, usize)>, col: Vec<(S, usize)>) -> Vec<QNIndex<S>> {
    legs([(row, Direction::Out), (col, Direction::In)])
}

/// An `Out` row leg and an `In` column leg sharing one sector list (square pair).
pub fn square_legs<S: Sector>(sectors: Vec<(S, usize)>) -> Vec<QNIndex<S>> {
    out_in_legs(sectors.clone(), sectors)
}
