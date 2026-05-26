mod construction;
mod tensor_data;

use super::*;
use crate::sector::{U1Sector, Z2Sector};

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

#[test]
fn direction_apply_out() {
    let s = U1Sector(3);
    assert_eq!(Direction::Out.apply(&s), U1Sector(3));
}

#[test]
fn direction_apply_in() {
    let s = U1Sector(3);
    assert_eq!(Direction::In.apply(&s), U1Sector(-3));
}

// ---------------------------------------------------------------------------
// QNIndex
// ---------------------------------------------------------------------------

#[test]
fn qnindex_sorts_blocks() {
    // Provide unsorted input; constructor should sort
    let idx = QNIndex::new(
        vec![(U1Sector(2), 3), (U1Sector(-1), 2), (U1Sector(0), 1)],
        Direction::Out,
    );
    let sectors: Vec<_> = idx.blocks().iter().map(|(s, _)| s.0).collect();
    assert_eq!(sectors, vec![-1, 0, 2]);
}

#[test]
fn qnindex_accessors() {
    let idx = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    assert_eq!(idx.num_blocks(), 2);
    assert_eq!(idx.total_dim(), 5);
    assert_eq!(idx.block_dim(0), 2);
    assert_eq!(idx.block_dim(1), 3);
    assert_eq!(*idx.sector(0), U1Sector(0));
    assert_eq!(*idx.sector(1), U1Sector(1));
    assert_eq!(idx.direction(), Direction::In);
}

#[test]
#[should_panic(expected = "block dimension must be > 0")]
fn qnindex_rejects_zero_dim() {
    QNIndex::new(vec![(U1Sector(0), 0)], Direction::Out);
}

#[test]
#[should_panic(expected = "duplicate sector")]
fn qnindex_rejects_duplicates() {
    QNIndex::new(vec![(U1Sector(1), 2), (U1Sector(1), 3)], Direction::Out);
}

#[test]
fn qnindex_single_sector() {
    // Trivial leg (e.g., MPS boundary)
    let idx = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    assert_eq!(idx.num_blocks(), 1);
    assert_eq!(idx.total_dim(), 1);
}

#[test]
fn qnindex_z2() {
    let idx = QNIndex::new(
        vec![(Z2Sector::new(1), 4), (Z2Sector::new(0), 3)],
        Direction::Out,
    );
    // Should be sorted: Z2(0) < Z2(1)
    assert_eq!(*idx.sector(0), Z2Sector::new(0));
    assert_eq!(idx.block_dim(0), 3);
    assert_eq!(*idx.sector(1), Z2Sector::new(1));
    assert_eq!(idx.block_dim(1), 4);
}

// ---------------------------------------------------------------------------
// BlockCoord
// ---------------------------------------------------------------------------

#[test]
fn block_coord_ord() {
    let a = BlockCoord(vec![0, 1]);
    let b = BlockCoord(vec![1, 0]);
    assert!(a < b);
}

#[test]
fn block_coord_eq_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(BlockCoord(vec![0, 1]));
    assert!(set.contains(&BlockCoord(vec![0, 1])));
    assert!(!set.contains(&BlockCoord(vec![1, 0])));
}
