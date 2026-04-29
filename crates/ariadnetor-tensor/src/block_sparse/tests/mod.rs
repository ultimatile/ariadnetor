mod construction;

use std::sync::Arc;

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

// ---------------------------------------------------------------------------
// BlockSparse
// ---------------------------------------------------------------------------

/// Helper: build a simple rank-2 BlockSparse with U(1) symmetry.
///
/// Indices: row = {charge 0: dim 2, charge 1: dim 3}, Out
///          col = {charge 0: dim 2, charge 1: dim 3}, In
/// Flux = identity (charge 0)
///
/// Allowed blocks: (0,0) → 2×2=4 elems, (1,1) → 3×3=9 elems
fn sample_u1_rank2() -> BlockSparse<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    // Block (0,0): Out(0) fuse In(0).dual() = 0 + 0 = 0 == flux ✓
    // Block (1,1): Out(1) fuse In(1).dual() = 1 + (-1) = 0 == flux ✓

    let data_00: Vec<f64> = (1..=4).map(|x| x as f64).collect();
    let data_11: Vec<f64> = (5..=13).map(|x| x as f64).collect();

    let mut data = Vec::with_capacity(13);
    data.extend_from_slice(&data_00);
    data.extend_from_slice(&data_11);

    let blocks = vec![
        BlockMeta {
            coord: BlockCoord(vec![0, 0]),
            offset: 0,
            size: 4,
        },
        BlockMeta {
            coord: BlockCoord(vec![1, 1]),
            offset: 4,
            size: 9,
        },
    ];

    BlockSparse::from_raw_parts(data, blocks, vec![row, col], U1Sector(0))
}

#[test]
fn block_sparse_basic_accessors() {
    let bs = sample_u1_rank2();
    assert_eq!(bs.rank(), 2);
    assert_eq!(bs.shape(), &[5, 5]);
    assert_eq!(bs.num_blocks(), 2);
    assert_eq!(bs.stored_len(), 13);
    assert_eq!(*bs.flux(), U1Sector(0));
}

#[test]
fn block_sparse_block_data() {
    let bs = sample_u1_rank2();

    let d00 = bs.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(d00, &[1.0, 2.0, 3.0, 4.0]);

    let d11 = bs.block_data(&BlockCoord(vec![1, 1])).unwrap();
    assert_eq!(d11, &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);

    // Non-existent block (forbidden by symmetry)
    assert!(bs.block_data(&BlockCoord(vec![0, 1])).is_none());
    assert!(bs.block_data(&BlockCoord(vec![1, 0])).is_none());
}

#[test]
fn block_sparse_block_shape() {
    let bs = sample_u1_rank2();
    assert_eq!(bs.block_shape(&BlockCoord(vec![0, 0])), Some(vec![2, 2]));
    assert_eq!(bs.block_shape(&BlockCoord(vec![1, 1])), Some(vec![3, 3]));
    assert_eq!(bs.block_shape(&BlockCoord(vec![0, 1])), Some(vec![2, 3]));

    // Invalid coord (out of range)
    assert_eq!(bs.block_shape(&BlockCoord(vec![2, 0])), None);
    // Wrong rank
    assert_eq!(bs.block_shape(&BlockCoord(vec![0])), None);
}

#[test]
fn block_sparse_is_allowed_block() {
    let bs = sample_u1_rank2();
    assert!(bs.is_allowed_block(&BlockCoord(vec![0, 0])));
    assert!(bs.is_allowed_block(&BlockCoord(vec![1, 1])));
    assert!(!bs.is_allowed_block(&BlockCoord(vec![0, 1])));
    assert!(!bs.is_allowed_block(&BlockCoord(vec![1, 0])));
}

#[test]
fn block_sparse_clone_shares_data() {
    let bs = sample_u1_rank2();
    let cloned = bs.clone();
    // Arc-shared data: same pointer
    assert!(Arc::ptr_eq(&bs.data, &cloned.data));
}

#[test]
fn block_sparse_nonzero_flux() {
    // Tensor with flux = U1(1)
    // Block (1,0): Out(1) + In(0).dual() = 1 + 0 = 1 == flux ✓
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let data: Vec<f64> = (1..=6).map(|x| x as f64).collect();
    let blocks = vec![BlockMeta {
        coord: BlockCoord(vec![1, 0]),
        offset: 0,
        size: 6,
    }];

    let bs = BlockSparse::from_raw_parts(data, blocks, vec![row, col], U1Sector(1));
    assert_eq!(bs.num_blocks(), 1);
    assert_eq!(*bs.flux(), U1Sector(1));
    assert!(bs.is_allowed_block(&BlockCoord(vec![1, 0])));
    assert!(!bs.is_allowed_block(&BlockCoord(vec![0, 0])));
}

#[test]
fn block_sparse_z2_symmetry() {
    let row = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)],
        Direction::Out,
    );
    let col = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)],
        Direction::In,
    );

    // Z2 is self-dual: allowed blocks (0,0) and (1,1) both fuse to 0
    let data = vec![0.0_f64; 4 + 9];
    let blocks = vec![
        BlockMeta {
            coord: BlockCoord(vec![0, 0]),
            offset: 0,
            size: 4,
        },
        BlockMeta {
            coord: BlockCoord(vec![1, 1]),
            offset: 4,
            size: 9,
        },
    ];

    let bs = BlockSparse::from_raw_parts(data, blocks, vec![row, col], Z2Sector::new(0));
    assert_eq!(bs.num_blocks(), 2);
    assert!(bs.is_allowed_block(&BlockCoord(vec![0, 0])));
    assert!(bs.is_allowed_block(&BlockCoord(vec![1, 1])));
    assert!(!bs.is_allowed_block(&BlockCoord(vec![0, 1])));
}

#[test]
#[should_panic(expected = "violates flux conservation")]
fn block_sparse_rejects_invalid_flux() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let data = vec![0.0_f64; 4];
    let blocks = vec![BlockMeta {
        coord: BlockCoord(vec![0, 0]),
        offset: 0,
        size: 4,
    }];

    BlockSparse::from_raw_parts(data, blocks, vec![row, col], U1Sector(1));
}

#[test]
#[should_panic(expected = "Duplicate block coordinate")]
fn block_sparse_rejects_duplicate_coords() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let data = vec![0.0_f64; 8];
    let blocks = vec![
        BlockMeta {
            coord: BlockCoord(vec![0, 0]),
            offset: 0,
            size: 4,
        },
        BlockMeta {
            coord: BlockCoord(vec![0, 0]),
            offset: 4,
            size: 4,
        },
    ];

    BlockSparse::from_raw_parts(data, blocks, vec![row, col], U1Sector(0));
}

#[test]
#[should_panic(expected = "size mismatch")]
fn block_sparse_rejects_wrong_size() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let data = vec![0.0_f64; 5];
    let blocks = vec![BlockMeta {
        coord: BlockCoord(vec![0, 0]),
        offset: 0,
        size: 5, // should be 4 (2×2)
    }];

    BlockSparse::from_raw_parts(data, blocks, vec![row, col], U1Sector(0));
}

#[test]
fn block_sparse_empty() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let bs: BlockSparse<f64, U1Sector> =
        BlockSparse::from_raw_parts(vec![], vec![], vec![row, col], U1Sector(1));
    assert_eq!(bs.num_blocks(), 0);
    assert_eq!(bs.stored_len(), 0);
    assert_eq!(bs.shape(), &[2, 2]);
}

#[test]
fn block_sparse_rank3() {
    // Rank-3: flux = 0
    // (0,0,0): Out(0) + Out(0) + In(0).dual() = 0  size = 2*3*2 = 12
    // (1,0,1): Out(1) + Out(0) + In(1).dual() = 1+0+(-1) = 0  size = 1*3*1 = 3
    let leg0 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let leg1 = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let leg2 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);

    let data = vec![0.0_f64; 12 + 3];
    let blocks = vec![
        BlockMeta {
            coord: BlockCoord(vec![0, 0, 0]),
            offset: 0,
            size: 12,
        },
        BlockMeta {
            coord: BlockCoord(vec![1, 0, 1]),
            offset: 12,
            size: 3,
        },
    ];

    let bs = BlockSparse::from_raw_parts(data, blocks, vec![leg0, leg1, leg2], U1Sector(0));
    assert_eq!(bs.rank(), 3);
    assert_eq!(bs.shape(), &[3, 3, 3]);
    assert_eq!(bs.num_blocks(), 2);
    assert_eq!(bs.stored_len(), 15);
}

#[test]
fn block_sparse_tuple_symmetry() {
    // U(1) × Z2 direct-product symmetry
    type Sym = (U1Sector, Z2Sector);
    let s00 = (U1Sector(0), Z2Sector::new(0));
    let s11 = (U1Sector(1), Z2Sector::new(1));

    let row = QNIndex::new(vec![(s00, 2), (s11, 3)], Direction::Out);
    let col = QNIndex::new(vec![(s00, 2), (s11, 3)], Direction::In);

    // flux = identity = (0, 0)
    // Block (0,0): Out(0,0) fuse In(0,0).dual() = (0,0) ✓
    // Block (1,1): Out(1,1) fuse In(1,1).dual() = (1,1).fuse((-1,1)) = (0,0) ✓

    let data = vec![0.0_f64; 4 + 9];
    let blocks = vec![
        BlockMeta {
            coord: BlockCoord(vec![0, 0]),
            offset: 0,
            size: 4,
        },
        BlockMeta {
            coord: BlockCoord(vec![1, 1]),
            offset: 4,
            size: 9,
        },
    ];

    let flux: Sym = Sector::identity();
    let bs: BlockSparse<f64, Sym> = BlockSparse::from_raw_parts(data, blocks, vec![row, col], flux);
    assert_eq!(bs.num_blocks(), 2);
}

#[test]
#[should_panic(expected = "gap or overlap")]
fn block_sparse_rejects_overlapping_blocks() {
    // Two valid blocks but with overlapping offsets
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let data = vec![0.0_f64; 9]; // only 9 elements, not 13
    let blocks = vec![
        BlockMeta {
            coord: BlockCoord(vec![0, 0]),
            offset: 0,
            size: 4,
        },
        BlockMeta {
            coord: BlockCoord(vec![1, 1]),
            offset: 0, // overlaps with block (0,0)
            size: 9,
        },
    ];

    BlockSparse::from_raw_parts(data, blocks, vec![row, col], U1Sector(0));
}

#[test]
#[should_panic(expected = "Data buffer has")]
fn block_sparse_rejects_trailing_padding() {
    // Data buffer larger than blocks require
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let data = vec![0.0_f64; 8]; // 8 elements but block only needs 4
    let blocks = vec![BlockMeta {
        coord: BlockCoord(vec![0, 0]),
        offset: 0,
        size: 4,
    }];

    BlockSparse::from_raw_parts(data, blocks, vec![row, col], U1Sector(0));
}
