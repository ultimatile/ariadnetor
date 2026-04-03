use std::sync::Arc;

use crate::block_sparse::*;
use crate::repr::TensorRepr;
use crate::sector::{U1Sector, Z2Sector};

// ---------------------------------------------------------------------------
// TensorRepr
// ---------------------------------------------------------------------------

#[test]
fn tensor_repr_block_sparse() {
    let bs = super::sample_u1_rank2();
    assert_eq!(TensorRepr::shape(&bs), &[5, 5]);
    assert_eq!(bs.rank(), 2);
    // len() is logical (dense) size: 5 * 5 = 25
    assert_eq!(bs.len(), 25);
    assert!(!bs.is_empty());
}

#[test]
fn tensor_repr_empty_block_sparse() {
    // No allowed blocks (flux mismatch), but logical shape is non-zero
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 3)], Direction::In);
    let bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![row, col], U1Sector(1));
    assert_eq!(bs.len(), 6); // 2 * 3 = 6 logical
    assert_eq!(bs.stored_len(), 0);
    assert!(!bs.is_empty()); // logical size is non-zero
}

// ---------------------------------------------------------------------------
// zeros constructor
// ---------------------------------------------------------------------------

#[test]
fn zeros_u1_identity_flux() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![row, col], U1Sector(0));

    // Allowed: (0,0) size 4, (1,1) size 9
    assert_eq!(bs.num_blocks(), 2);
    assert_eq!(bs.stored_len(), 13);
    assert_eq!(bs.shape(), &[5, 5]);

    // Data is zero-filled
    let d00 = bs.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert!(d00.iter().all(|&v| v == 0.0));
    let d11 = bs.block_data(&BlockCoord(vec![1, 1])).unwrap();
    assert!(d11.iter().all(|&v| v == 0.0));

    // Non-allowed blocks are absent
    assert!(bs.block_data(&BlockCoord(vec![0, 1])).is_none());
    assert!(bs.block_data(&BlockCoord(vec![1, 0])).is_none());
}

#[test]
fn zeros_u1_nonzero_flux() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 4)], Direction::In);

    let bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![row, col], U1Sector(1));

    // Allowed: (1,0) Out(1) + In(0).dual() = 1 + 0 = 1 = flux  size = 3*4 = 12
    assert_eq!(bs.num_blocks(), 1);
    assert_eq!(bs.stored_len(), 12);
    assert!(bs.block_data(&BlockCoord(vec![1, 0])).is_some());
}

#[test]
fn zeros_z2() {
    let row = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)],
        Direction::Out,
    );
    let col = QNIndex::new(
        vec![(Z2Sector::new(0), 4), (Z2Sector::new(1), 5)],
        Direction::In,
    );

    let bs: BlockSparse<f64, Z2Sector> = BlockSparse::zeros(vec![row, col], Z2Sector::new(0));

    // Z2: dual is identity. Allowed if Out(a) fuse In(b).dual() = a+b mod 2 = 0
    // (0,0): 0+0=0 ✓  size=2*4=8
    // (1,1): 1+1=0 ✓  size=3*5=15
    assert_eq!(bs.num_blocks(), 2);
    assert_eq!(bs.stored_len(), 23);
}

#[test]
fn zeros_rank3() {
    let leg0 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let leg1 = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let leg2 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);

    let bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![leg0, leg1, leg2], U1Sector(0));

    // (0,0,0): 0+0+0 = 0 ✓  size=2*3*2=12
    // (1,0,1): 1+0+(-1) = 0 ✓  size=1*3*1=3
    assert_eq!(bs.num_blocks(), 2);
    assert_eq!(bs.stored_len(), 15);
}

#[test]
fn zeros_rank0_identity_flux() {
    let bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![], U1Sector(0));
    assert_eq!(bs.rank(), 0);
    assert_eq!(bs.shape(), &[] as &[usize]);
    // Single scalar block
    assert_eq!(bs.num_blocks(), 1);
    assert_eq!(bs.stored_len(), 1);
    let d = bs.block_data(&BlockCoord(vec![])).unwrap();
    assert_eq!(d, &[0.0]);
}

#[test]
fn zeros_rank0_nonidentity_flux() {
    let bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![], U1Sector(1));
    assert_eq!(bs.rank(), 0);
    // No block can satisfy non-identity flux with no legs
    assert_eq!(bs.num_blocks(), 0);
    assert_eq!(bs.stored_len(), 0);
}

#[test]
fn zeros_no_allowed_blocks() {
    // All sectors are charge 0/Out, flux = 1 → no blocks
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);

    let bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![row, col], U1Sector(1));
    assert_eq!(bs.num_blocks(), 0);
    assert_eq!(bs.stored_len(), 0);
    assert_eq!(bs.shape(), &[2, 3]);
}

#[test]
fn zeros_matches_from_raw_parts() {
    // Verify zeros produces the same structure as manually built from_raw_parts
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let bs = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));

    assert_eq!(bs.num_blocks(), 2);
    // Blocks should be in lexicographic coord order
    let metas = bs.block_metas();
    assert_eq!(metas[0].coord, BlockCoord(vec![0, 0]));
    assert_eq!(metas[0].size, 4);
    assert_eq!(metas[1].coord, BlockCoord(vec![1, 1]));
    assert_eq!(metas[1].size, 9);
    // Offsets are contiguous
    assert_eq!(metas[0].offset, 0);
    assert_eq!(metas[1].offset, 4);
}

// ---------------------------------------------------------------------------
// block_data_mut
// ---------------------------------------------------------------------------

#[test]
fn block_data_mut_fills_block() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let mut bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![row, col], U1Sector(0));

    // Fill block (0,0)
    let d = bs.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    for (i, v) in d.iter_mut().enumerate() {
        *v = (i + 1) as f64;
    }

    let d = bs.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(d, &[1.0, 2.0, 3.0, 4.0]);

    // Other block unchanged
    let d11 = bs.block_data(&BlockCoord(vec![1, 1])).unwrap();
    assert!(d11.iter().all(|&v| v == 0.0));
}

#[test]
fn block_data_mut_nonexistent_returns_none() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let mut bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![row, col], U1Sector(0));
    assert!(bs.block_data_mut(&BlockCoord(vec![0, 1])).is_none());
}

#[test]
fn block_data_mut_cow_semantics() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let mut bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![row, col], U1Sector(0));
    let cloned = bs.clone();

    // Both share the same Arc
    assert!(Arc::ptr_eq(&bs.data, &cloned.data));

    // Mutation triggers CoW — bs gets its own copy
    let d = bs.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    d[0] = 42.0;

    assert!(!Arc::ptr_eq(&bs.data, &cloned.data));
    assert_eq!(bs.block_data(&BlockCoord(vec![0, 0])).unwrap()[0], 42.0);
    assert_eq!(cloned.block_data(&BlockCoord(vec![0, 0])).unwrap()[0], 0.0);
}
