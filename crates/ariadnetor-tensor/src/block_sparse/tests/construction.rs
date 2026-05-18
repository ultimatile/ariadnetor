//! Tests for `BlockSparseTensorData` constructors (`zeros`, `random`)
//! and `block_data_mut`.

use arnet_core::backend::MemoryOrder;
use rand::SeedableRng;

use crate::block_sparse::*;
use crate::sector::{U1Sector, Z2Sector};

// ---------------------------------------------------------------------------
// Logical-vs-stored extent
// ---------------------------------------------------------------------------

#[test]
fn logical_vs_stored_extent_for_empty_block_sparse() {
    // No allowed blocks (flux mismatch), but logical shape is non-zero.
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 3)], Direction::In);
    let td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![row, col], U1Sector(1), MemoryOrder::RowMajor);
    let logical_len: usize = td.shape().iter().product();
    assert_eq!(logical_len, 6); // 2 * 3
    assert_eq!(td.storage().stored_len(), 0);
}

// ---------------------------------------------------------------------------
// zeros constructor
// ---------------------------------------------------------------------------

#[test]
fn zeros_u1_identity_flux() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![row, col], U1Sector(0), MemoryOrder::RowMajor);

    // Allowed: (0,0) size 4, (1,1) size 9
    assert_eq!(td.num_blocks(), 2);
    assert_eq!(td.storage().stored_len(), 13);
    assert_eq!(td.shape(), vec![5, 5]);

    // Data is zero-filled
    let d00 = td.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert!(d00.iter().all(|&v| v == 0.0));
    let d11 = td.block_data(&BlockCoord(vec![1, 1])).unwrap();
    assert!(d11.iter().all(|&v| v == 0.0));

    // Non-allowed blocks are absent
    assert!(td.block_data(&BlockCoord(vec![0, 1])).is_none());
    assert!(td.block_data(&BlockCoord(vec![1, 0])).is_none());
}

#[test]
fn zeros_u1_nonzero_flux() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 4)], Direction::In);

    let td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![row, col], U1Sector(1), MemoryOrder::RowMajor);

    // Allowed: (1,0) Out(1) + In(0).dual() = 1 + 0 = 1 = flux  size = 3*4 = 12
    assert_eq!(td.num_blocks(), 1);
    assert_eq!(td.storage().stored_len(), 12);
    assert!(td.block_data(&BlockCoord(vec![1, 0])).is_some());
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

    let td: BlockSparseTensorData<f64, Z2Sector> =
        BlockSparseTensorData::zeros(vec![row, col], Z2Sector::new(0), MemoryOrder::RowMajor);

    // Z2: dual is identity. Allowed if Out(a) fuse In(b).dual() = a+b mod 2 = 0
    // (0,0): 0+0=0 ✓  size=2*4=8
    // (1,1): 1+1=0 ✓  size=3*5=15
    assert_eq!(td.num_blocks(), 2);
    assert_eq!(td.storage().stored_len(), 23);
}

#[test]
fn zeros_rank3() {
    let leg0 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let leg1 = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let leg2 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);

    let td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![leg0, leg1, leg2], U1Sector(0), MemoryOrder::RowMajor);

    // (0,0,0): 0+0+0 = 0 ✓  size=2*3*2=12
    // (1,0,1): 1+0+(-1) = 0 ✓  size=1*3*1=3
    assert_eq!(td.num_blocks(), 2);
    assert_eq!(td.storage().stored_len(), 15);
}

#[test]
fn zeros_rank0_identity_flux() {
    let td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![], U1Sector(0), MemoryOrder::RowMajor);
    assert_eq!(td.rank(), 0);
    assert_eq!(td.shape(), Vec::<usize>::new());
    // Single scalar block
    assert_eq!(td.num_blocks(), 1);
    assert_eq!(td.storage().stored_len(), 1);
    let d = td.block_data(&BlockCoord(vec![])).unwrap();
    assert_eq!(d, &[0.0]);
}

#[test]
fn zeros_rank0_nonidentity_flux() {
    let td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![], U1Sector(1), MemoryOrder::RowMajor);
    assert_eq!(td.rank(), 0);
    // No block can satisfy non-identity flux with no legs
    assert_eq!(td.num_blocks(), 0);
    assert_eq!(td.storage().stored_len(), 0);
}

#[test]
fn zeros_no_allowed_blocks() {
    // All sectors are charge 0/Out, flux = 1 → no blocks
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);

    let td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![row, col], U1Sector(1), MemoryOrder::RowMajor);
    assert_eq!(td.num_blocks(), 0);
    assert_eq!(td.storage().stored_len(), 0);
    assert_eq!(td.shape(), vec![2, 3]);
}

#[test]
fn zeros_block_layout() {
    // Verify zeros produces correct block ordering and contiguous offsets
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let td = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
    );

    assert_eq!(td.num_blocks(), 2);
    // Blocks should be in lexicographic coord order
    let metas = td.block_metas();
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

    let mut td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![row, col], U1Sector(0), MemoryOrder::RowMajor);

    // Fill block (0,0)
    let d = td.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    for (i, v) in d.iter_mut().enumerate() {
        *v = (i + 1) as f64;
    }

    let d = td.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(d, &[1.0, 2.0, 3.0, 4.0]);

    // Other block unchanged
    let d11 = td.block_data(&BlockCoord(vec![1, 1])).unwrap();
    assert!(d11.iter().all(|&v| v == 0.0));
}

#[test]
fn block_data_mut_nonexistent_returns_none() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let mut td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![row, col], U1Sector(0), MemoryOrder::RowMajor);
    assert!(td.block_data_mut(&BlockCoord(vec![0, 1])).is_none());
}

#[test]
fn block_data_mut_cow_semantics() {
    // CoW: after cloning, mutating one side must not affect the other.
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let mut td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![row, col], U1Sector(0), MemoryOrder::RowMajor);
    let cloned = td.clone();

    // Mutation triggers CoW on the storage half — td gets its own copy.
    let d = td.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    d[0] = 42.0;

    assert_eq!(td.block_data(&BlockCoord(vec![0, 0])).unwrap()[0], 42.0);
    assert_eq!(cloned.block_data(&BlockCoord(vec![0, 0])).unwrap()[0], 0.0);
}

// ---------------------------------------------------------------------------
// random constructor
// ---------------------------------------------------------------------------

#[test]
fn random_matches_zeros_structure() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let row = QNIndex::new(
        vec![(U1Sector(0), 2), (U1Sector(1), 3), (U1Sector(2), 4)],
        Direction::Out,
    );
    let col = QNIndex::new(vec![(U1Sector(0), 5), (U1Sector(1), 2)], Direction::In);

    let zeros = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row.clone(), col.clone()],
        U1Sector(1),
        MemoryOrder::RowMajor,
    );
    let rand_td = BlockSparseTensorData::<f64, U1Sector>::random(
        vec![row, col],
        U1Sector(1),
        MemoryOrder::RowMajor,
        &mut rng,
    );

    assert_eq!(rand_td.shape(), zeros.shape());
    assert_eq!(rand_td.num_blocks(), zeros.num_blocks());
    assert_eq!(rand_td.storage().stored_len(), zeros.storage().stored_len());
    assert_eq!(rand_td.flux(), zeros.flux());
    assert_eq!(rand_td.indices().len(), zeros.indices().len());
}

#[test]
fn random_reproducible() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let mut rng1 = rand::rngs::StdRng::seed_from_u64(123);
    let td1 = BlockSparseTensorData::<f64, U1Sector>::random(
        vec![row.clone(), col.clone()],
        U1Sector(0),
        MemoryOrder::RowMajor,
        &mut rng1,
    );

    let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);
    let td2 = BlockSparseTensorData::<f64, U1Sector>::random(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
        &mut rng2,
    );

    for meta in td1.block_metas() {
        let d1 = td1.block_data(&meta.coord).unwrap();
        let d2 = td2.block_data(&meta.coord).unwrap();
        assert_eq!(d1, d2);
    }
}

#[test]
fn random_data_is_nonzero() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let row = QNIndex::new(vec![(U1Sector(0), 4), (U1Sector(1), 4)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 4), (U1Sector(1), 4)], Direction::In);

    let td = BlockSparseTensorData::<f64, U1Sector>::random(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
        &mut rng,
    );

    // With 32 random f64 values, probability of all zero is negligible
    let has_nonzero = td.block_metas().iter().any(|meta| {
        td.block_data(&meta.coord)
            .unwrap()
            .iter()
            .any(|&v| v != 0.0)
    });
    assert!(has_nonzero);
}
