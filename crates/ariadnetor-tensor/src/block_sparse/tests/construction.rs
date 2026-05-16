use std::sync::Arc;

use arnet_core::backend::MemoryOrder;
use rand::SeedableRng;

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
fn zeros_block_layout() {
    // Verify zeros produces correct block ordering and contiguous offsets
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

    let zeros = BlockSparse::<f64, U1Sector>::zeros(vec![row.clone(), col.clone()], U1Sector(1));
    let rand_bs = BlockSparse::<f64, U1Sector>::random(vec![row, col], U1Sector(1), &mut rng);

    assert_eq!(rand_bs.shape(), zeros.shape());
    assert_eq!(rand_bs.num_blocks(), zeros.num_blocks());
    assert_eq!(rand_bs.stored_len(), zeros.stored_len());
    assert_eq!(rand_bs.flux(), zeros.flux());
    assert_eq!(rand_bs.indices().len(), zeros.indices().len());
}

#[test]
fn random_reproducible() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let mut rng1 = rand::rngs::StdRng::seed_from_u64(123);
    let bs1 = BlockSparse::<f64, U1Sector>::random(
        vec![row.clone(), col.clone()],
        U1Sector(0),
        &mut rng1,
    );

    let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);
    let bs2 = BlockSparse::<f64, U1Sector>::random(vec![row, col], U1Sector(0), &mut rng2);

    for meta in bs1.block_metas() {
        let d1 = bs1.block_data(&meta.coord).unwrap();
        let d2 = bs2.block_data(&meta.coord).unwrap();
        assert_eq!(d1, d2);
    }
}

#[test]
fn zeros_default_order_is_column_major() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let bs: BlockSparse<f64, U1Sector> = BlockSparse::zeros(vec![row, col], U1Sector(0));
    assert_eq!(bs.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn random_default_order_is_column_major() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let bs: BlockSparse<f64, U1Sector> = BlockSparse::random(vec![row, col], U1Sector(0), &mut rng);
    assert_eq!(bs.order(), MemoryOrder::ColumnMajor);
}

// ---------------------------------------------------------------------------
// from_block_fn constructor
// ---------------------------------------------------------------------------

#[test]
fn from_block_fn_basic_u1() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let bs = BlockSparse::<f64, U1Sector>::from_block_fn(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
        |coord, shape| {
            let size: usize = shape.iter().product();
            let base = (coord.0[0] * 100 + coord.0[1]) as f64;
            (0..size).map(|i| base + i as f64).collect()
        },
    );

    // (0,0) block: size 4, values [0, 1, 2, 3]
    let d00 = bs.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(d00, &[0.0, 1.0, 2.0, 3.0]);

    // (1,1) block: size 9, base = 101, values [101..110)
    let d11 = bs.block_data(&BlockCoord(vec![1, 1])).unwrap();
    let expected: Vec<f64> = (0..9).map(|i| 101.0 + i as f64).collect();
    assert_eq!(d11, expected.as_slice());

    // Flux-forbidden coords are absent.
    assert!(bs.block_data(&BlockCoord(vec![0, 1])).is_none());
    assert!(bs.block_data(&BlockCoord(vec![1, 0])).is_none());
}

#[test]
fn from_block_fn_structure_matches_zeros() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let flux = U1Sector(0);

    let z = BlockSparse::<f64, U1Sector>::zeros(vec![row.clone(), col.clone()], flux);
    let f = BlockSparse::<f64, U1Sector>::from_block_fn(
        vec![row, col],
        flux,
        MemoryOrder::ColumnMajor,
        |_, shape| vec![0.0; shape.iter().product()],
    );

    assert_eq!(f.shape(), z.shape());
    assert_eq!(f.num_blocks(), z.num_blocks());
    assert_eq!(f.stored_len(), z.stored_len());
    let f_metas = f.block_metas();
    let z_metas = z.block_metas();
    for (fm, zm) in f_metas.iter().zip(z_metas.iter()) {
        assert_eq!(fm.coord, zm.coord);
        assert_eq!(fm.offset, zm.offset);
        assert_eq!(fm.size, zm.size);
    }
}

#[test]
fn from_block_fn_honors_source_order() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let bs_rm = BlockSparse::<f64, U1Sector>::from_block_fn(
        vec![row.clone(), col.clone()],
        U1Sector(0),
        MemoryOrder::RowMajor,
        |_, _| vec![1.0, 2.0, 3.0, 4.0],
    );
    assert_eq!(bs_rm.order(), MemoryOrder::RowMajor);
    assert_eq!(
        bs_rm.block_data(&BlockCoord(vec![0, 0])).unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );

    let bs_cm = BlockSparse::<f64, U1Sector>::from_block_fn(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
        |_, _| vec![1.0, 2.0, 3.0, 4.0],
    );
    assert_eq!(bs_cm.order(), MemoryOrder::ColumnMajor);
    // Bytes are stored verbatim under either tag — no internal reorder.
    assert_eq!(
        bs_cm.block_data(&BlockCoord(vec![0, 0])).unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn from_block_fn_empty_blocks() {
    // flux mismatch: no allowed blocks; closure must not be invoked.
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);

    let mut calls = 0;
    let bs = BlockSparse::<f64, U1Sector>::from_block_fn(
        vec![row, col],
        U1Sector(1),
        MemoryOrder::ColumnMajor,
        |_, _| {
            calls += 1;
            vec![]
        },
    );
    assert_eq!(calls, 0);
    assert_eq!(bs.num_blocks(), 0);
    assert_eq!(bs.stored_len(), 0);
    assert_eq!(bs.shape(), &[2, 3]);
}

#[test]
fn from_block_fn_rank0_identity_flux() {
    let mut received: Option<(BlockCoord, Vec<usize>)> = None;
    let bs = BlockSparse::<f64, U1Sector>::from_block_fn(
        vec![],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
        |coord, shape| {
            received = Some((coord.clone(), shape.to_vec()));
            vec![42.0]
        },
    );
    assert_eq!(bs.num_blocks(), 1);
    assert_eq!(bs.stored_len(), 1);
    let (coord, shape) = received.expect("closure invoked");
    assert_eq!(coord, BlockCoord(vec![]));
    assert_eq!(shape, Vec::<usize>::new());
    assert_eq!(bs.block_data(&BlockCoord(vec![])).unwrap(), &[42.0]);
}

#[test]
fn from_block_fn_passes_block_shape() {
    let leg0 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 4)], Direction::Out);
    let leg1 = QNIndex::new(vec![(U1Sector(0), 3), (U1Sector(1), 5)], Direction::In);

    let mut shapes_seen: Vec<(BlockCoord, Vec<usize>)> = Vec::new();
    let _bs = BlockSparse::<f64, U1Sector>::from_block_fn(
        vec![leg0, leg1],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
        |coord, shape| {
            shapes_seen.push((coord.clone(), shape.to_vec()));
            vec![0.0; shape.iter().product()]
        },
    );

    // Allowed: (0,0) shape=[2,3]; (1,1) shape=[4,5]. Lex order.
    assert_eq!(
        shapes_seen,
        vec![
            (BlockCoord(vec![0, 0]), vec![2, 3]),
            (BlockCoord(vec![1, 1]), vec![4, 5]),
        ]
    );
}

#[test]
fn from_block_fn_visits_blocks_in_lex_order() {
    let leg0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let leg1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);

    let mut visit: Vec<BlockCoord> = Vec::new();
    let _bs = BlockSparse::<f64, U1Sector>::from_block_fn(
        vec![leg0, leg1],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
        |coord, shape| {
            visit.push(coord.clone());
            vec![0.0; shape.iter().product()]
        },
    );
    assert_eq!(visit, vec![BlockCoord(vec![0, 0]), BlockCoord(vec![1, 1])]);
}

#[test]
#[should_panic(expected = "from_block_fn: closure returned")]
fn from_block_fn_wrong_size_panics() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let _bs = BlockSparse::<f64, U1Sector>::from_block_fn(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
        |_, _| vec![1.0, 2.0, 3.0], // wrong size: expected 4
    );
}

#[test]
fn from_block_fn_z2_sector() {
    let row = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)],
        Direction::Out,
    );
    let col = QNIndex::new(
        vec![(Z2Sector::new(0), 4), (Z2Sector::new(1), 5)],
        Direction::In,
    );
    let bs = BlockSparse::<f64, Z2Sector>::from_block_fn(
        vec![row, col],
        Z2Sector::new(0),
        MemoryOrder::ColumnMajor,
        |_, shape| vec![7.0; shape.iter().product()],
    );
    // Allowed: (0,0) size 2*4=8; (1,1) size 3*5=15.
    assert_eq!(bs.num_blocks(), 2);
    assert_eq!(bs.stored_len(), 23);
    assert!(
        bs.block_data(&BlockCoord(vec![0, 0]))
            .unwrap()
            .iter()
            .all(|&v| v == 7.0)
    );
}

#[test]
fn from_block_fn_preserves_order_through_clone() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let bs = BlockSparse::<f64, U1Sector>::from_block_fn(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
        |_, _| vec![1.0, 2.0, 3.0, 4.0],
    );
    let bs_clone = bs.clone();
    assert_eq!(bs_clone.order(), MemoryOrder::RowMajor);
}

#[test]
fn random_data_is_nonzero() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let row = QNIndex::new(vec![(U1Sector(0), 4), (U1Sector(1), 4)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 4), (U1Sector(1), 4)], Direction::In);

    let bs = BlockSparse::<f64, U1Sector>::random(vec![row, col], U1Sector(0), &mut rng);

    // With 32 random f64 values, probability of all zero is negligible
    let has_nonzero = bs.block_metas().iter().any(|meta| {
        bs.block_data(&meta.coord)
            .unwrap()
            .iter()
            .any(|&v| v != 0.0)
    });
    assert!(has_nonzero);
}
