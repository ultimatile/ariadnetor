use arnet_native::NativeBackend;
use arnet_tensor::{BlockCoord, BlockSparse, Direction, QNIndex, U1Sector, Z2Sector};

use crate::contract_block_sparse;
use crate::fuse_legs_block_sparse;
use crate::permute_block_sparse;

fn backend() -> NativeBackend {
    NativeBackend
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a rank-2 U1, flux=0. Out(0:2, 1:3), In(0:2, 1:3).
fn sample_u1_rank2() -> BlockSparse<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut bs = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));
    let d = bs.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    d.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let d = bs.block_data_mut(&BlockCoord(vec![1, 1])).unwrap();
    d.copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    bs
}

/// Build a rank-3 U1, flux=0. Out(0:2, 1:3), Out(0:2, 1:1), In(0:2, 1:3).
fn sample_u1_rank3() -> BlockSparse<f64, U1Sector> {
    let i0 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let i1 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let i2 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut bs = BlockSparse::zeros(vec![i0, i1, i2], U1Sector(0));
    let mut val = 1.0;
    for meta in bs.block_metas().to_vec() {
        let data = bs.block_data_mut(&meta.coord).unwrap();
        for elem in data.iter_mut() {
            *elem = val;
            val += 1.0;
        }
    }
    bs
}

// ---------------------------------------------------------------------------
// Basic: fuse rank-2 → rank-1
// ---------------------------------------------------------------------------

#[test]
fn fuse_rank2_to_rank1() {
    let bs = sample_u1_rank2();
    let fused = fuse_legs_block_sparse(&backend(), &bs, 0, 2, Direction::Out).unwrap();

    assert_eq!(fused.rank(), 1);
    // Fused dimension = sum of all block dims for flux-allowed (i,j) pairs
    // Pairs: (0,0) → 2*2=4, (1,1) → 3*3=9
    // Fused sectors: directed (Out(0), In(0)) → fuse(0, 0)=0; directed (Out(1), In(1)) → fuse(1, -1)=0
    // Both map to sector 0 (Out), total dim = 4 + 9 = 13
    assert_eq!(fused.shape(), &[13]);

    // Total stored data should be preserved
    let orig_elems: usize = bs.block_metas().iter().map(|m| m.size).sum();
    let fused_elems: usize = fused.block_metas().iter().map(|m| m.size).sum();
    assert_eq!(fused_elems, orig_elems);
}

// ---------------------------------------------------------------------------
// Fuse leading axes of rank-3
// ---------------------------------------------------------------------------

#[test]
fn fuse_leading_axes_rank3() {
    let bs = sample_u1_rank3();
    let fused = fuse_legs_block_sparse(&backend(), &bs, 0, 2, Direction::Out).unwrap();

    assert_eq!(fused.rank(), 2);
    // The output should have [fused_01, original_2] structure
    assert_eq!(fused.indices()[1].direction(), Direction::In);

    // Flux preserved
    assert_eq!(fused.flux(), bs.flux());
    for meta in fused.block_metas() {
        assert!(fused.is_allowed_block(&meta.coord));
    }
}

// ---------------------------------------------------------------------------
// Contract: fused tuple offsets follow lexicographic order
// ---------------------------------------------------------------------------

/// Multiple tuples fuse to the same sector. Verify that fused data within
/// a block follows lexicographic tuple order, not block_metas encounter order.
/// This is a regression test for a bug where tuple offsets depended on
/// block_metas iteration order rather than the canonical lexicographic order.
#[test]
fn fuse_tuple_offset_is_lexicographic() {
    // Rank-3 tensor with axes Out(0:1, 1:1), Out(0:1, 1:1), In(0:1).
    // Flux = 0. With Out+Out fuse → sector can be 0+0=0 or 1+1=2. Sector 0
    // should have tuples (0,0) before (1,1) would be a different sector.
    // Use a case where multiple tuples map to the SAME fused sector:
    // Out(0:1, 1:1), In(0:1, 1:1), fusing these two with direction Out
    // Tuples: (0,0)→fuse(0,0)=0 dim=1, (0,1)→fuse(0,-1)=-1 dim=1,
    //         (1,0)→fuse(1,0)=1 dim=1, (1,1)→fuse(1,-1)=0 dim=1
    // Sector 0 has tuples (0,0) and (1,1), both dim=1 → fused dim=2
    let i0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let i1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let i2 = QNIndex::new(vec![(U1Sector(0), 3)], Direction::In);
    // flux = 0, so allowed blocks must satisfy Out(s0).fuse(In(s1)).fuse(In(s2)) = 0
    // s2 = 0 always. So Out(s0).fuse(In(s1)) = 0 → s0 - s1 = 0 → s0 == s1
    // Blocks: (0,0,0) and (1,1,0)
    let mut bs = BlockSparse::<f64, U1Sector>::zeros(vec![i0, i1, i2], U1Sector(0));
    // Block (0,0,0): 1×1×3 = 3 elements, fill with [10, 20, 30]
    bs.block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .unwrap()
        .copy_from_slice(&[10.0, 20.0, 30.0]);
    // Block (1,1,0): 1×1×3 = 3 elements, fill with [40, 50, 60]
    bs.block_data_mut(&BlockCoord(vec![1, 1, 0]))
        .unwrap()
        .copy_from_slice(&[40.0, 50.0, 60.0]);

    // Fuse axes (0,1) with Out direction
    let fused = fuse_legs_block_sparse(&backend(), &bs, 0, 2, Direction::Out).unwrap();

    assert_eq!(fused.rank(), 2);
    // Fused sector 0 (Out) has dim=2 (tuple (0,0) dim=1 + tuple (1,1) dim=1)
    // Remaining axis: In(0:3)
    // Output block (sector_0_idx, 0) should have shape [2, 3]

    // Find the block for fused sector 0
    let fused_block = fused.block_metas().iter().find(|m| m.size == 6).unwrap();
    let data = fused.block_data(&fused_block.coord).unwrap();

    // Lexicographic order: tuple (0,0) comes before (1,1).
    // In the NativeBackend (column-major), data layout for [2, 3]:
    // CM: col0=[row0,row1], col1=[row0,row1], col2=[row0,row1]
    // = [10, 40, 20, 50, 30, 60]
    // In RM: [10, 20, 30, 40, 50, 60]
    // Either way, the tuple (0,0) data should come in the fused-dim-0 position
    // and (1,1) in fused-dim-1. Verify the multiset is correct.
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    // Stronger check: verify ordering depends on fused dimension, not encounter order.
    // In CM [2,3]: element (fused_idx, col) at flat = fused_idx + 2*col
    // So fused_idx=0 elements: data[0], data[2], data[4] should be 10, 20, 30
    // fused_idx=1 elements: data[1], data[3], data[5] should be 40, 50, 60
    assert_eq!(data[0], 10.0, "fused_idx=0, col=0");
    assert_eq!(data[2], 20.0, "fused_idx=0, col=1");
    assert_eq!(data[4], 30.0, "fused_idx=0, col=2");
    assert_eq!(data[1], 40.0, "fused_idx=1, col=0");
    assert_eq!(data[3], 50.0, "fused_idx=1, col=1");
    assert_eq!(data[5], 60.0, "fused_idx=1, col=2");
}

// ---------------------------------------------------------------------------
// Fuse trailing axes of rank-3
// ---------------------------------------------------------------------------

#[test]
fn fuse_trailing_axes_rank3() {
    let bs = sample_u1_rank3();
    let fused = fuse_legs_block_sparse(&backend(), &bs, 1, 2, Direction::In).unwrap();

    assert_eq!(fused.rank(), 2);
    assert_eq!(fused.indices()[0].direction(), Direction::Out);
    assert_eq!(fused.indices()[1].direction(), Direction::In);

    // Flux preserved
    assert_eq!(fused.flux(), bs.flux());
    for meta in fused.block_metas() {
        assert!(fused.is_allowed_block(&meta.coord));
    }
}

// ---------------------------------------------------------------------------
// Data correctness: compare with Dense reshape
// ---------------------------------------------------------------------------

/// For a U1 rank-2 tensor with a single non-identity-sector block,
/// fusing all legs and checking data matches Dense reshape.
#[test]
fn fuse_data_matches_dense_reshape_single_sector() {
    // Build a rank-2 tensor with a single sector pair (only 0-sector) for direct comparison
    let row = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 4)], Direction::In);
    let mut bs = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));
    let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    bs.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&data);

    let fused = fuse_legs_block_sparse(&backend(), &bs, 0, 2, Direction::Out).unwrap();

    // Single sector → fused is rank-1 with dim 12
    assert_eq!(fused.shape(), &[12]);
    let fused_data = fused.block_data(&BlockCoord(vec![0])).unwrap();

    // In both RM and CM, fusing all axes of a single block is a no-op on data.
    assert_eq!(fused_data, &data);
}

// ---------------------------------------------------------------------------
// Non-trivial: fuse with In direction (non-self-dual sectors)
// ---------------------------------------------------------------------------

#[test]
fn fuse_with_in_direction_u1() {
    // U1 is non-self-dual: dual(1) = -1. This tests the sector→block_index lookup.
    let bs = sample_u1_rank2();
    let fused_out = fuse_legs_block_sparse(&backend(), &bs, 0, 2, Direction::Out).unwrap();
    let fused_in = fuse_legs_block_sparse(&backend(), &bs, 0, 2, Direction::In).unwrap();

    // Both should have the same shape (same total dims)
    assert_eq!(fused_out.shape(), fused_in.shape());

    // Stored sectors differ but data should be the same
    let out_data = fused_out.block_data(&BlockCoord(vec![0])).unwrap();
    let in_data = fused_in.block_data(&BlockCoord(vec![0])).unwrap();
    assert_eq!(out_data, in_data);

    // Direction differs
    assert_eq!(fused_out.indices()[0].direction(), Direction::Out);
    assert_eq!(fused_in.indices()[0].direction(), Direction::In);
}

// ---------------------------------------------------------------------------
// Apply scenario: rank-5 → permute → fuse → fuse → rank-3
// ---------------------------------------------------------------------------

#[test]
fn apply_scenario_fuse_rank5_to_rank3() {
    // Simulate the apply pipeline:
    // result[w_L, d_bra, w_R, chi_L, chi_R] (rank-5)
    // → permute [0, 3, 1, 2, 4] → [w_L, chi_L, d_bra, w_R, chi_R]
    // → fuse(0,2,Out) → [left, d_bra, w_R, chi_R]
    // → fuse(2,2,In) → [left, d_bra, right]
    let w_l = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let d_bra = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let w_r = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
    let chi_l = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let chi_r = QNIndex::new(vec![(U1Sector(0), 3)], Direction::In);

    let mut bs =
        BlockSparse::<f64, U1Sector>::zeros(vec![w_l, d_bra, w_r, chi_l, chi_r], U1Sector(0));
    let data: Vec<f64> = (1..=18).map(|i| i as f64).collect();
    bs.block_data_mut(&BlockCoord(vec![0, 0, 0, 0, 0]))
        .unwrap()
        .copy_from_slice(&data);

    // Permute: [0, 3, 1, 2, 4]
    let permuted = permute_block_sparse(&backend(), &bs, &[0, 3, 1, 2, 4]).unwrap();
    assert_eq!(permuted.shape(), &[1, 3, 2, 1, 3]);

    // Fuse (0,1) → left bond
    let fused1 = fuse_legs_block_sparse(&backend(), &permuted, 0, 2, Direction::Out).unwrap();
    assert_eq!(fused1.rank(), 4);
    assert_eq!(fused1.shape(), &[3, 2, 1, 3]);

    // Fuse (2,3) → right bond
    let fused2 = fuse_legs_block_sparse(&backend(), &fused1, 2, 2, Direction::In).unwrap();
    assert_eq!(fused2.rank(), 3);
    assert_eq!(fused2.shape(), &[3, 2, 3]);

    // All 18 elements should be preserved (single block)
    let result_data = fused2.block_data(&BlockCoord(vec![0, 0, 0])).unwrap();
    assert_eq!(result_data.len(), 18);
    // Data should be non-trivially rearranged by the permute
    // Just verify all values are present (permutation preserves the multiset)
    let mut sorted_result: Vec<f64> = result_data.to_vec();
    sorted_result.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected: Vec<f64> = (1..=18).map(|i| i as f64).collect();
    assert_eq!(sorted_result, expected);
}

// ---------------------------------------------------------------------------
// Multi-sector apply scenario with contraction
// ---------------------------------------------------------------------------

#[test]
fn apply_scenario_multi_sector() {
    // Contract two block-sparse tensors, then permute+fuse.
    // MPO: W[w_L(Out), d_ket(In), d_bra(Out), w_R(In)]
    // MPS: A[chi_L(Out), d_ket(Out), chi_R(In)]
    let w_l = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let d_ket_w = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let d_bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let w_r = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);

    let chi_l = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let d_ket_a = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let chi_r = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let mut w = BlockSparse::<f64, U1Sector>::zeros(vec![w_l, d_ket_w, d_bra, w_r], U1Sector(0));
    // Fill W with known values
    for meta in w.block_metas().to_vec() {
        let data = w.block_data_mut(&meta.coord).unwrap();
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i + 1) as f64;
        }
    }

    let mut a = BlockSparse::<f64, U1Sector>::zeros(vec![chi_l, d_ket_a, chi_r], U1Sector(0));
    for meta in a.block_metas().to_vec() {
        let data = a.block_data_mut(&meta.coord).unwrap();
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i + 1) as f64 * 0.5;
        }
    }

    // Contract W and A over d_ket: W axis 1, A axis 1
    let contracted = contract_block_sparse(&backend(), &w, &a, &[1], &[1]).unwrap();
    let result = match contracted {
        crate::BlockSparseContractResult::Tensor(t) => t,
        _ => panic!("expected tensor"),
    };
    assert_eq!(result.rank(), 5); // [w_L, d_bra, w_R, chi_L, chi_R]

    // Permute → fuse → fuse
    let permuted = permute_block_sparse(&backend(), &result, &[0, 3, 1, 2, 4]).unwrap();
    let fused1 = fuse_legs_block_sparse(&backend(), &permuted, 0, 2, Direction::Out).unwrap();
    let fused2 = fuse_legs_block_sparse(&backend(), &fused1, 2, 2, Direction::In).unwrap();

    assert_eq!(fused2.rank(), 3);
    // Verify flux conservation
    assert_eq!(fused2.flux(), &U1Sector(0));
    for meta in fused2.block_metas() {
        assert!(fused2.is_allowed_block(&meta.coord));
    }
    // Verify all data is finite
    for meta in fused2.block_metas() {
        let data = fused2.block_data(&meta.coord).unwrap();
        for &v in data {
            assert!(v.is_finite());
        }
    }
}

// ---------------------------------------------------------------------------
// Z2 sector
// ---------------------------------------------------------------------------

#[test]
fn fuse_z2_rank2() {
    let row = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)],
        Direction::Out,
    );
    let col = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)],
        Direction::In,
    );
    let mut bs = BlockSparse::<f64, Z2Sector>::zeros(vec![row, col], Z2Sector::new(0));
    let d = bs.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    d.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let d = bs.block_data_mut(&BlockCoord(vec![1, 1])).unwrap();
    d.copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);

    let fused = fuse_legs_block_sparse(&backend(), &bs, 0, 2, Direction::Out).unwrap();

    // Z2: (0,0)→0, (1,1)→0 (both 0+0=0, 1+1=0 mod 2)
    // Single fused sector 0 with dim 4+9=13
    assert_eq!(fused.rank(), 1);
    assert_eq!(fused.shape(), &[13]);
}

// ---------------------------------------------------------------------------
// Non-trivial flux
// ---------------------------------------------------------------------------

#[test]
fn fuse_nonzero_flux() {
    // Tensor with flux = U1(1)
    let i0 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let i1 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let i2 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut bs = BlockSparse::<f64, U1Sector>::zeros(vec![i0, i1, i2], U1Sector(1));
    for meta in bs.block_metas().to_vec() {
        let data = bs.block_data_mut(&meta.coord).unwrap();
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i + 1) as f64;
        }
    }

    let fused = fuse_legs_block_sparse(&backend(), &bs, 0, 2, Direction::Out).unwrap();
    assert_eq!(fused.rank(), 2);
    assert_eq!(fused.flux(), &U1Sector(1));
    for meta in fused.block_metas() {
        assert!(fused.is_allowed_block(&meta.coord));
    }
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn fuse_count_less_than_2() {
    let bs = sample_u1_rank2();
    let result = fuse_legs_block_sparse(&backend(), &bs, 0, 1, Direction::Out);
    assert!(result.is_err());
    assert!(format!("{}", result.err().unwrap()).contains("count"));
}

#[test]
fn fuse_out_of_range() {
    let bs = sample_u1_rank2();
    let result = fuse_legs_block_sparse(&backend(), &bs, 1, 2, Direction::Out);
    assert!(result.is_err());
    assert!(format!("{}", result.err().unwrap()).contains("out of range"));
}
