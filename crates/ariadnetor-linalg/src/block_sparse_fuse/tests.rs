use arnet_core::backend::MemoryOrder;
use arnet_native::NativeBackend;
use arnet_tensor::{BlockCoord, BlockSparse, Direction, QNIndex, U1Sector, Z2Sector};

use super::copy_fused_block;
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
    // Full Kronecker product: (0,0)→sector 0 dim 4, (0,1)→sector -1 dim 6,
    // (1,0)→sector 1 dim 6, (1,1)→sector 0 dim 9. Sectors: {-1:6, 0:13, 1:6}
    assert_eq!(fused.shape(), &[25]);

    // Only flux-conserving blocks are stored (sector 0, dim 13)
    let orig_elems: usize = bs.block_metas().iter().map(|m| m.size).sum();
    let fused_stored: usize = fused.block_metas().iter().map(|m| m.size).sum();
    assert_eq!(fused_stored, orig_elems);
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

    // Find the flux-conserving block (sector 0) in each.
    // Out: stored sectors sorted = [-1, 0, 1], sector 0 is block index 1
    // In: stored sectors = dual of directed = [1, 0, -1], sorted = [-1, 0, 1], sector 0 is block 1
    let out_data = fused_out.block_data(&BlockCoord(vec![1])).unwrap();
    let in_data = fused_in.block_data(&BlockCoord(vec![1])).unwrap();
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

    // Z2: (0,0)→0 dim 4, (0,1)→1 dim 6, (1,0)→1 dim 6, (1,1)→0 dim 9
    // Sectors: {0: 13, 1: 12}. Total = 25.
    assert_eq!(fused.rank(), 1);
    assert_eq!(fused.shape(), &[25]);
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
// Contract: fused QNIndex depends only on input QNIndices, not stored blocks
// ---------------------------------------------------------------------------

/// Two tensors with identical QNIndices but different stored blocks must
/// produce the same fused QNIndex. This guarantees that adjacent MPS sites
/// fusing the same bond from opposite sides get compatible sector structures.
#[test]
fn fused_qnindex_independent_of_stored_blocks() {
    // Use indices where flux 0 vs flux 2 gives different block counts.
    // Out(0:1, 1:1), Out(0:1, 1:1), In(0:1, 1:1, 2:1)
    // Flux conservation: s0 + s1 - s2 = flux
    // flux=0: s2 = s0+s1, blocks: (0,0,0),(0,1,1),(1,0,1),(1,1,2) → 4 blocks
    // flux=2: s2 = s0+s1-2, blocks: (1,1,0) only → 1 block
    let i0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let i1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let i2 = QNIndex::new(
        vec![(U1Sector(0), 1), (U1Sector(1), 1), (U1Sector(2), 1)],
        Direction::In,
    );

    let bs_many =
        BlockSparse::<f64, U1Sector>::zeros(vec![i0.clone(), i1.clone(), i2.clone()], U1Sector(0));
    let bs_few = BlockSparse::<f64, U1Sector>::zeros(vec![i0, i1, i2], U1Sector(2));

    // Precondition: different number of stored blocks
    assert!(
        bs_many.num_blocks() > bs_few.num_blocks(),
        "fixture should have different block counts: {} vs {}",
        bs_many.num_blocks(),
        bs_few.num_blocks()
    );

    // Fuse axes (0,1): both should produce the same fused QNIndex
    let fused_many = fuse_legs_block_sparse(&backend(), &bs_many, 0, 2, Direction::Out).unwrap();
    let fused_few = fuse_legs_block_sparse(&backend(), &bs_few, 0, 2, Direction::Out).unwrap();

    let qi_many = fused_many.indices()[0].blocks();
    let qi_few = fused_few.indices()[0].blocks();
    assert_eq!(
        qi_many, qi_few,
        "fused QNIndex must be identical regardless of stored blocks"
    );
}

// ---------------------------------------------------------------------------
// Non-trivial leading: fuse trailing axes with leading > 1 and fused_offset > 0
// ---------------------------------------------------------------------------

/// Catches CM mutation: `fused_offset * leading` → `fused_offset / leading`.
/// Requires `leading > 1` and `fused_offset > 0` so that `fused_offset * leading`
/// differs from `fused_offset / leading` (integer division).
#[test]
fn fuse_trailing_axes_with_nontrivial_leading() {
    // Rank-3: Out(0:2), Out(0:1, 1:1), In(0:1, 1:1). Flux = 0.
    // Fuse axes 1 and 2 → leading = dim(axis 0) = 2.
    // Sector fusion for (Out s1, In s2) → s1 - s2:
    //   (0,0)→0 dim=1, (1,1)→0 dim=1 → fused sector 0 has fused_dim=2
    // The second tuple (1,1) has fused_offset=1. With leading=2, the CM
    // dst_start = fused_offset * leading = 1 * 2 = 2. A mutation to / gives 0.
    let i0 = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let i1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let i2 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let mut bs = BlockSparse::<f64, U1Sector>::zeros(vec![i0, i1, i2], U1Sector(0));

    // Block (0, 0, 0): shape [2, 1, 1] = 2 elements
    bs.block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .unwrap()
        .copy_from_slice(&[10.0, 20.0]);
    // Block (0, 1, 1): shape [2, 1, 1] = 2 elements
    bs.block_data_mut(&BlockCoord(vec![0, 1, 1]))
        .unwrap()
        .copy_from_slice(&[30.0, 40.0]);

    // Fuse axes (1, 2) with Out direction
    let fused = fuse_legs_block_sparse(&backend(), &bs, 1, 2, Direction::Out).unwrap();
    assert_eq!(fused.rank(), 2);

    // Output block for fused sector 0: shape [2 (leading), 2 (fused_dim)].
    // CM layout [2, 2]: flat = leading_idx + leading * fused_idx
    //   (0,0)=10, (1,0)=20, (0,1)=30, (1,1)=40
    // So data should be [10, 20, 30, 40].
    // With mutation (fused_offset*leading → fused_offset/leading):
    // tuple (1,1) would overwrite offset 0 instead of 2, corrupting the data.
    let fused_block = fused
        .block_metas()
        .iter()
        .find(|m| m.size == 4)
        .expect("should have a 2×2 block");
    let data = fused.block_data(&fused_block.coord).unwrap();
    assert_eq!(data, &[10.0, 20.0, 30.0, 40.0]);
}

// ---------------------------------------------------------------------------
// RowMajor: direct tests for copy_fused_block RM path
// ---------------------------------------------------------------------------

/// RM path: two blocks with fused=1 are merged into fused_total=2.
/// Catches all arithmetic mutations on lines 194-200 (RM branch).
#[test]
fn copy_fused_block_row_major_basic() {
    // leading=2, fused=1, trailing=3 for each source block.
    // RM tensor [2, 1, 3]: flat index = l * (fused*trailing) + f * trailing + t
    let src_a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src_b: Vec<f64> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let mut dst = vec![0.0f64; 12]; // [2, 2, 3] in RM

    copy_fused_block(&src_a, &mut dst, 2, 1, 2, 3, 0, MemoryOrder::RowMajor);
    copy_fused_block(&src_b, &mut dst, 2, 1, 2, 3, 1, MemoryOrder::RowMajor);

    // RM [2, 2, 3]:
    // l=0: [1,2,3, 7,8,9]  (fused_offset=0 then 1)
    // l=1: [4,5,6, 10,11,12]
    assert_eq!(
        dst,
        vec![
            1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0
        ]
    );
}

/// RM path with fused > 1: exercises src_stride = fused * trailing and
/// dst_stride = fused_total * trailing with all factors > 1.
#[test]
fn copy_fused_block_row_major_fused_gt_1() {
    // leading=2, fused=2, trailing=3, fused_total=3, fused_offset=1
    let src: Vec<f64> = (1..=12).map(|x| x as f64).collect(); // 2*2*3=12
    let mut dst = vec![0.0f64; 18]; // 2*3*3=18

    copy_fused_block(&src, &mut dst, 2, 2, 3, 3, 1, MemoryOrder::RowMajor);

    // RM: src_stride = 2*3 = 6, dst_stride = 3*3 = 9
    // l=0: dst_start = 0*9 + 1*3 = 3, copy src[0..6] → dst[3..9]
    // l=1: dst_start = 1*9 + 1*3 = 12, copy src[6..12] → dst[12..18]
    let mut expected = vec![0.0; 18];
    expected[3..9].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    expected[12..18].copy_from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    assert_eq!(dst, expected);
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
