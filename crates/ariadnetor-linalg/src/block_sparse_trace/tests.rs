//! Tests for the block-sparse partial trace.
//!
//! Fixtures are derived from the flux-conservation law, not assumed:
//! a block at coord `c` is allowed iff the fused directed sectors equal the
//! flux, so each fixture's allowed-block set and expected trace are computed
//! directly from that predicate.

use ariadnetor_core::backend::{ComputeBackend, MemoryOrder};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::test_fixtures::{legs, out_in_legs, square_legs};
use ariadnetor_tensor::{BlockCoord, BlockSparseTensorData, Direction, U1Sector};

use super::trace_block_sparse_dense;

fn backend() -> NativeBackend {
    NativeBackend::new()
}

fn order() -> MemoryOrder {
    backend().preferred_order()
}

/// Set a block's data from a flat slice in the layout's memory order.
fn set_block(bs: &mut BlockSparseTensorData<f64, U1Sector>, coord: &[usize], data: &[f64]) {
    bs.block_data_mut(&BlockCoord(coord.to_vec()))
        .unwrap()
        .copy_from_slice(data);
}

// ---------------------------------------------------------------------------
// Rank-2 full trace: a flux=0 (block-diagonal) operator.
//
// leg0 Out{0:2, 1:2}, leg1 In{0:2, 1:2}, flux 0.
// Allowed blocks satisfy s_i - s_j == 0, i.e. the diagonal (0,0), (1,1).
// Tracing pair (0,1) sums the diagonal of both sector blocks into a scalar.
// ---------------------------------------------------------------------------

fn sample_rank2() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 2)]),
        U1Sector(0),
        order(),
    );
    // Block (0,0) = [[1,2],[3,4]] -> diagonal 1+4 = 5.
    set_block(&mut bs, &[0, 0], &[1.0, 2.0, 3.0, 4.0]);
    // Block (1,1) = [[5,6],[7,8]] -> diagonal 5+8 = 13.
    set_block(&mut bs, &[1, 1], &[5.0, 6.0, 7.0, 8.0]);
    bs
}

#[test]
fn full_trace_rank2_sums_sector_diagonals() {
    let bs = sample_rank2();
    let result = trace_block_sparse_dense(&backend(), &bs, &[(0, 1)]).unwrap();

    // Output is a rank-0 scalar with the input flux.
    assert_eq!(result.rank(), 0);
    assert_eq!(result.flux(), &U1Sector(0));
    assert_eq!(result.num_blocks(), 1);
    let scalar = result.block_data(&BlockCoord(vec![])).unwrap();
    assert_eq!(scalar, &[18.0]); // 5 + 13
}

#[test]
fn full_trace_of_non_identity_flux_is_zero() {
    // flux 1: allowed blocks satisfy s_i - s_j == 1, i.e. (1,0) only — no
    // diagonal block, so the full trace has no diagonal support.
    let mut bs = BlockSparseTensorData::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 2)]),
        U1Sector(1),
        order(),
    );
    set_block(&mut bs, &[1, 0], &[1.0, 2.0, 3.0, 4.0]);

    let result = trace_block_sparse_dense(&backend(), &bs, &[(0, 1)]).unwrap();
    // Non-identity flux admits no rank-0 block: the result is the zero scalar.
    assert_eq!(result.rank(), 0);
    assert_eq!(result.flux(), &U1Sector(1));
    assert_eq!(result.num_blocks(), 0);
}

// ---------------------------------------------------------------------------
// Rank-4 partial trace: two traced sectors fold into one free output block.
//
// legs [Out{0:1}, Out{0:2, 1:2}, In{0:2, 1:2}, In{0:1}], flux 0.
// Allowed blocks satisfy s_a - s_b == 0 (the unit free legs contribute 0):
// (0,0,0,0) and (0,1,1,0). Tracing pair (1,2) leaves free axes [0,3]; both
// blocks map to free coord (0,0), so the two traced sectors sum there.
// ---------------------------------------------------------------------------

fn sample_rank4() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::zeros(
        legs([
            (vec![(U1Sector(0), 1)], Direction::Out),
            (vec![(U1Sector(0), 2), (U1Sector(1), 2)], Direction::Out),
            (vec![(U1Sector(0), 2), (U1Sector(1), 2)], Direction::In),
            (vec![(U1Sector(0), 1)], Direction::In),
        ]),
        U1Sector(0),
        order(),
    );
    // Block (0,0,0,0): shape [1,2,2,1]; traced 2x2 part [[10,20],[30,40]] -> 50.
    set_block(&mut bs, &[0, 0, 0, 0], &[10.0, 20.0, 30.0, 40.0]);
    // Block (0,1,1,0): shape [1,2,2,1]; traced 2x2 part [[1,2],[3,4]] -> 5.
    set_block(&mut bs, &[0, 1, 1, 0], &[1.0, 2.0, 3.0, 4.0]);
    bs
}

#[test]
fn partial_trace_rank4_folds_sectors_into_one_block() {
    let bs = sample_rank4();
    let result = trace_block_sparse_dense(&backend(), &bs, &[(1, 2)]).unwrap();

    // Free legs [0, 3] survive in order; flux preserved.
    assert_eq!(result.rank(), 2);
    assert_eq!(result.flux(), &U1Sector(0));
    assert_eq!(result.shape(), &[1, 1]);
    let block = result.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(block, &[55.0]); // 50 + 5
}

#[test]
fn empty_pairs_is_clone() {
    let bs = sample_rank2();
    let result = trace_block_sparse_dense(&backend(), &bs, &[]).unwrap();
    assert_eq!(result.shape(), bs.shape());
    for meta in bs.block_metas() {
        assert_eq!(
            result.block_data(&meta.coord).unwrap(),
            bs.block_data(&meta.coord).unwrap()
        );
    }
}

// ---------------------------------------------------------------------------
// RowMajor / ColumnMajor invariance: the same logical operator laid out in
// each order yields the same trace.
// ---------------------------------------------------------------------------

fn build_rank2(order: MemoryOrder) -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 2)]),
        U1Sector(0),
        order,
    );
    // Logical [[1,2],[3,4]] in the requested order; the diagonal (1,4) is
    // layout-invariant, so the trace must match across orders.
    let (b00, b11): (&[f64], &[f64]) = match order {
        MemoryOrder::RowMajor => (&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]),
        MemoryOrder::ColumnMajor => (&[1.0, 3.0, 2.0, 4.0], &[5.0, 7.0, 6.0, 8.0]),
    };
    set_block(&mut bs, &[0, 0], b00);
    set_block(&mut bs, &[1, 1], b11);
    bs
}

#[test]
fn trace_is_rowmajor_columnmajor_invariant() {
    let rm = build_rank2(MemoryOrder::RowMajor);
    let cm = build_rank2(MemoryOrder::ColumnMajor);
    let r_rm = trace_block_sparse_dense(&backend(), &rm, &[(0, 1)]).unwrap();
    let r_cm = trace_block_sparse_dense(&backend(), &cm, &[(0, 1)]).unwrap();
    assert_eq!(
        r_rm.block_data(&BlockCoord(vec![])).unwrap(),
        r_cm.block_data(&BlockCoord(vec![])).unwrap()
    );
}

// ---------------------------------------------------------------------------
// Validation errors.
// ---------------------------------------------------------------------------

#[test]
fn rejects_mismatched_block_structure() {
    // leg0 {0:2, 1:2} vs leg1 {0:2, 1:3}: equal sectors, unequal dims.
    let bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        out_in_legs(
            vec![(U1Sector(0), 2), (U1Sector(1), 2)],
            vec![(U1Sector(0), 2), (U1Sector(1), 3)],
        ),
        U1Sector(0),
        order(),
    );
    let Err(err) = trace_block_sparse_dense(&backend(), &bs, &[(0, 1)]) else {
        panic!("expected mismatched-block-structure error");
    };
    assert!(format!("{err:?}").contains("block structure"));
}

#[test]
fn rejects_equal_directions() {
    // Two Out legs with identical blocks; flux 0 allows the (0,0) block.
    let bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 2), (U1Sector(1), 2)], Direction::Out),
            (vec![(U1Sector(0), 2), (U1Sector(1), 2)], Direction::Out),
        ]),
        U1Sector(0),
        order(),
    );
    let Err(err) = trace_block_sparse_dense(&backend(), &bs, &[(0, 1)]) else {
        panic!("expected equal-directions error");
    };
    assert!(format!("{err:?}").contains("opposite leg directions"));
}

#[test]
fn rejects_out_of_range_index() {
    let bs = sample_rank2();
    let Err(err) = trace_block_sparse_dense(&backend(), &bs, &[(0, 2)]) else {
        panic!("expected out-of-range error");
    };
    assert!(format!("{err:?}").contains("out of range"));
}

#[test]
fn rejects_self_pair() {
    let bs = sample_rank2();
    let Err(err) = trace_block_sparse_dense(&backend(), &bs, &[(0, 0)]) else {
        panic!("expected self-pair error");
    };
    assert!(format!("{err:?}").contains("Self-pair"));
}

#[test]
fn rejects_reused_index() {
    let bs = sample_rank4();
    // Axis 1 appears in two pairs.
    let Err(err) = trace_block_sparse_dense(&backend(), &bs, &[(1, 2), (1, 0)]) else {
        panic!("expected reused-index error");
    };
    assert!(format!("{err:?}").contains("multiple pairs"));
}
