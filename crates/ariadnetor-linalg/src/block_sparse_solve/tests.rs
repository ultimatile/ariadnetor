use arnet_core::Complex;
use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_native::NativeBackend;
use arnet_tensor::{
    BlockCoord, BlockSparseTensor, BlockSparseTensorData, Direction, QNIndex, Sector, U1Sector,
};

use super::{inverse_block_sparse_dense, solve_block_sparse_dense};
use crate::BlockSparseContractResult;
use crate::BlockSparseHostOps;
use crate::block_sparse_decomp::fused_sector::{
    assemble_sector_matrix, compute_fused_sector_groups,
};
use crate::block_sparse_decomp::to_vec_in_order;
use crate::solve::{inverse_dense, solve_dense};

fn backend() -> NativeBackend {
    NativeBackend::new()
}

fn order() -> MemoryOrder {
    backend().preferred_order()
}

type SolveOut<T, S> = Result<BlockSparseTensorData<T, S>, crate::error::LinalgError>;

/// Assert a result is `Err(LinalgError::InvalidArgument)` carrying a message
/// substring, failing on a wrong-variant or wrong-path error.
fn expect_invalid_argument<T: Scalar, S: Sector>(result: SolveOut<T, S>, substr: &str) {
    match result {
        Err(crate::error::LinalgError::InvalidArgument(msg)) => assert!(
            msg.contains(substr),
            "expected InvalidArgument containing {substr:?}, got: {msg}"
        ),
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        Ok(_) => panic!("expected an error, got Ok"),
    }
}

fn mat_idx(row: usize, col: usize, rows: usize, cols: usize, order: MemoryOrder) -> usize {
    match order {
        MemoryOrder::RowMajor => row * cols + col,
        MemoryOrder::ColumnMajor => col * rows + row,
    }
}

/// Write a `rows × cols` block from a logical row-major matrix into the tensor's
/// storage order.
fn fill_block<T: Scalar, S: Sector>(
    bs: &mut BlockSparseTensorData<T, S>,
    coord: &[usize],
    logical_rowmajor: &[T],
    rows: usize,
    cols: usize,
    order: MemoryOrder,
) {
    let data = bs.block_data_mut(&BlockCoord(coord.to_vec())).unwrap();
    for r in 0..rows {
        for c in 0..cols {
            data[mat_idx(r, c, rows, cols, order)] = logical_rowmajor[r * cols + c];
        }
    }
}

fn assert_close<T: Scalar<Real = f64>>(a: &[T], b: &[T], tol: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = ((x.re() - y.re()).powi(2) + (x.im() - y.im()).powi(2)).sqrt();
        assert!(
            d < tol,
            "index {i}: ({},{}) vs ({},{}) d={d}",
            x.re(),
            x.im(),
            y.re(),
            y.im()
        );
    }
}

/// Compare two block-sparse tensors' joined data block by block.
fn assert_data_close<T: Scalar<Real = f64>>(
    a: &BlockSparseTensorData<T, impl Sector>,
    b: &BlockSparseTensorData<T, impl Sector>,
    tol: f64,
) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    assert_eq!(a.num_blocks(), b.num_blocks(), "block count mismatch");
    for meta in a.block_metas() {
        let xa = a.block_data(&meta.coord).unwrap();
        let xb = b.block_data(&meta.coord).unwrap();
        assert_close(xa, xb, tol);
    }
}

/// Differential per-sector oracle for `solve`: for each fused sector, the
/// assembled block of the block-sparse result must equal the dense solve of the
/// assembled operator and RHS blocks. This validates the plumbing (fuse →
/// assemble → solve → scatter) by reducing it to the dense kernel, which has
/// its own tests; the sector-count equality guards a dropped or spurious sector.
fn verify_solve<T: Scalar<Real = f64>, S: Sector>(
    a: &BlockSparseTensorData<T, S>,
    b: &BlockSparseTensorData<T, S>,
    x: &BlockSparseTensorData<T, S>,
    nrow_a: usize,
    order: MemoryOrder,
) {
    let groups_a = compute_fused_sector_groups(a, nrow_a);
    let groups_b = compute_fused_sector_groups(b, nrow_a);
    let groups_x = compute_fused_sector_groups(x, nrow_a);
    assert_eq!(groups_x.len(), groups_b.len(), "sector count");
    for b_group in &groups_b {
        let a_group = groups_a
            .iter()
            .find(|g| g.sector() == b_group.sector())
            .unwrap();
        let x_group = groups_x
            .iter()
            .find(|g| g.sector() == b_group.sector())
            .unwrap();
        let m = a_group.m;
        let nrhs = b_group.n;
        let a_q = assemble_sector_matrix(a, a_group, order);
        let b_q = assemble_sector_matrix(b, b_group, order);
        let expected = solve_dense(
            &backend(),
            &arnet_tensor::DenseTensorData::from_raw_parts(a_q, vec![m, m], order),
            &arnet_tensor::DenseTensorData::from_raw_parts(b_q, vec![m, nrhs], order),
            1,
        )
        .unwrap();
        let actual = assemble_sector_matrix(x, x_group, order);
        assert_close(&to_vec_in_order(&expected, order), &actual, 1e-9);
    }
}

/// Differential per-sector oracle for `inverse`: the assembled block of the
/// result must equal the dense inverse of the assembled operand block.
fn verify_inverse<T: Scalar<Real = f64>, S: Sector>(
    a: &BlockSparseTensorData<T, S>,
    inv: &BlockSparseTensorData<T, S>,
    nrow: usize,
    order: MemoryOrder,
) {
    let groups = compute_fused_sector_groups(a, nrow);
    let res_groups = compute_fused_sector_groups(inv, nrow);
    assert_eq!(res_groups.len(), groups.len(), "sector count");
    for group in &groups {
        let m = group.m;
        let a_q = assemble_sector_matrix(a, group, order);
        let expected = inverse_dense(
            &backend(),
            &arnet_tensor::DenseTensorData::from_raw_parts(a_q, vec![m, m], order),
            1,
        )
        .unwrap();
        let actual = assemble_sector_matrix(inv, group, order);
        assert_close(&to_vec_in_order(&expected, order), &actual, 1e-9);
    }
}

// -- Fixtures ----------------------------------------------------------------

/// Rank-2 U1, identity flux, leg-mirrored. Sector 0 is the invertible 2×2 block
/// `[[1, -2], [2, 1]]` (det 5); sector 1 is an invertible 3×3 block (reused from
/// the `expm` / `eig` fixtures).
fn mirrored_rank2_f64() -> BlockSparseTensorData<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut bs = BlockSparseTensorData::zeros(vec![row, col], U1Sector(0), order());
    fill_block(&mut bs, &[0, 0], &[1.0, -2.0, 2.0, 1.0], 2, 2, order());
    fill_block(
        &mut bs,
        &[1, 1],
        &[5.0, 1.0, 2.0, 0.0, 6.0, 1.0, 1.0, 0.0, 7.0],
        3,
        3,
        order(),
    );
    bs
}

/// Rank-2 U1, identity flux, leg-mirrored, complex invertible blocks.
fn mirrored_rank2_c64() -> BlockSparseTensorData<Complex<f64>, U1Sector> {
    let c = |re: f64, im: f64| Complex::new(re, im);
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);
    let mut bs = BlockSparseTensorData::zeros(vec![row, col], U1Sector(0), order());
    fill_block(
        &mut bs,
        &[0, 0],
        &[c(1.0, 1.0), c(2.0, 0.0), c(0.0, 0.0), c(3.0, -1.0)],
        2,
        2,
        order(),
    );
    fill_block(&mut bs, &[1, 1], &[c(4.0, 2.0)], 1, 1, order());
    bs
}

/// Rank-4 U1, identity flux, leg-mirrored at `nrow = 2`. Fused sector 1 merges
/// left/right tuples `[(0,1),(1,0)]` into an invertible 2×2 block; sectors 0 and
/// 2 are dim-1. Reused from the `expm` rank-4 fixture.
fn mirrored_rank4_f64() -> BlockSparseTensorData<f64, U1Sector> {
    let out = || QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let inn = || QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let mut bs =
        BlockSparseTensorData::zeros(vec![out(), out(), inn(), inn()], U1Sector(0), order());
    bs.block_data_mut(&BlockCoord(vec![0, 0, 0, 0])).unwrap()[0] = 2.0;
    bs.block_data_mut(&BlockCoord(vec![1, 1, 1, 1])).unwrap()[0] = 7.0;
    bs.block_data_mut(&BlockCoord(vec![0, 1, 0, 1])).unwrap()[0] = 3.0;
    bs.block_data_mut(&BlockCoord(vec![0, 1, 1, 0])).unwrap()[0] = 1.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 0, 1])).unwrap()[0] = 4.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 1, 0])).unwrap()[0] = 5.0;
    bs
}

/// RHS for [`mirrored_rank2_f64`]: row leg equal to the operator's, identity
/// flux, one column block per sector. Sector 0 is `2×1`, sector 1 is `3×2`.
fn rhs_rank2_f64() -> BlockSparseTensorData<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 2)], Direction::In);
    let mut bs = BlockSparseTensorData::zeros(vec![row, col], U1Sector(0), order());
    fill_block(&mut bs, &[0, 0], &[3.0, 7.0], 2, 1, order());
    fill_block(
        &mut bs,
        &[1, 1],
        &[1.0, 2.0, 0.0, 4.0, 5.0, 6.0],
        3,
        2,
        order(),
    );
    bs
}

/// RHS for [`mirrored_rank2_c64`]: complex, identity flux, one column per sector.
fn rhs_rank2_c64() -> BlockSparseTensorData<Complex<f64>, U1Sector> {
    let c = |re: f64, im: f64| Complex::new(re, im);
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let mut bs = BlockSparseTensorData::zeros(vec![row, col], U1Sector(0), order());
    fill_block(
        &mut bs,
        &[0, 0],
        &[c(2.0, 1.0), c(0.0, -1.0)],
        2,
        1,
        order(),
    );
    fill_block(&mut bs, &[1, 1], &[c(1.0, 3.0)], 1, 1, order());
    bs
}

/// RHS for [`mirrored_rank2_f64`] with a **non-identity** flux (sector 1). The
/// flux shifts each row-sector's RHS columns to a different right-sector, so the
/// per-sector column counts differ from the identity-flux RHS.
fn rhs_rank2_flux1_f64() -> BlockSparseTensorData<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(-1), 2), (U1Sector(0), 1)], Direction::In);
    let mut bs = BlockSparseTensorData::zeros(vec![row, col], U1Sector(1), order());
    // flux 1 selects blocks with row_sec - col_sec = 1: row 0 (sec 0) pairs with
    // the col block of sector -1, row 1 (sec 1) with the col block of sector 0.
    fill_block(&mut bs, &[0, 0], &[1.0, 2.0, 3.0, 4.0], 2, 2, order());
    fill_block(&mut bs, &[1, 1], &[5.0, 6.0, 7.0], 3, 1, order());
    bs
}

/// RHS for [`mirrored_rank4_f64`]: rank-3, leading two legs equal the operator's
/// row legs, one column per fused sector.
fn rhs_rank4_f64() -> BlockSparseTensorData<f64, U1Sector> {
    let out = || QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let col = QNIndex::new(
        vec![(U1Sector(0), 1), (U1Sector(1), 1), (U1Sector(2), 1)],
        Direction::In,
    );
    let mut bs = BlockSparseTensorData::zeros(vec![out(), out(), col], U1Sector(0), order());
    bs.block_data_mut(&BlockCoord(vec![0, 0, 0])).unwrap()[0] = 1.0;
    bs.block_data_mut(&BlockCoord(vec![0, 1, 1])).unwrap()[0] = 2.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 1])).unwrap()[0] = 3.0;
    bs.block_data_mut(&BlockCoord(vec![1, 1, 2])).unwrap()[0] = 4.0;
    bs
}

// -- solve: per-sector oracle ------------------------------------------------

#[test]
fn solve_rank2_f64() {
    let (a, b) = (mirrored_rank2_f64(), rhs_rank2_f64());
    let x = solve_block_sparse_dense(&backend(), &a, &b, 1).unwrap();
    verify_solve(&a, &b, &x, 1, order());
}

#[test]
fn solve_rank2_complex() {
    let (a, b) = (mirrored_rank2_c64(), rhs_rank2_c64());
    let x = solve_block_sparse_dense(&backend(), &a, &b, 1).unwrap();
    verify_solve(&a, &b, &x, 1, order());
}

#[test]
fn solve_rank2_nonidentity_b_flux() {
    let (a, b) = (mirrored_rank2_f64(), rhs_rank2_flux1_f64());
    let x = solve_block_sparse_dense(&backend(), &a, &b, 1).unwrap();
    assert_eq!(x.layout().flux(), b.layout().flux(), "X inherits B's flux");
    verify_solve(&a, &b, &x, 1, order());
}

#[test]
fn solve_rank4_multi_tuple_nrow2() {
    let (a, b) = (mirrored_rank4_f64(), rhs_rank4_f64());
    let x = solve_block_sparse_dense(&backend(), &a, &b, 2).unwrap();
    verify_solve(&a, &b, &x, 2, order());
}

// -- inverse: per-sector oracle ----------------------------------------------

#[test]
fn inverse_rank2_f64() {
    let a = mirrored_rank2_f64();
    let inv = inverse_block_sparse_dense(&backend(), &a, 1).unwrap();
    verify_inverse(&a, &inv, 1, order());
}

#[test]
fn inverse_rank2_complex() {
    let a = mirrored_rank2_c64();
    let inv = inverse_block_sparse_dense(&backend(), &a, 1).unwrap();
    verify_inverse(&a, &inv, 1, order());
}

#[test]
fn inverse_rank4_multi_tuple_nrow2() {
    let a = mirrored_rank4_f64();
    let inv = inverse_block_sparse_dense(&backend(), &a, 2).unwrap();
    verify_inverse(&a, &inv, 2, order());
}

// -- End-to-end: re-contraction validates the output leg labels --------------

/// `A` contracted with the solved `X` (over `A`'s column legs and `X`'s row
/// legs) reproduces `B`. This validates that the output legs are labeled so the
/// result is re-contractable — the property the leg-mirroring precondition
/// exists to guarantee, which the per-sector oracle (self-consistent in `X`'s
/// own legs) cannot check.
#[test]
fn solve_reconstructs_b_via_contraction() {
    let a = BlockSparseTensor::from_data(mirrored_rank2_f64());
    let b = BlockSparseTensor::from_data(rhs_rank2_f64());
    let x = a.solve(&b, 1).unwrap();
    match a.contract(&x, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(ax) => assert_data_close(ax.data(), b.data(), 1e-9),
        BlockSparseContractResult::Scalar(_) => panic!("expected a tensor result"),
    }
}

/// `A` contracted with its inverse yields the identity on each sector block.
#[test]
fn inverse_contracts_to_identity() {
    let a = BlockSparseTensor::from_data(mirrored_rank2_f64());
    let inv = a.inverse(1).unwrap();
    match a.contract(&inv, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(prod) => {
            let order = order();
            for group in &compute_fused_sector_groups(prod.data(), 1) {
                let n = group.n;
                let block = assemble_sector_matrix(prod.data(), group, order);
                for i in 0..n {
                    for j in 0..n {
                        let want = if i == j { 1.0 } else { 0.0 };
                        let got = block[mat_idx(i, j, n, n, order)];
                        assert!((got - want).abs() < 1e-9, "I[{i},{j}] = {got}");
                    }
                }
            }
        }
        BlockSparseContractResult::Scalar(_) => panic!("expected a tensor result"),
    }
}

// -- Validation --------------------------------------------------------------

#[test]
fn solve_nrow_out_of_range_rejected() {
    let (a, b) = (mirrored_rank2_f64(), rhs_rank2_f64());
    expect_invalid_argument(solve_block_sparse_dense(&backend(), &a, &b, 0), "nrow");
    expect_invalid_argument(solve_block_sparse_dense(&backend(), &a, &b, 2), "nrow");
}

#[test]
fn solve_nonidentity_a_flux_rejected() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let a = BlockSparseTensorData::<f64, U1Sector>::zeros(vec![row, col], U1Sector(1), order());
    let b = rhs_rank2_f64();
    expect_invalid_argument(solve_block_sparse_dense(&backend(), &a, &b, 1), "flux");
}

#[test]
fn solve_non_mirrored_a_rejected() {
    // Square fused-sector universe but NOT leg-mirrored: both legs are `Out`, so
    // a column leg is not the dual of the row leg.
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let a = BlockSparseTensorData::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0), order());
    let b = rhs_rank2_f64();
    expect_invalid_argument(solve_block_sparse_dense(&backend(), &a, &b, 1), "mirrored");
}

#[test]
fn solve_b_row_legs_mismatch_rejected() {
    let a = mirrored_rank2_f64();
    // B's row leg has different block dims than A's row leg.
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let b = BlockSparseTensorData::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0), order());
    expect_invalid_argument(solve_block_sparse_dense(&backend(), &a, &b, 1), "row legs");
}

#[test]
fn inverse_nrow_out_of_range_rejected() {
    let a = mirrored_rank2_f64();
    expect_invalid_argument(inverse_block_sparse_dense(&backend(), &a, 0), "nrow");
    expect_invalid_argument(inverse_block_sparse_dense(&backend(), &a, 2), "nrow");
}

#[test]
fn inverse_nonidentity_flux_rejected() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let a = BlockSparseTensorData::<f64, U1Sector>::zeros(vec![row, col], U1Sector(1), order());
    expect_invalid_argument(inverse_block_sparse_dense(&backend(), &a, 1), "flux");
}

#[test]
fn inverse_non_mirrored_rejected() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let a = BlockSparseTensorData::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0), order());
    expect_invalid_argument(inverse_block_sparse_dense(&backend(), &a, 1), "mirrored");
}
