use arnet_core::Complex;
use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_native::NativeBackend;
use arnet_tensor::test_fixtures::{legs, out_in_legs, square_legs};
use arnet_tensor::{
    BlockCoord, BlockSparseTensorData, DenseTensorData, Direction, Sector, U1Sector,
};

use super::{
    expm_antihermitian_block_sparse_dense, expm_block_sparse_dense,
    expm_hermitian_block_sparse_dense,
};
use crate::block_sparse_decomp::fused_sector::{
    assemble_sector_matrix, compute_fused_sector_groups,
};
use crate::block_sparse_decomp::to_vec_in_order;
use crate::expm::{expm_antihermitian_dense, expm_dense, expm_hermitian_dense};

fn backend() -> NativeBackend {
    NativeBackend::new()
}

fn order() -> MemoryOrder {
    backend().preferred_order()
}

type ExpmOut<T, S> = Result<BlockSparseTensorData<T, S>, crate::error::LinalgError>;

/// Assert a result is `Err(LinalgError::InvalidArgument)`, optionally carrying a
/// message substring. Matching the variant explicitly fails on a wrong-variant
/// or wrong-path error rather than passing on a coincidental message match.
fn expect_invalid_argument<T: Scalar, S: Sector>(result: ExpmOut<T, S>, substr: Option<&str>) {
    match result {
        Err(crate::error::LinalgError::InvalidArgument(msg)) => {
            if let Some(s) = substr {
                assert!(
                    msg.contains(s),
                    "expected InvalidArgument containing {s:?}, got: {msg}"
                );
            }
        }
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
/// storage order. A non-symmetric block differs between the two orders, so the
/// placement must honor `order`.
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

/// Differential per-sector oracle: for each fused sector, the assembled block of
/// the block-sparse result must equal the dense exponential of the assembled
/// operand block. This validates the block-sparse plumbing (fuse → assemble →
/// exponentiate → scatter) by reducing it to the per-sector dense kernel, which
/// has its own tests; it does not re-test the dense exponential. The sector-count
/// equality guards against a dropped or spurious sector.
fn verify_expm<T, S, FD>(
    tensor: &BlockSparseTensorData<T, S>,
    result: &BlockSparseTensorData<T, S>,
    nrow: usize,
    order: MemoryOrder,
    dense_expm: FD,
) where
    T: Scalar<Real = f64>,
    S: Sector + PartialEq,
    FD: Fn(&DenseTensorData<T>) -> DenseTensorData<T>,
{
    let groups = compute_fused_sector_groups(tensor, nrow);
    let res_groups = compute_fused_sector_groups(result, nrow);
    assert_eq!(res_groups.len(), groups.len(), "sector count");
    for group in &groups {
        let original = assemble_sector_matrix(tensor, group, order);
        let dense_in = DenseTensorData::from_raw_parts(original, vec![group.m, group.n], order);
        let expected = to_vec_in_order(&dense_expm(&dense_in), order);
        let actual = assemble_sector_matrix(result, group, order);
        assert_close(&expected, &actual, 1e-10);
    }
}

/// Assert a per-sector block equals its own conjugate transpose (Hermitian) or
/// its negation (anti-Hermitian), so a fixture's claimed structural property is
/// checked rather than assumed. `sign` is `+1` for Hermitian, `-1` for
/// anti-Hermitian.
fn assert_adjoint_structure<T, S>(
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    order: MemoryOrder,
    sign: f64,
) where
    T: Scalar<Real = f64>,
    S: Sector,
{
    for group in &compute_fused_sector_groups(tensor, nrow) {
        let n = group.n;
        let a = assemble_sector_matrix(tensor, group, order);
        for i in 0..n {
            for j in 0..n {
                let aij = a[mat_idx(i, j, n, n, order)];
                let aji = a[mat_idx(j, i, n, n, order)];
                let target = T::from_real_imag(sign * aji.re(), -sign * aji.im());
                let d =
                    ((aij.re() - target.re()).powi(2) + (aij.im() - target.im()).powi(2)).sqrt();
                assert!(d < 1e-12, "adjoint structure violated at ({i},{j})");
            }
        }
    }
}

// -- Fixtures ----------------------------------------------------------------

/// Rank-2 U1, identity flux, non-symmetric real blocks. Sector 0 is the 2×2
/// rotation-like block `[[1, -2], [2, 1]]` (complex spectrum `1 ± 2i`); sector 1
/// is a 3×3 general block. Reused from the `eig` fixtures: identity flux and
/// QN-square, so it exercises the general exponential path.
fn general_rank2_f64() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 3)]),
        U1Sector(0),
        order(),
    );
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

/// Rank-2 U1, identity flux, complex non-Hermitian blocks. Sector 0 is a 2×2
/// block not equal to its conjugate transpose; sector 1 is a 1×1 scalar.
fn general_rank2_c64() -> BlockSparseTensorData<Complex<f64>, U1Sector> {
    let c = |re: f64, im: f64| Complex::new(re, im);
    let mut bs = BlockSparseTensorData::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 1)]),
        U1Sector(0),
        order(),
    );
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

/// Rank-4 U1, identity flux, `nrow = 2`. Fused sector 1 merges left/right tuples
/// `[(0,1),(1,0)]` into a non-symmetric 2×2 block; sectors 0 and 2 are dim-1.
/// Exercises the multi-tuple fused-sector path of the scatter helper.
fn general_rank4_f64() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::zeros(
        legs([
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In),
        ]),
        U1Sector(0),
        order(),
    );
    bs.block_data_mut(&BlockCoord(vec![0, 0, 0, 0])).unwrap()[0] = 2.0;
    bs.block_data_mut(&BlockCoord(vec![1, 1, 1, 1])).unwrap()[0] = 7.0;
    bs.block_data_mut(&BlockCoord(vec![0, 1, 0, 1])).unwrap()[0] = 3.0;
    bs.block_data_mut(&BlockCoord(vec![0, 1, 1, 0])).unwrap()[0] = 1.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 0, 1])).unwrap()[0] = 4.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 1, 0])).unwrap()[0] = 5.0;
    bs
}

/// Rank-2 U1, identity flux, real symmetric blocks (Hermitian for a real type).
/// Sector 0 is `[[2, 1], [1, 3]]` (`A == Aᵀ`); sector 1 is the scalar `[5]`.
fn hermitian_rank2_f64() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 1)]),
        U1Sector(0),
        order(),
    );
    fill_block(&mut bs, &[0, 0], &[2.0, 1.0, 1.0, 3.0], 2, 2, order());
    fill_block(&mut bs, &[1, 1], &[5.0], 1, 1, order());
    bs
}

/// Rank-2 U1, identity flux, complex Hermitian blocks. Sector 0 is
/// `[[2, 1+i], [1-i, 3]]` (real diagonal, conjugate off-diagonals, so
/// `A == A†`); sector 1 is the real scalar `[4]`.
fn hermitian_rank2_c64() -> BlockSparseTensorData<Complex<f64>, U1Sector> {
    let c = |re: f64, im: f64| Complex::new(re, im);
    let mut bs = BlockSparseTensorData::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 1)]),
        U1Sector(0),
        order(),
    );
    fill_block(
        &mut bs,
        &[0, 0],
        &[c(2.0, 0.0), c(1.0, 1.0), c(1.0, -1.0), c(3.0, 0.0)],
        2,
        2,
        order(),
    );
    fill_block(&mut bs, &[1, 1], &[c(4.0, 0.0)], 1, 1, order());
    bs
}

/// Rank-2 U1, identity flux, complex anti-Hermitian blocks. Sector 0 is
/// `[[i, 1+i], [-1+i, 2i]]` (purely imaginary diagonal, `A[j,i] = -conj(A[i,j])`,
/// so `A† == -A`); sector 1 is the purely imaginary scalar `[3i]`.
fn antihermitian_rank2_c64() -> BlockSparseTensorData<Complex<f64>, U1Sector> {
    let c = |re: f64, im: f64| Complex::new(re, im);
    let mut bs = BlockSparseTensorData::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 1)]),
        U1Sector(0),
        order(),
    );
    fill_block(
        &mut bs,
        &[0, 0],
        &[c(0.0, 1.0), c(1.0, 1.0), c(-1.0, 1.0), c(0.0, 2.0)],
        2,
        2,
        order(),
    );
    fill_block(&mut bs, &[1, 1], &[c(0.0, 3.0)], 1, 1, order());
    bs
}

// -- Reconstruction (general expm) -------------------------------------------

#[test]
fn expm_rank2_f64() {
    let a = general_rank2_f64();
    let result = expm_block_sparse_dense(&backend(), &a, 1).unwrap();
    verify_expm(&a, &result, 1, order(), |d| {
        expm_dense(&backend(), d, 1).unwrap()
    });
}

#[test]
fn expm_rank2_complex() {
    let a = general_rank2_c64();
    let result = expm_block_sparse_dense(&backend(), &a, 1).unwrap();
    verify_expm(&a, &result, 1, order(), |d| {
        expm_dense(&backend(), d, 1).unwrap()
    });
}

#[test]
fn expm_rank4_multi_tuple_nrow2() {
    let a = general_rank4_f64();
    let result = expm_block_sparse_dense(&backend(), &a, 2).unwrap();
    verify_expm(&a, &result, 2, order(), |d| {
        expm_dense(&backend(), d, 1).unwrap()
    });
}

// -- Reconstruction (Hermitian / anti-Hermitian) -----------------------------

#[test]
fn expm_hermitian_rank2_f64() {
    let a = hermitian_rank2_f64();
    assert_adjoint_structure(&a, 1, order(), 1.0);
    let result = expm_hermitian_block_sparse_dense(&backend(), &a, 1).unwrap();
    verify_expm(&a, &result, 1, order(), |d| {
        expm_hermitian_dense(&backend(), d, 1).unwrap()
    });
}

#[test]
fn expm_hermitian_rank2_c64() {
    let a = hermitian_rank2_c64();
    assert_adjoint_structure(&a, 1, order(), 1.0);
    let result = expm_hermitian_block_sparse_dense(&backend(), &a, 1).unwrap();
    verify_expm(&a, &result, 1, order(), |d| {
        expm_hermitian_dense(&backend(), d, 1).unwrap()
    });
}

#[test]
fn expm_antihermitian_rank2_c64() {
    let a = antihermitian_rank2_c64();
    assert_adjoint_structure(&a, 1, order(), -1.0);
    let result = expm_antihermitian_block_sparse_dense(&backend(), &a, 1).unwrap();
    verify_expm(&a, &result, 1, order(), |d| {
        expm_antihermitian_dense(&backend(), d, 1).unwrap()
    });
}

// Order invariance: block-sparse kernels run only under the backend's
// `preferred_order` (the `NativeBackend` is fixed to `ColumnMajor`), and the
// op-entry layout-order gate rejects a mismatched tensor — so there is no
// `RowMajor` native backend to exercise here. The `RowMajor` branch of
// `build_square_tensor` (and of the shared `assemble_sector_matrix`) is covered
// by `cargo make litmus`, which re-runs the host-pinned crates against an
// alternate substrate, exactly as for the sibling decomposition builders.

// -- Validation --------------------------------------------------------------

#[test]
fn nonidentity_flux_rejected() {
    let bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        out_in_legs(
            vec![(U1Sector(0), 2), (U1Sector(1), 3)],
            vec![(U1Sector(0), 4)],
        ),
        U1Sector(1),
        order(),
    );
    expect_invalid_argument(expm_block_sparse_dense(&backend(), &bs, 1), Some("flux"));
}

#[test]
fn missing_partner_sector_rejected() {
    let bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        out_in_legs(
            vec![(U1Sector(0), 2), (U1Sector(1), 3)],
            vec![(U1Sector(0), 2)],
        ),
        U1Sector(0),
        order(),
    );
    expect_invalid_argument(expm_block_sparse_dense(&backend(), &bs, 1), Some("square"));
}

#[test]
fn dimension_mismatch_rejected() {
    let bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        out_in_legs(vec![(U1Sector(0), 2)], vec![(U1Sector(0), 3)]),
        U1Sector(0),
        order(),
    );
    expect_invalid_argument(expm_block_sparse_dense(&backend(), &bs, 1), Some("square"));
}

#[test]
fn nrow_out_of_range_rejected() {
    let a = general_rank2_f64();
    expect_invalid_argument(expm_block_sparse_dense(&backend(), &a, 0), Some("nrow"));
    expect_invalid_argument(expm_block_sparse_dense(&backend(), &a, 2), Some("nrow"));
}

#[test]
fn antihermitian_real_type_rejected() {
    let a = hermitian_rank2_f64();
    expect_invalid_argument(
        expm_antihermitian_block_sparse_dense(&backend(), &a, 1),
        Some("complex"),
    );
}

/// The real-type rejection precedes the sector loop, so an empty real operand
/// (no sectors to iterate) still errors rather than vacuously succeeding.
#[test]
fn antihermitian_real_type_rejected_when_empty() {
    let bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        out_in_legs(Vec::<(U1Sector, usize)>::new(), Vec::new()),
        U1Sector(0),
        order(),
    );
    expect_invalid_argument(
        expm_antihermitian_block_sparse_dense(&backend(), &bs, 1),
        Some("complex"),
    );
}
