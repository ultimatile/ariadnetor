use ariadnetor_core::Complex;
use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::test_fixtures::{legs, out_in_legs, square_legs};
use ariadnetor_tensor::{BlockCoord, BlockSparseTensorData, Direction, Sector, U1Sector};
use num_traits::Zero;

use super::eig_block_sparse_with_policy_dense;
use crate::block_sparse_decomp::BlockScalars;
use crate::block_sparse_decomp::fused_sector::{
    assemble_sector_matrix, compute_fused_sector_groups,
};

fn backend() -> NativeBackend {
    NativeBackend::new()
}

fn order() -> MemoryOrder {
    backend().preferred_order()
}

/// Output shape keyed on the complex element type `C` the decomposition
/// produces. Keying on `C` (not the input scalar `T`) keeps the type inferable
/// from a result value: the eigenvalues and eigenvectors mention only `C`, so a
/// `T::Complex`-keyed alias would leave `T` ambiguous at a call site.
type EigOut<C, S> =
    Result<(BlockScalars<C, S>, BlockSparseTensorData<C, S>), crate::error::LinalgError>;

fn run<T: Scalar, S: Sector>(
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
) -> EigOut<T::Complex, S> {
    eig_block_sparse_with_policy_dense(&backend(), tensor, nrow, ExecPolicy::Sequential)
}

/// Assert a result is `Err(LinalgError::InvalidArgument)`, optionally carrying
/// a message substring. Matching the variant explicitly fails a test when the
/// error comes from a different variant or code path, not just any error whose
/// message happens to contain the substring.
fn expect_invalid_argument<C: Scalar>(result: EigOut<C, U1Sector>, substr: Option<&str>) {
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

/// Write a `rows × cols` block from a logical row-major matrix into the
/// tensor's storage order. A non-symmetric block differs between row-major and
/// column-major, so the placement must honor `order`.
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

/// Per-sector eigenpair check: `A_q V_q ≈ V_q diag(w_q)`, i.e. each column is a
/// right eigenvector. This holds for the raw eigenpairs regardless of
/// diagonalizability (no `V⁻¹`), so it covers defective and repeated-spectrum
/// blocks too. The original block (type `T`) is widened to `T::Complex` to
/// multiply against the complex eigenvectors.
fn verify_reconstruction<T, S: Sector>(
    tensor: &BlockSparseTensorData<T, S>,
    w: &BlockScalars<T::Complex, S>,
    v: &BlockSparseTensorData<T::Complex, S>,
    nrow: usize,
    order: MemoryOrder,
) where
    T: Scalar<Real = f64>,
    T::Complex: Scalar<Real = f64>,
{
    let groups = compute_fused_sector_groups(tensor, nrow);
    let v_groups = compute_fused_sector_groups(v, nrow);
    // Completeness: eigenvalues and eigenvector bond carry exactly the matched
    // fused sectors — no sector dropped, none spurious.
    assert_eq!(w.values.len(), groups.len(), "eigenvalue sector count");
    assert_eq!(v_groups.len(), groups.len(), "eigenvector sector count");
    for group in &groups {
        let original = assemble_sector_matrix(tensor, group, order);
        let w_q: &[T::Complex] = w
            .values
            .iter()
            .find(|(s, _)| *s == group.sector)
            .map(|(_, vs)| vs.as_slice())
            .unwrap();
        let v_g = v_groups.iter().find(|g| g.sector == group.sector).unwrap();
        let v_mat = assemble_sector_matrix(v, v_g, order);
        let n = group.n;
        let mut lhs = vec![T::Complex::zero(); n * n];
        let mut rhs = vec![T::Complex::zero(); n * n];
        for i in 0..n {
            for k in 0..n {
                lhs[mat_idx(i, k, n, n, order)] = (0..n).fold(T::Complex::zero(), |acc, j| {
                    acc + original[mat_idx(i, j, n, n, order)].into_complex()
                        * v_mat[mat_idx(j, k, n, n, order)]
                });
                rhs[mat_idx(i, k, n, n, order)] = v_mat[mat_idx(i, k, n, n, order)] * w_q[k];
            }
        }
        assert_close(&lhs, &rhs, 1e-10);
    }
}

// -- Fixtures ----------------------------------------------------------------

/// Rank-2 U1, identity flux, non-symmetric real blocks: sector 0 is a 2×2
/// rotation-like block `[[1, -2], [2, 1]]` with eigenvalues `1 ± 2i` (derived:
/// char. poly `λ² − 2λ + 5`, discriminant `−16 < 0`), so the real operand has a
/// genuinely complex spectrum; sector 1 is a 3×3 general block.
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

/// Rank-2 U1, identity flux, complex non-Hermitian blocks: sector 0 is a 2×2
/// block that is not equal to its conjugate transpose; sector 1 is a 1×1
/// scalar.
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

/// Rank-4 U1, identity flux, `nrow = 2`. Fused sector 1 merges left/right
/// tuples [(0,1),(1,0)] into a non-symmetric 2×2 block; sectors 0 and 2 are
/// dim-1.
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
    // Fused sector 1, 2×2 over tuples [(0,1),(1,0)] — non-symmetric.
    bs.block_data_mut(&BlockCoord(vec![0, 1, 0, 1])).unwrap()[0] = 3.0;
    bs.block_data_mut(&BlockCoord(vec![0, 1, 1, 0])).unwrap()[0] = 1.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 0, 1])).unwrap()[0] = 4.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 1, 0])).unwrap()[0] = 5.0;
    bs
}

// -- Reconstruction ----------------------------------------------------------

#[test]
fn reconstruct_rank2_f64() {
    let a = general_rank2_f64();
    let (w, v) = run(&a, 1).unwrap();
    verify_reconstruction(&a, &w, &v, 1, order());
}

#[test]
fn reconstruct_rank2_complex() {
    let a = general_rank2_c64();
    let (w, v) = run(&a, 1).unwrap();
    verify_reconstruction(&a, &w, &v, 1, order());
}

#[test]
fn reconstruct_rank4_multi_tuple_nrow2() {
    let a = general_rank4_f64();
    let (w, v) = run(&a, 2).unwrap();
    verify_reconstruction(&a, &w, &v, 2, order());
}

// -- Complex spectrum --------------------------------------------------------

#[test]
fn eigenvalues_complex_for_real_operand() {
    // Sector 0 block [[1, -2], [2, 1]] has eigenvalues 1 ± 2i, so a real
    // operand must yield non-real eigenvalues — the defining reason `eig`
    // returns `T::Complex` rather than `T`.
    let a = general_rank2_f64();
    let (w, _v) = run(&a, 1).unwrap();
    let s0 = w
        .values
        .iter()
        .find(|(s, _)| *s == U1Sector(0))
        .map(|(_, vs)| vs.as_slice())
        .unwrap();
    assert!(
        s0.iter().any(|z| z.im().abs() > 1e-9),
        "expected a non-real eigenvalue from a non-symmetric real block, got {s0:?}"
    );
}

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
    expect_invalid_argument(run(&bs, 1), Some("flux"));
}

#[test]
fn missing_partner_sector_rejected() {
    // Left has fused sectors {0, 1}; right has only {0}, so sector 1 has no
    // matching right partner.
    let bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        out_in_legs(
            vec![(U1Sector(0), 2), (U1Sector(1), 3)],
            vec![(U1Sector(0), 2)],
        ),
        U1Sector(0),
        order(),
    );
    expect_invalid_argument(run(&bs, 1), Some("square"));
}

#[test]
fn dimension_mismatch_rejected() {
    // Both sides have fused sector 0, but with mismatched total dimension.
    let bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        out_in_legs(vec![(U1Sector(0), 2)], vec![(U1Sector(0), 3)]),
        U1Sector(0),
        order(),
    );
    expect_invalid_argument(run(&bs, 1), Some("square"));
}

#[test]
fn nrow_out_of_range_rejected() {
    let a = general_rank2_f64();
    expect_invalid_argument(run(&a, 0), Some("nrow"));
    expect_invalid_argument(run(&a, 2), Some("nrow"));
}
