use arnet_core::Complex;
use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder};
use arnet_native::NativeBackend;
use arnet_tensor::test_fixtures::{legs, out_in_legs, square_legs};
use arnet_tensor::{BlockCoord, BlockSparseTensorData, Direction, Sector, U1Sector};

use super::eigh_block_sparse_with_policy_dense;
use crate::block_sparse_decomp::fused_sector::{
    assemble_sector_matrix, compute_fused_sector_groups,
};

fn backend() -> NativeBackend {
    NativeBackend::new()
}

fn order() -> MemoryOrder {
    backend().preferred_order()
}

type EighRunResult<T, S> = Result<
    (
        crate::block_sparse_decomp::BlockScalars<<T as Scalar>::Real, S>,
        BlockSparseTensorData<T, S>,
    ),
    crate::error::LinalgError,
>;

fn run<T: Scalar, S: Sector>(
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
) -> EighRunResult<T, S> {
    eigh_block_sparse_with_policy_dense(&backend(), tensor, nrow, ExecPolicy::Sequential)
}

/// Assert a result is `Err(LinalgError::InvalidArgument)`, optionally carrying
/// a message substring. Matching the variant explicitly fails a test when the
/// error comes from a different variant or code path, not just any error whose
/// message happens to contain the substring.
fn expect_invalid_argument<T: Scalar>(result: EighRunResult<T, U1Sector>, substr: Option<&str>) {
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
/// tensor's storage order. A complex Hermitian block differs between
/// row-major and column-major, so the placement must honor `order`.
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

/// Reconstruct `V diag(w) V†` for one sector block (`n × n`, in `order`).
fn reconstruct_eigh<T: Scalar>(v: &[T], w: &[T::Real], n: usize, order: MemoryOrder) -> Vec<T> {
    let mut h = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            h[mat_idx(i, j, n, n, order)] = (0..n).fold(T::zero(), |acc, k| {
                acc + v[mat_idx(i, k, n, n, order)].scale_real(w[k])
                    * v[mat_idx(j, k, n, n, order)].conj()
            });
        }
    }
    h
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

/// Per-sector reconstruction check: `V_q diag(w_q) V_q† ≈ H_q`.
fn verify_reconstruction<T: Scalar<Real = f64>, S: Sector + PartialEq>(
    tensor: &BlockSparseTensorData<T, S>,
    w: &crate::block_sparse_decomp::BlockScalars<T::Real, S>,
    v: &BlockSparseTensorData<T, S>,
    nrow: usize,
    order: MemoryOrder,
) {
    let groups = compute_fused_sector_groups(tensor, nrow);
    let v_groups = compute_fused_sector_groups(v, nrow);
    // Completeness: eigenvalues and eigenvector bond carry exactly the matched
    // fused sectors — no sector dropped, none spurious.
    assert_eq!(w.values.len(), groups.len(), "eigenvalue sector count");
    assert_eq!(v_groups.len(), groups.len(), "eigenvector sector count");
    for group in &groups {
        let original = assemble_sector_matrix(tensor, group, order);
        let w_q: &[f64] = w
            .values
            .iter()
            .find(|(s, _)| *s == group.sector)
            .map(|(_, vs)| vs.as_slice())
            .unwrap();
        let v_g = v_groups.iter().find(|g| g.sector == group.sector).unwrap();
        let v_mat = assemble_sector_matrix(v, v_g, order);
        let recon = reconstruct_eigh(&v_mat, w_q, group.n, order);
        assert_close(&recon, &original, 1e-10);
    }
}

// -- Fixtures ----------------------------------------------------------------

/// Rank-2 U1, identity flux, symmetric blocks: sector 0 is 2×2, sector 1 is 3×3.
fn hermitian_rank2_f64() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 3)]),
        U1Sector(0),
        order(),
    );
    // Symmetric matrices are order-agnostic in flat form.
    fill_block(&mut bs, &[0, 0], &[2.0, 1.0, 1.0, 3.0], 2, 2, order());
    fill_block(
        &mut bs,
        &[1, 1],
        &[5.0, 1.0, 0.0, 1.0, 6.0, 2.0, 0.0, 2.0, 7.0],
        3,
        3,
        order(),
    );
    bs
}

/// Rank-2 U1, identity flux, complex Hermitian blocks (`a_ij = conj(a_ji)`,
/// real diagonal): sector 0 is 2×2, sector 1 is a 1×1 real scalar.
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
    // dim-1 sector: a single real value.
    fill_block(&mut bs, &[1, 1], &[c(4.0, 0.0)], 1, 1, order());
    bs
}

/// Rank-4 U1, identity flux, `nrow = 2`. Fused sector 1 merges left/right
/// tuples [(0,1),(1,0)] into a symmetric 2×2 block; sectors 0 and 2 are dim-1.
fn hermitian_rank4_f64() -> BlockSparseTensorData<f64, U1Sector> {
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
    // Fused sector 0 (dim 1) and sector 2 (dim 1): diagonal reals.
    bs.block_data_mut(&BlockCoord(vec![0, 0, 0, 0])).unwrap()[0] = 2.0;
    bs.block_data_mut(&BlockCoord(vec![1, 1, 1, 1])).unwrap()[0] = 7.0;
    // Fused sector 1, 2×2 over tuples [(0,1),(1,0)] — symmetric: off-diagonals equal.
    bs.block_data_mut(&BlockCoord(vec![0, 1, 0, 1])).unwrap()[0] = 3.0;
    bs.block_data_mut(&BlockCoord(vec![0, 1, 1, 0])).unwrap()[0] = 1.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 0, 1])).unwrap()[0] = 1.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 1, 0])).unwrap()[0] = 5.0;
    bs
}

// -- Reconstruction ----------------------------------------------------------

#[test]
fn reconstruct_rank2_f64() {
    let h = hermitian_rank2_f64();
    let (w, v) = run(&h, 1).unwrap();
    verify_reconstruction(&h, &w, &v, 1, order());
}

#[test]
fn reconstruct_rank2_complex() {
    let h = hermitian_rank2_c64();
    let (w, v) = run(&h, 1).unwrap();
    verify_reconstruction(&h, &w, &v, 1, order());
}

#[test]
fn reconstruct_rank4_multi_tuple_nrow2() {
    let h = hermitian_rank4_f64();
    let (w, v) = run(&h, 2).unwrap();
    verify_reconstruction(&h, &w, &v, 2, order());
}

// -- Eigenvalue ordering -----------------------------------------------------

#[test]
fn eigenvalues_ascending_within_sector() {
    let h = hermitian_rank2_f64();
    let (w, _v) = run(&h, 1).unwrap();
    for (sector, vals) in &w.values {
        for pair in vals.windows(2) {
            assert!(
                pair[0] <= pair[1],
                "sector {sector:?} eigenvalues not ascending: {vals:?}"
            );
        }
    }
}

// -- Eigenvector orthonormality ----------------------------------------------

#[test]
fn eigenvectors_orthonormal_complex() {
    let h = hermitian_rank2_c64();
    let (_w, v) = run(&h, 1).unwrap();
    let v_groups = compute_fused_sector_groups(&v, 1);
    for g in &v_groups {
        let mat = assemble_sector_matrix(&v, g, order());
        let n = g.n;
        // (V† V)[k,l] = Σ_i conj(V[i,k]) V[i,l] ≈ δ_kl.
        for k in 0..n {
            for l in 0..n {
                let sum = (0..n).fold(Complex::new(0.0, 0.0), |acc, i| {
                    acc + mat[mat_idx(i, k, n, n, order())].conj()
                        * mat[mat_idx(i, l, n, n, order())]
                });
                let expected = if k == l { 1.0 } else { 0.0 };
                assert!(
                    (sum.re - expected).abs() < 1e-10 && sum.im.abs() < 1e-10,
                    "sector {:?} (V†V)[{k},{l}] = {sum} expected {expected}",
                    g.sector
                );
            }
        }
    }
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
    let h = hermitian_rank2_f64();
    expect_invalid_argument(run(&h, 0), Some("nrow"));
    expect_invalid_argument(run(&h, 2), Some("nrow"));
}
