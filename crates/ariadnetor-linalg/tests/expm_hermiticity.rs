//! The Hermitian and anti-Hermitian exponentials reject an operand that is not
//! (anti-)Hermitian within `sqrt(epsilon)` relative Frobenius tolerance.
//!
//! Each case runs through both public entries, the `DenseHostOps` method and
//! its `*_with_backend` twin, and checks the error message so a case cannot
//! pass on an earlier validation (element type, `nrow`, squareness) firing
//! instead.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{
    DenseHostOps, LinalgError, expm_antihermitian_with_backend, expm_hermitian_with_backend,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseTensor, DenseTensorData, MemoryOrder};
use num_complex::Complex;
use num_traits::{Float, NumCast};

const NOT_HERMITIAN: &str = "expm_hermitian requires a Hermitian matrix";
const NOT_ANTIHERMITIAN: &str = "expm_antihermitian requires an anti-Hermitian matrix";
const NOT_FINITE: &str = "requires an operand with a finite Frobenius norm";

type Out<T> = Result<DenseTensor<T>, LinalgError>;

/// Dense tensor from row-major data, stored column-major as `NativeBackend` expects.
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> DenseTensor<T> {
    let rm = DenseTensorData::from_raw_parts(data, shape, MemoryOrder::RowMajor);
    DenseTensor::from_data(ariadnetor_tensor::reorder_data(
        &rm,
        MemoryOrder::ColumnMajor,
    ))
}

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

/// Both public entries of the Hermitian exponential.
fn hermitian<T: Scalar>(a: &DenseTensor<T>, nrow: usize) -> [Out<T>; 2] {
    [
        a.expm_hermitian(nrow),
        expm_hermitian_with_backend(&NativeBackend::new(), a, nrow),
    ]
}

/// Both public entries of the anti-Hermitian exponential.
fn antihermitian<T: Scalar>(a: &DenseTensor<T>, nrow: usize) -> [Out<T>; 2] {
    [
        a.expm_antihermitian(nrow),
        expm_antihermitian_with_backend(&NativeBackend::new(), a, nrow),
    ]
}

fn assert_rejected<T: Scalar>(results: [Out<T>; 2], substr: &str) {
    for result in results {
        match result {
            Err(LinalgError::InvalidArgument(msg)) => assert!(
                msg.contains(substr),
                "message {msg:?} does not contain {substr:?}"
            ),
            Err(other) => panic!("expected InvalidArgument containing {substr:?}, got {other:?}"),
            Ok(_) => panic!("expected InvalidArgument containing {substr:?}, got Ok"),
        }
    }
}

fn assert_accepted<T: Scalar>(results: [Out<T>; 2]) {
    for result in results {
        if let Err(err) = result {
            panic!("expected Ok, got {err:?}");
        }
    }
}

#[test]
fn hermitian_rejects_complex_symmetric_and_accepts_hermitian() {
    // [[1, i], [i, 1]] equals its transpose but not its adjoint.
    let symmetric = cm(
        vec![c(1.0, 0.0), c(0.0, 1.0), c(0.0, 1.0), c(1.0, 0.0)],
        vec![2, 2],
    );
    assert_rejected(hermitian(&symmetric, 1), NOT_HERMITIAN);
    let herm = cm(
        vec![c(2.0, 0.0), c(1.0, -1.0), c(1.0, 1.0), c(3.0, 0.0)],
        vec![2, 2],
    );
    assert_accepted(hermitian(&herm, 1));
}

#[test]
fn hermitian_rejects_a_non_real_diagonal() {
    let a = cm(
        vec![c(1.0, 1.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        vec![2, 2],
    );
    assert_rejected(hermitian(&a, 1), NOT_HERMITIAN);
}

#[test]
fn hermitian_rejects_a_real_non_symmetric_matrix() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_rejected(hermitian(&a, 1), NOT_HERMITIAN);
}

#[test]
fn antihermitian_rejects_hermitian_and_accepts_antihermitian() {
    // [[0, i], [-i, 0]] is Hermitian, so it passes one check and fails the other.
    let herm = cm(
        vec![c(0.0, 0.0), c(0.0, 1.0), c(0.0, -1.0), c(0.0, 0.0)],
        vec![2, 2],
    );
    assert_rejected(antihermitian(&herm, 1), NOT_ANTIHERMITIAN);
    assert_accepted(hermitian(&herm, 1));
    let anti = cm(
        vec![c(0.0, 1.0), c(1.0, 0.0), c(-1.0, 0.0), c(0.0, 2.0)],
        vec![2, 2],
    );
    assert_accepted(antihermitian(&anti, 1));
}

#[test]
fn antihermitian_rejects_a_real_diagonal_part() {
    let a = cm(
        vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 1.0)],
        vec![2, 2],
    );
    assert_rejected(antihermitian(&a, 1), NOT_ANTIHERMITIAN);
}

/// A Hermitian matrix of norm `sqrt(153)`, about 12.4, with entry `(0, 1)` shifted by
/// `delta` times the norm, so the relative deviation is about `sqrt(2) * delta`.
fn perturbed<R>(delta: f64) -> DenseTensor<Complex<R>>
where
    R: Float,
    Complex<R>: Scalar<Real = R>,
{
    let base = [(6.0, 0.0), (3.0, -3.0), (3.0, 3.0), (9.0, 0.0)];
    let norm = base
        .iter()
        .map(|(re, im): &(f64, f64)| re * re + im * im)
        .sum::<f64>()
        .sqrt();
    let r = |x: f64| <R as NumCast>::from(x).unwrap();
    let mut data: Vec<Complex<R>> = base
        .iter()
        .map(|&(re, im)| Complex::new(r(re), r(im)))
        .collect();
    data[1].re = data[1].re + r(delta * norm);
    cm(data, vec![2, 2])
}

#[test]
fn tolerance_is_sqrt_epsilon_relative_f64() {
    // sqrt(epsilon) is about 1.5e-8 for f64; the cases sit a decade either side.
    assert_accepted(hermitian(&perturbed::<f64>(1e-9), 1));
    assert_rejected(hermitian(&perturbed::<f64>(1e-7), 1), NOT_HERMITIAN);
}

#[test]
fn tolerance_is_sqrt_epsilon_relative_f32() {
    // sqrt(epsilon) is about 3.5e-4 for f32; the cases sit either side of it.
    assert_accepted(hermitian(&perturbed::<f32>(1e-4), 1));
    assert_rejected(hermitian(&perturbed::<f32>(1e-3), 1), NOT_HERMITIAN);
}

#[test]
fn rejection_holds_at_extreme_scales() {
    // The squared entries overflow or underflow in the element type. The
    // rejection must still be the tolerance one, which needs norms that are
    // computed without forming those squares.
    const DEVIATION: &str = "exceeds the tolerance";
    for scale in [1e200_f64, 1e-200] {
        let a = cm(
            vec![scale, 2.0 * scale, 3.0 * scale, 4.0 * scale],
            vec![2, 2],
        );
        assert_rejected(hermitian(&a, 1), DEVIATION);
    }
    for scale in [1e25_f32, 1e-25] {
        let a = cm(
            vec![scale, 2.0 * scale, 3.0 * scale, 4.0 * scale],
            vec![2, 2],
        );
        assert_rejected(hermitian(&a, 1), DEVIATION);
    }
}

#[test]
fn non_finite_entries_are_rejected() {
    // On the diagonal, and off the diagonal, where an infinite entry would make
    // the residual and the bound both infinite.
    for bad in [f64::NAN, f64::INFINITY] {
        for at in [0, 1] {
            let mut data = vec![1.0, 0.0, 0.0, 1.0];
            data[at] = bad;
            assert_rejected(hermitian(&cm(data, vec![2, 2]), 1), NOT_FINITE);
            let mut data = vec![c(0.0, 1.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 1.0)];
            data[at] = c(0.0, bad);
            assert_rejected(antihermitian(&cm(data, vec![2, 2]), 1), NOT_FINITE);
        }
    }
}

#[test]
fn an_overflowing_norm_is_rejected_as_non_finite() {
    // Exactly Hermitian with finite entries, but the Frobenius norm overflows.
    let big = 0.8 * f64::MAX;
    let a = cm(vec![0.0, big, big, 0.0], vec![2, 2]);
    assert_rejected(hermitian(&a, 1), NOT_FINITE);
}

#[test]
fn zero_matrix_is_accepted() {
    assert_accepted(hermitian(&cm(vec![0.0_f64; 4], vec![2, 2]), 1));
    assert_accepted(antihermitian(&cm(vec![c(0.0, 0.0); 4], vec![2, 2]), 1));
}

/// A complex Hermitian 6x6 matrix with distinct, non-symmetric entries.
fn hermitian_6x6() -> Vec<Vec<Complex<f64>>> {
    let upper = |i: usize, j: usize| c(0.1 * (i + 2 * j) as f64, 0.05 * (3 * i + j) as f64);
    (0..6)
        .map(|i| {
            (0..6)
                .map(|j| match i.cmp(&j) {
                    std::cmp::Ordering::Equal => c(i as f64 + 1.0, 0.0),
                    std::cmp::Ordering::Less => upper(i, j),
                    std::cmp::Ordering::Greater => upper(j, i).conj(),
                })
                .collect()
        })
        .collect()
}

/// Rank-4 tensor `[2, 3, 3, 2]` whose row group `(a, b)` and column group
/// `(c, d)` index `m` as `m[3a + b][2c + d]`.
fn grouped(m: &[Vec<Complex<f64>>]) -> DenseTensor<Complex<f64>> {
    let mut data = Vec::with_capacity(36);
    for a in 0..2 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..2 {
                    data.push(m[3 * a + b][2 * c + d]);
                }
            }
        }
    }
    cm(data, vec![2, 3, 3, 2])
}

#[test]
fn rank4_operand_is_checked_on_its_row_column_grouping() {
    let m = hermitian_6x6();
    // Grouping the axes column-major instead, row (a, b) -> a + 2b and column
    // (c, d) -> c + 3d, does not give a Hermitian matrix, so a check that read
    // the wrong grouping would reject this operand.
    let misread = |i: usize, j: usize| m[3 * (i % 2) + i / 2][2 * (j % 3) + j / 3];
    assert!(
        (0..6).any(|i| (0..6).any(|j| (misread(i, j) - misread(j, i).conj()).norm() > 1e-3)),
        "fixture must not be Hermitian under the column-major grouping"
    );
    assert_accepted(hermitian(&grouped(&m), 2));

    // A violation at (5, 4), outside the leading 2x2 block of the grouping.
    let mut bad = m.clone();
    bad[5][4] += c(0.5, 0.0);
    assert_rejected(hermitian(&grouped(&bad), 2), NOT_HERMITIAN);
}

#[test]
fn earlier_validations_take_precedence() {
    // A real element type is rejected before shape or Hermiticity.
    let real_nonsquare = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_rejected(
        antihermitian(&real_nonsquare, 1),
        "requires complex input type",
    );
    // nrow and squareness are checked before Hermiticity, on non-Hermitian data.
    let nonsym = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_rejected(hermitian(&nonsym, 0), "nrow must satisfy");
    assert_rejected(
        hermitian(&real_nonsquare, 1),
        "expm_hermitian requires a square matrix",
    );
    let complex_nonsquare = cm(vec![c(1.0, 2.0); 6], vec![2, 3]);
    assert_rejected(
        antihermitian(&complex_nonsquare, 1),
        "expm_antihermitian requires a square matrix",
    );
}
