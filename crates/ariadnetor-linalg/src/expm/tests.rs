use super::{
    compute_scaling_decision, expm_antihermitian_dense, expm_hermitian_dense, validate_expm_nrow,
};
use crate::error::LinalgError;
use crate::test_util::{RowMajorBackend, assert_invalid_argument};
use ariadnetor_core::Complex;
use ariadnetor_core::backend::MemoryOrder;
use ariadnetor_tensor::DenseTensorData;

/// Entry `(i, j)` of a `6 x 6` complex Hermitian matrix with distinct entries.
fn hermitian_entry(i: usize, j: usize) -> Complex<f64> {
    let upper =
        |i: usize, j: usize| Complex::new(0.1 * (i + 2 * j) as f64, 0.05 * (3 * i + j) as f64);
    match i.cmp(&j) {
        std::cmp::Ordering::Equal => Complex::new(i as f64 + 1.0, 0.0),
        std::cmp::Ordering::Less => upper(i, j),
        std::cmp::Ordering::Greater => upper(j, i).conj(),
    }
}

/// Row-major `[2, 3, 3, 2]` operand whose grouped `6 x 6` matrix is
/// `scale * H` plus `offset` at row 5, column 4, with `H` from
/// [`hermitian_entry`]. The anti-Hermitian test passes `scale = -i`.
fn grouped_operand(scale: Complex<f64>, offset: f64) -> DenseTensorData<Complex<f64>> {
    let mut data: Vec<_> = (0..36)
        .map(|k| hermitian_entry(k / 6, k % 6) * scale)
        .collect();
    data[5 * 6 + 4] += offset;
    DenseTensorData::from_raw_parts(data, vec![2, 3, 3, 2], MemoryOrder::RowMajor)
}

/// Whether `result` failed with the Hermiticity error naming `requirement`.
fn hermiticity_error<T>(result: &Result<T, LinalgError>, requirement: &str) -> bool {
    matches!(result, Err(LinalgError::InvalidArgument(msg)) if msg.contains(requirement))
}

/// On a row-major backend the check reads the grouped matrix from the
/// row-major buffer. `eigh_dense` is column-major only on this backend, so an
/// operand that passes the check fails afterwards with a different error.
#[test]
fn hermiticity_check_on_row_major_backend() {
    let backend = RowMajorBackend::new();
    let one = Complex::new(1.0, 0.0);
    const HERMITIAN: &str = "requires a Hermitian matrix";
    const ANTI: &str = "requires an anti-Hermitian matrix";

    let bad = expm_hermitian_dense(&backend, &grouped_operand(one, 0.5), 2);
    assert_invalid_argument(bad, HERMITIAN);
    let good = expm_hermitian_dense(&backend, &grouped_operand(one, 0.0), 2);
    assert!(
        !hermiticity_error(&good, HERMITIAN),
        "got {:?}",
        good.as_ref().err()
    );

    // `-i H` is anti-Hermitian; the same offset breaks it.
    let minus_i = Complex::new(0.0, -1.0);
    let bad = expm_antihermitian_dense(&backend, &grouped_operand(minus_i, 0.5), 2);
    assert_invalid_argument(bad, ANTI);
    let good = expm_antihermitian_dense(&backend, &grouped_operand(minus_i, 0.0), 2);
    assert!(
        !hermiticity_error(&good, ANTI),
        "got {:?}",
        good.as_ref().err()
    );
}

#[test]
fn validate_expm_nrow_rejects_zero() {
    assert!(validate_expm_nrow(0, 2).is_err());
}

#[test]
fn validate_expm_nrow_rejects_equal_to_rank() {
    assert!(validate_expm_nrow(2, 2).is_err());
}

#[test]
fn validate_expm_nrow_rejects_greater_than_rank() {
    assert!(validate_expm_nrow(3, 2).is_err());
}

#[test]
fn validate_expm_nrow_accepts_valid() {
    assert!(validate_expm_nrow(1, 3).is_ok());
    assert!(validate_expm_nrow(2, 3).is_ok());
}

/// `norm = 1.5 * 2^62` with `theta = 1` forces `s = ceil(log2(1.5 * 2^62)) = 63`,
/// which selects the doubling-loop branch (`s > 62`). Verifies both the
/// returned `s` and that the matrix is scaled by `1 / 2^63`. Kills the
/// `v + v -> v - v` (yields 0) and `v + v -> v * v` (yields 1) mutations
/// on the doubling accumulator.
#[test]
fn compute_scaling_decision_above_shift_threshold() {
    let theta = 1.0_f64;
    let norm = (1u64 << 62) as f64 * 1.5;
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let (b, s) = compute_scaling_decision::<f64>(a, norm, theta);
    assert_eq!(s, 63);
    let expected_factor = 1.0_f64 / 2.0_f64.powi(63);
    for (i, &x) in b.data().iter().enumerate() {
        let expected = (i + 1) as f64 * expected_factor;
        assert!(
            (x - expected).abs() < 1e-30,
            "b[{i}] = {x}, expected {expected}"
        );
    }
}

/// `s == 0` early-return path: when `norm <= theta`, the helper returns
/// the input matrix without copying. Asserting pointer identity (rather
/// than just data equality) is what kills the `== with !=` mutation:
/// under that mutation the early return is skipped and the scaling path
/// runs with `s = 0`, producing a fresh allocation whose elements equal
/// the input but whose storage pointer differs.
#[test]
fn compute_scaling_decision_zero_steps_returns_input_unchanged() {
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let a_ptr = a.data().as_ptr();
    let (b, s) = compute_scaling_decision::<f64>(a, 0.5, 1.0);
    assert_eq!(s, 0);
    assert_eq!(b.data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        b.data().as_ptr(),
        a_ptr,
        "s = 0 path must return input without copying"
    );
}
