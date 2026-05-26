use super::{compute_scaling_decision, validate_expm_nrow};
use arnet_core::backend::MemoryOrder;
use arnet_tensor::DenseTensorData;

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
