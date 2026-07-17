//! Unit tests for the private estimator helper. The sweep-level behavior
//! (stopping decisions, error surfacing, scale invariance) is covered by
//! the crate's integration tests; these pin the `Option` contract of
//! [`leave_one_out_estimate`] at the function boundary, where the
//! uninformative regimes are constructed directly instead of through a
//! full sweep.

use approx::assert_relative_eq;

use super::leave_one_out_estimate;

#[test]
fn finite_row_norms_match_the_closed_form() {
    // err = sqrt((1/p) * sum ||g_i||^-2) with rows [1, 2]:
    // sqrt((1 + 1/4) / 2) = sqrt(0.625).
    let err = leave_one_out_estimate::<f64>(&[1.0, 2.0]).expect("finite input yields an estimate");
    assert_relative_eq!(err, 0.625_f64.sqrt(), max_relative = 1e-15);
}

#[test]
fn overflowed_row_norm_is_uninformative() {
    // An inf row norm is an overflow artifact of the maintained inverse;
    // mapping it through recip() would contribute an exact zero and
    // collapse the estimate toward false convergence.
    assert!(leave_one_out_estimate::<f64>(&[1.0, f64::INFINITY]).is_none());
    assert!(leave_one_out_estimate::<f64>(&[f64::INFINITY, f64::INFINITY]).is_none());
}

#[test]
fn nan_row_norm_is_uninformative() {
    assert!(leave_one_out_estimate::<f64>(&[1.0, f64::NAN]).is_none());
}

#[test]
fn saturated_reciprocal_accumulation_is_uninformative() {
    // Every row norm is finite, but their reciprocals sit near the top of
    // the f32 range: each recip (~3.3e38) is representable, while the
    // finished reciprocal norm (~4.7e38) exceeds f32::MAX and saturates.
    // A saturated estimate must not enter the stopping comparison, where
    // `inf <= inf` against a saturated threshold would read as converged.
    let r = 3.0e-39_f32;
    assert!(r.recip().is_finite(), "each reciprocal is representable");
    assert!(
        (r.recip() * 2.0_f32.sqrt()).is_infinite(),
        "their combined norm is not"
    );
    assert!(leave_one_out_estimate::<f32>(&[r, r]).is_none());
}
