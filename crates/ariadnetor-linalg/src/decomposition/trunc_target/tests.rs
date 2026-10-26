use super::{TruncErrTarget, kept_by_target};

fn kept(values: &[f64], target: TruncErrTarget) -> usize {
    kept_by_target(values.len(), |i| values[i], target)
}

/// `[4, 2, 1]` has norm `sqrt(21) ≈ 4.583`. An absolute bound of 0.25 keeps
/// all three; the same number as a fraction is a budget of about 1.146,
/// which discards the 1 and stops at `sqrt(5) ≈ 2.236`.
#[test]
fn absolute_and_relative_read_the_same_number_differently() {
    let s = [4.0, 2.0, 1.0];
    assert_eq!(kept(&s, TruncErrTarget::Absolute(0.25)), 3);
    assert_eq!(kept(&s, TruncErrTarget::Relative(0.25)), 2);
    assert_eq!(kept(&s, TruncErrTarget::Relative(0.2)), 3);
    assert_eq!(kept(&s, TruncErrTarget::Absolute(1.0)), 2);
    assert_eq!(kept(&s, TruncErrTarget::Absolute(2.0)), 2);
}

#[test]
fn relative_keep_count_is_invariant_under_scaling() {
    for scale in [1e-300, 1.0, 1e300] {
        let s = [4.0 * scale, 2.0 * scale, 1.0 * scale];
        assert_eq!(kept(&s, TruncErrTarget::Relative(0.25)), 2, "scale {scale}");
    }
}

/// The norm overflows `f64`, so `r * norm` would be an infinite budget that
/// discards everything; the budget in units of the largest value is about
/// `2.1e298`, below the `1e300` it must keep.
#[test]
fn relative_survives_an_overflowing_norm() {
    let s = [1.5e308, 1.5e308, 1e300];
    assert_eq!(kept(&s, TruncErrTarget::Relative(1e-10)), 3);
}

/// `1e-200 / 1e200` underflows to zero, so comparing ratios would discard
/// it against a zero budget; compared in its own units it stays.
#[test]
fn relative_zero_keeps_a_value_far_below_the_largest() {
    let s = [1e200, 1e-200];
    assert_eq!(kept(&s, TruncErrTarget::Relative(0.0)), 2);
}

/// Four values of `1.7e308` have norm `3.4e308` and budget `2.04e308`, past
/// `f64::MAX`. In units of the largest value the budget is 1.2: one value
/// (ratio 1) fits, two (ratio `sqrt(2)`) do not.
#[test]
fn relative_compares_in_ratios_when_the_budget_overflows() {
    let s = [1.7e308; 4];
    assert_eq!(kept(&s, TruncErrTarget::Relative(0.6)), 3);
}

#[test]
fn relative_discards_nothing_against_a_non_finite_value() {
    assert_eq!(
        kept(&[f64::INFINITY, 1.0], TruncErrTarget::Relative(0.5)),
        2
    );
    assert_eq!(kept(&[1.0, f64::NAN], TruncErrTarget::Relative(0.5)), 2);
}

#[test]
fn relative_discards_an_all_zero_spectrum() {
    assert_eq!(kept(&[0.0; 3], TruncErrTarget::Relative(0.1)), 0);
}
