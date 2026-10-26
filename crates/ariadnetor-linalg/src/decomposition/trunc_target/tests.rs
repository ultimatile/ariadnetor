use super::{TruncErrTarget, kept_by_target};

fn kept(values: &[f64], target: TruncErrTarget) -> usize {
    kept_by_target(values.len(), |i| values[i], target)
}

/// The smallest positive `f64`.
const MIN_SUBNORMAL: f64 = f64::from_bits(1);

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

/// A relative target decides on the spectrum's shape alone, so scaling every
/// value by `2^k` must not change the keep count anywhere in the `f64`
/// range. The spectra hold small integers, so `n * 2^k` is exact from the
/// smallest positive `f64` (`k = -1074`) up to `k = 1020`, where the largest
/// value, 10, stays below `f64::MAX`.
#[test]
fn relative_keep_count_is_invariant_under_exact_binary_scaling() {
    let spectra: [&[f64]; 4] = [
        &[1.0, 1.0],
        &[4.0, 2.0, 1.0],
        &[10.0, 7.0, 1.0],
        &[5.0, 4.0, 3.0, 1.0],
    ];
    for spectrum in spectra {
        for fraction in [0.05, 0.1, 0.25, 0.5, 0.75] {
            let target = TruncErrTarget::Relative(fraction);
            let expected = kept(spectrum, target);
            for k in (-1074..=1020).step_by(7).chain([-1074, 1020]) {
                let scaled: Vec<f64> = spectrum.iter().map(|&n| libm::scalbn(n, k)).collect();
                assert_eq!(
                    kept(&scaled, target),
                    expected,
                    "{spectrum:?} * 2^{k}, {fraction}"
                );
            }
        }
    }
}

/// The norm overflows `f64`, while the budget, about `2.1e298`, does not
/// and keeps the `1e300`; forming `r * norm` first would give an infinite
/// budget that discards everything.
#[test]
fn relative_survives_an_overflowing_norm() {
    let s = [1.5e308, 1.5e308, 1e300];
    assert_eq!(kept(&s, TruncErrTarget::Relative(1e-10)), 3);
}

/// A zero budget discards only zeros, however small the other values are.
#[test]
fn relative_zero_keeps_a_value_far_below_the_largest() {
    let s = [1e200, 1e-200];
    assert_eq!(kept(&s, TruncErrTarget::Relative(0.0)), 2);
}

/// Four values of `1.7e308` have norm `3.4e308` and budget `2.04e308`, past
/// `f64::MAX`: one value (`1.7e308`) fits, two (`sqrt(2) * 1.7e308`) do not.
#[test]
fn relative_handles_a_budget_past_the_largest_f64() {
    let s = [1.7e308; 4];
    assert_eq!(kept(&s, TruncErrTarget::Relative(0.6)), 3);
}

/// `[1.6e308, 1.2e308, 1e300]` has norm `2e308`, so `Relative(0.95)` is a
/// budget of `1.9e308`, past `f64::MAX`. The two smaller values (discarded
/// norm `1.2e308`) fit, the largest (discarded norm `2e308`) does not.
#[test]
fn relative_budget_past_the_largest_f64_on_distinct_values() {
    let s = [1.6e308, 1.2e308, 1e300];
    assert_eq!(kept(&s, TruncErrTarget::Relative(0.95)), 1);
}

/// Two copies of the smallest positive `f64` have norm `sqrt(2)` of it, so
/// `Relative(0.5)` is a budget of about 0.707 of that value, below the
/// value itself: discarding one would exceed the budget.
#[test]
fn relative_handles_a_subnormal_budget() {
    let m = MIN_SUBNORMAL;
    assert_eq!(kept(&[m, m], TruncErrTarget::Relative(0.5)), 2);
}

/// With the smallest positive `f64` as the fraction, three values of
/// `1e300` give a budget of `sqrt(3) * 1e300 * MIN_SUBNORMAL ≈ 8.557e-24`,
/// below `9e-24`. Rounding the fraction times `sqrt(3)` in the subnormal
/// range first would raise the budget to about `9.88e-24` and discard it.
#[test]
fn relative_handles_a_subnormal_fraction() {
    let s = [1e300, 1e300, 1e300, 9e-24];
    assert_eq!(kept(&s, TruncErrTarget::Relative(MIN_SUBNORMAL)), 4);
}

/// `[10, 14 * MIN_SUBNORMAL]` under `Relative(MIN_SUBNORMAL)` has a budget
/// of `10 * MIN_SUBNORMAL`, below the smaller value. Comparing in units of
/// the largest value instead would round the value, `1.4 * MIN_SUBNORMAL`
/// in those units, to `MIN_SUBNORMAL`, which is the budget there, and
/// discard it.
#[test]
fn relative_compares_a_subnormal_value_without_rounding_it() {
    let s = [10.0, 14.0 * MIN_SUBNORMAL];
    assert_eq!(kept(&s, TruncErrTarget::Relative(MIN_SUBNORMAL)), 2);
}

#[test]
fn no_target_discards_from_a_non_finite_spectrum() {
    for target in [TruncErrTarget::Absolute(0.5), TruncErrTarget::Relative(0.5)] {
        assert_eq!(kept(&[f64::INFINITY, 1.0], target), 2, "{target:?}");
        assert_eq!(kept(&[1.0, f64::NAN], target), 2, "{target:?}");
    }
}

#[test]
fn relative_discards_an_all_zero_spectrum() {
    assert_eq!(kept(&[0.0; 3], TruncErrTarget::Relative(0.1)), 0);
}
