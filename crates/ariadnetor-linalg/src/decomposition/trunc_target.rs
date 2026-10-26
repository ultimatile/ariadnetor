//! Truncation-error targets of the truncated SVD and the keep count they
//! select, shared by the dense and block-sparse kernels.

use std::cmp::Ordering;

use ariadnetor_core::{NormAccumulator, combine_norms};

/// Bound on the singular values a truncated SVD may discard.
///
/// The target discards singular values from the smallest upward while the
/// Frobenius norm of the discarded ones stays within the bound, and never
/// discards the largest one; a `chi_max` in the same
/// [`TruncSvdParams`](super::TruncSvdParams) can discard more. Neither
/// variant discards anything from a spectrum holding a value that is not
/// finite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncErrTarget {
    /// The bound is this value.
    Absolute(f64),
    /// The bound is this fraction of the Frobenius norm of the matrix being
    /// truncated; for a block-sparse matrix, of its norm over all sectors
    /// together. The fraction is a ratio of norms, not of squared norms.
    Relative(f64),
}

impl TruncErrTarget {
    /// The bound or fraction the variant carries.
    pub fn value(self) -> f64 {
        match self {
            Self::Absolute(v) | Self::Relative(v) => v,
        }
    }
}

/// Number of leading singular values `target` keeps out of the `len`
/// values `value(0) >= value(1) >= ...`; zero when it would discard them
/// all, which the caller floors.
///
/// `target` must have passed [`TruncSvdParams::validate`](super::TruncSvdParams::validate).
pub(crate) fn kept_by_target(
    len: usize,
    value: impl Fn(usize) -> f64,
    target: TruncErrTarget,
) -> usize {
    debug_assert!((1..len).all(|i| value(i - 1).partial_cmp(&value(i)) != Some(Ordering::Less)));
    debug_assert!(target.value().is_finite() && target.value() >= 0.0);
    // `NormAccumulator` holds the norm as `scale * sqrt(sumsq)` with `scale`
    // the largest value and `sumsq` at most `len + 1`, so the pair stays
    // representable where the norm itself overflows.
    let mut acc = NormAccumulator::new();
    for i in 0..len {
        acc.push(value(i));
    }
    let scale = acc.scale();
    if !scale.is_finite() {
        // An infinite or NaN singular value leaves the discarded norm, and
        // the matrix norm, without a finite value to compare.
        return len;
    }
    match target {
        TruncErrTarget::Absolute(budget) => kept_within(len, &value, budget),
        TruncErrTarget::Relative(fraction) => {
            if fraction == 0.0 || scale == 0.0 {
                return kept_within(len, &value, 0.0);
            }
            // The budget `fraction * scale * sqrt(sumsq)` can leave the normal
            // range at either end, where rounding it misjudges the values
            // near it. `fraction` and `scale` split exactly into mantissas in
            // `[1/2, 1)` times powers of two, and `sqrt(sumsq)` lies in
            // `[1, sqrt(len + 1)]`, so the budget is `scaled_budget *
            // 2^exponent` with `scaled_budget` at least 1/4. The values are
            // multiplied by `2^-exponent`, and the norm of the discarded ones
            // is folded from those products and compared with
            // `scaled_budget`; each product is exact unless it leaves the
            // normal range, and then it lies far below or far above
            // `scaled_budget`.
            let (fraction_m, fraction_e) = libm::frexp(fraction);
            let (scale_m, scale_e) = libm::frexp(scale);
            let scaled_budget = fraction_m * scale_m * acc.sumsq().sqrt();
            debug_assert!(scaled_budget >= 0.25);
            let exponent = fraction_e + scale_e;
            kept_within(len, |i| libm::scalbn(value(i), -exponent), scaled_budget)
        }
    }
}

/// Fold the discarded norm from the smallest value upward, scale-safely,
/// and stop at the first value that would take it past `budget`.
fn kept_within(len: usize, value: impl Fn(usize) -> f64, budget: f64) -> usize {
    let mut discarded = 0.0_f64;
    let mut kept = len;
    for i in (0..len).rev() {
        let candidate = combine_norms(discarded, value(i));
        if candidate > budget {
            break;
        }
        discarded = candidate;
        kept = i;
    }
    kept
}

#[cfg(test)]
mod tests;
