//! Truncation-error targets of the truncated SVD and the keep count they
//! select, shared by the dense and block-sparse kernels.

use std::cmp::Ordering;

use ariadnetor_core::{NormAccumulator, combine_norms};

/// Bound on the singular values a truncated SVD may discard.
///
/// Singular values are discarded from the smallest upward while the
/// Frobenius norm of the discarded ones stays within the bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncErrTarget {
    /// The discarded norm is at most this value.
    Absolute(f64),
    /// The discarded norm is at most this fraction of the Frobenius norm of
    /// the matrix being truncated; for a block-sparse matrix, of its norm
    /// over all sectors together. The fraction is a ratio of norms, not of
    /// squared norms.
    Relative(f64),
}

impl TruncErrTarget {
    /// The bound or fraction the variant carries.
    pub(crate) fn value(self) -> f64 {
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
    match target {
        TruncErrTarget::Absolute(budget) => kept_within(len, &value, budget),
        TruncErrTarget::Relative(fraction) => {
            // `NormAccumulator` holds the norm as `scale * sqrt(sumsq)` with
            // `scale` the largest value and `sumsq` at most `len + 1`, so the
            // pair stays representable where the norm itself overflows.
            let mut acc = NormAccumulator::new();
            for i in 0..len {
                acc.push(value(i));
            }
            let scale = acc.scale();
            if !scale.is_finite() {
                // An infinite or NaN singular value leaves no finite norm to
                // measure the fraction against.
                return len;
            }
            let ratio_budget = fraction * acc.sumsq().sqrt();
            let budget = scale * ratio_budget;
            if budget.is_finite() {
                // Compared in the values' own units, so a value far below
                // `scale` is never rounded away before it is compared.
                kept_within(len, &value, budget)
            } else {
                // The budget exceeds the type's range only when `ratio_budget`
                // is at least 1 in units of `scale`; a ratio that underflows
                // there lies far inside it either way.
                kept_within(len, |i| value(i) / scale, ratio_budget)
            }
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
