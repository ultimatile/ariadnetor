//! Shared Frobenius-norm kernel.
//!
//! Both the dense and block-sparse storage halves compute the
//! Frobenius norm over their flat element buffer. Routing them through
//! one helper keeps the numerics in a single place: a naive
//! `sqrt(Σ |x|²)` overflows in `f32` once any `|x|` nears
//! `sqrt(T::Real::MAX)` (~`1.8e19`), because the squared term saturates
//! to `inf` before the sum. This kernel uses the scaled algorithm
//! (BLAS `dnrm2` / Higham, *Accuracy and Stability of Numerical
//! Algorithms*, 2nd ed., §27.5): it tracks a running maximum `scale`
//! and accumulates `(|x| / scale)²`, so no intermediate overflows until
//! the result itself would.

use ariadnetor_core::Scalar;
use num_traits::{Float, One, Zero};

/// Frobenius norm `sqrt(Σ |xᵢ|²)` over a flat buffer, computed with the
/// scaled (overflow-avoiding) algorithm.
///
/// Reproduces the naive loop's non-finite behavior exactly: any `NaN`
/// element yields `NaN`, and an infinite element yields `inf` (the true
/// norm of an unbounded vector).
pub(crate) fn frobenius_norm<T: Scalar>(data: &[T]) -> T::Real {
    let zero = T::Real::zero();
    let one = T::Real::one();
    let mut scale = zero;
    let mut sumsq = one;
    for &x in data {
        let a = x.abs();
        if a.is_nan() {
            return a; // NaN propagates, matching the naive loop.
        }
        if scale < a {
            // New running maximum. `scale < a` implies `a > 0`; if `a`
            // is `+inf` with a finite `scale`, `scale / a` underflows to
            // 0 rather than producing a NaN.
            let r = scale / a;
            sumsq = one + sumsq * r * r;
            scale = a;
        } else if a > zero && scale.is_finite() {
            // `0 < a <= scale` with a finite `scale`, so `a / scale` is
            // finite.
            let r = a / scale;
            sumsq = sumsq + r * r;
        }
        // Otherwise `a == 0` (contributes nothing), or `a > 0` with an
        // infinite `scale` (the infinity already dominates the result) —
        // skip either way.
    }
    scale * sumsq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive_for_moderate_input() {
        // sqrt(3² + 4²) = 5, exactly representable.
        assert_eq!(frobenius_norm::<f64>(&[3.0, 4.0]), 5.0);
        assert_eq!(frobenius_norm::<f32>(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn leading_and_interior_zeros_are_skipped() {
        assert_eq!(frobenius_norm::<f64>(&[0.0, 0.0, 3.0, 4.0]), 5.0);
        assert_eq!(frobenius_norm::<f64>(&[3.0, 0.0, 4.0]), 5.0);
    }

    #[test]
    fn zero_vector_has_zero_norm() {
        assert_eq!(frobenius_norm::<f64>(&[0.0, 0.0]), 0.0);
        assert_eq!(frobenius_norm::<f64>(&[]), 0.0);
    }

    #[test]
    fn f32_extreme_magnitude_stays_finite() {
        // The naive `sqrt(Σ |x|²)` overflows here: 1e20² = 1e40 exceeds
        // f32::MAX (~3.4e38) and saturates to inf. The scaled algorithm
        // stays finite.
        let n = frobenius_norm::<f32>(&[1e20, 2e20]);
        assert!(n.is_finite(), "expected finite norm, got {n}");
        // sqrt(1e40 + 4e40) = sqrt(5) * 1e20 ≈ 2.2360680e20.
        let expected = 5.0_f32.sqrt() * 1e20;
        assert!(
            (n - expected).abs() / expected < 1e-6,
            "expected ~{expected}, got {n}"
        );
    }

    #[test]
    fn nan_propagates() {
        assert!(frobenius_norm::<f64>(&[1.0, f64::NAN, 2.0]).is_nan());
        assert!(frobenius_norm::<f64>(&[f64::INFINITY, f64::NAN]).is_nan());
    }

    #[test]
    fn infinity_yields_infinite_norm() {
        // Single and repeated infinities both give inf, as the naive
        // loop does (the true norm of an unbounded vector is inf).
        assert!(frobenius_norm::<f64>(&[f64::INFINITY]).is_infinite());
        assert!(frobenius_norm::<f64>(&[f64::INFINITY, f64::INFINITY]).is_infinite());
        assert!(frobenius_norm::<f64>(&[1.0, f64::INFINITY, 2.0]).is_infinite());
        assert!(frobenius_norm::<f64>(&[f64::NEG_INFINITY, 3.0]).is_infinite());
    }
}
