//! Shared Frobenius-norm kernel.
//!
//! Both the dense and block-sparse storage halves compute the Frobenius
//! norm over their flat element buffer. The overflow-safe numerics live in
//! [`ariadnetor_core::NormAccumulator`]; this helper only maps elements to
//! their magnitudes so both storage halves route through one call site.

use ariadnetor_core::{NormAccumulator, Scalar};

/// Frobenius norm `sqrt(Σ |xᵢ|²)` over a flat buffer, computed with the
/// scaled (overflow-avoiding) accumulation.
///
/// Reproduces the naive loop's non-finite behavior exactly: any `NaN`
/// element yields `NaN`, and an infinite element yields `inf` (the true
/// norm of an unbounded vector).
pub(crate) fn frobenius_norm<T: Scalar>(data: &[T]) -> T::Real {
    let mut acc = NormAccumulator::new();
    for &x in data {
        acc.push(x.abs());
    }
    acc.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The kernel-level cases (extreme magnitudes, NaN stickiness against
    // infinity, zero handling) are pinned by `ariadnetor-core`'s norm
    // tests; these cover the wrapper's element-to-magnitude mapping.

    #[test]
    fn matches_naive_for_moderate_input() {
        // sqrt(3² + 4²) = 5, exactly representable.
        assert_eq!(frobenius_norm::<f64>(&[3.0, 4.0]), 5.0);
        assert_eq!(frobenius_norm::<f32>(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn complex_elements_accumulate_their_modulus() {
        use ariadnetor_core::Complex;
        // |3 + 4i| = 5, |0 - 12i| = 12, sqrt(5² + 12²) = 13.
        let data = [Complex::new(3.0_f64, 4.0), Complex::new(0.0, -12.0)];
        assert_eq!(frobenius_norm(&data), 13.0);
    }

    #[test]
    fn extreme_magnitude_stays_finite() {
        // The naive `sqrt(Σ |x|²)` overflows here: 1e20² = 1e40 exceeds
        // f32::MAX (~3.4e38) and saturates to inf. The scaled accumulation
        // stays finite through the wrapper as well.
        let n = frobenius_norm::<f32>(&[1e20, 2e20]);
        assert!(n.is_finite(), "expected finite norm, got {n}");
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
}
