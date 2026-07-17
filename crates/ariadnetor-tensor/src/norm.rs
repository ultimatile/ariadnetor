//! Shared Frobenius-norm kernel.
//!
//! Both the dense and block-sparse storage halves compute the Frobenius
//! norm over their flat element buffer. The overflow-safe numerics live in
//! [`ariadnetor_core::scale_safe_norm`]; this helper only maps elements to
//! their magnitudes so both storage halves route through one call site.

use ariadnetor_core::{Scalar, scale_safe_norm};

/// Frobenius norm `sqrt(Σ |xᵢ|²)` over a flat buffer, computed with the
/// scaled (overflow-avoiding) accumulation.
///
/// The non-finite contract is [`scale_safe_norm`]'s: any `NaN` element
/// yields `NaN`, and an infinite element yields `inf` (the true norm of
/// an unbounded vector).
pub(crate) fn frobenius_norm<T: Scalar>(data: &[T]) -> T::Real {
    scale_safe_norm(data.iter().map(|x| x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The kernel-level corner cases (extreme magnitudes, NaN stickiness
    // against infinity, zero handling) are pinned by `ariadnetor-core`'s
    // norm tests. The complex-modulus test is the only one exercising
    // behavior unique to this wrapper (the element-to-magnitude mapping);
    // the others re-check a kernel representative through the generic
    // `T: Scalar` path as routing smoke tests.

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
        approx::assert_relative_eq!(n, 5.0_f32.sqrt() * 1e20, max_relative = 1e-6);
    }

    #[test]
    fn nan_propagates() {
        assert!(frobenius_norm::<f64>(&[1.0, f64::NAN, 2.0]).is_nan());
        assert!(frobenius_norm::<f64>(&[f64::INFINITY, f64::NAN]).is_nan());
    }
}
