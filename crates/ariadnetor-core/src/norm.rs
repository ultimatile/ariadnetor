//! Scale-safe sum-of-squares accumulation.
//!
//! Several functionals across the workspace reduce to `sqrt(Σ |xᵢ|²)` over
//! runtime-scale data: Frobenius norms over element buffers, row norms of
//! triangular inverses, and reciprocal-based error estimates. A naive
//! `sqrt(Σ |x|²)` overflows once any `|x|` nears `sqrt(R::MAX)` (~`1.8e19`
//! in `f32`), because the squared term saturates to `inf` before the sum.
//! This module hosts the shared overflow-safe kernel for that functional:
//! the standard scaled sum-of-squares algorithm (the approach of the BLAS
//! `dnrm2` / LAPACK `dlassq` reference routines). It tracks a running
//! maximum `scale` and accumulates `(|x| / scale)²`, so no intermediate
//! overflows until the result itself would.
//!
//! NaN contract: any NaN entering the accumulation yields a NaN result.
//! This is deliberately stricter than chained `hypot`, whose IEEE contract
//! lets an infinity dominate a NaN (`hypot(inf, NaN) = inf`), so an
//! undefined element cannot be masked by an unbounded one.

use num_traits::Float;

use crate::scalar::Scalar;

/// Scale-safe accumulator state for `sqrt(Σ |xᵢ|²)` over pushed values.
///
/// `(scale, sumsq)` represents the norm of everything pushed so far as
/// `scale * sqrt(sumsq)`, where `scale` is the running maximum magnitude
/// and `sumsq`, once anything nonzero has been pushed, the sum of squared
/// ratios `(|x| / scale)²` (while `scale` is zero it idles at one, keeping
/// the represented norm zero). Each ratio is at most 1, so `sumsq` is then
/// bounded by the push count and the accumulation overflows only when the
/// result itself is unrepresentable.
///
/// The pair itself is exposed (see [`Self::scale`] / [`Self::sumsq`])
/// because it stays representable past the point where the finished norm
/// saturates: with every pushed value finite, `scale` is their running
/// maximum (finite by construction) and `sumsq` is at most the push count
/// plus one, so a consumer comparing against the represented value can
/// keep working on the `(scale, sumsq)` form when
/// `scale * sqrt(sumsq)` itself would overflow to infinity. This is the
/// same representation the LAPACK `dlassq` routine hands back to its
/// callers.
#[derive(Debug, Clone)]
pub struct NormAccumulator<R: Float> {
    /// Running maximum magnitude; NaN once a NaN has been pushed.
    scale: R,
    /// Sum of squared ratios against `scale`.
    sumsq: R,
}

impl<R: Float> Default for NormAccumulator<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Float> NormAccumulator<R> {
    /// Empty accumulation; its finished norm is zero.
    #[inline]
    pub fn new() -> Self {
        // The initial `sumsq` is unobservable: the first nonzero push
        // rescales it away (its ratio against the zero scale vanishes),
        // and until then `finish` multiplies it by the zero `scale`. One
        // is the `dlassq` convention.
        Self {
            scale: R::zero(),
            sumsq: R::one(),
        }
    }

    /// Accumulates `|value|` into the represented sum of squares.
    ///
    /// A NaN is sticky: it overwrites `scale`, after which neither
    /// accumulation branch can fire — the max-update comparison is false
    /// against a NaN `scale`, and the finite-scale test rejects it — so
    /// the finished norm is NaN regardless of later pushes, including
    /// infinite ones.
    #[inline]
    pub fn push(&mut self, value: R) {
        let a = value.abs();
        if a.is_nan() {
            self.scale = a;
        } else if self.scale < a {
            // New running maximum. `scale < a` implies `a > 0`; if `a` is
            // `+inf` with a finite `scale`, `scale / a` underflows to 0
            // rather than producing a NaN.
            let r = self.scale / a;
            self.sumsq = R::one() + self.sumsq * r * r;
            self.scale = a;
        } else if a > R::zero() && self.scale.is_finite() {
            // `0 < a <= scale` with a finite `scale`, so `a / scale` is
            // finite.
            let r = a / self.scale;
            self.sumsq = self.sumsq + r * r;
        }
        // Otherwise `a == 0` (contributes nothing), `a > 0` with an
        // infinite `scale` (the infinity already dominates the result), or
        // `scale` is NaN (sticky) — skip either way.
    }

    /// Accumulates a scalar element by its real and imaginary components
    /// (`|z|² = re² + im²`), so the represented value is the Frobenius
    /// norm of the elements pushed this way. Component-wise entry is
    /// load-bearing, not a convenience: a complex modulus (`hypot`) can
    /// overflow to `inf` even when both components are finite, which
    /// would poison the scale exactly where the components themselves
    /// are still representable. Real scalars contribute a vanishing
    /// imaginary part, which the accumulation skips.
    #[inline]
    pub fn push_scalar<T: Scalar<Real = R>>(&mut self, value: T) {
        self.push(value.re());
        self.push(value.im());
    }

    /// The accumulated norm `sqrt(Σ |xᵢ|²)`.
    ///
    /// This finishes the `(scale, sumsq)` representation into one scalar,
    /// which saturates to infinity when the true norm exceeds the real
    /// type's range even though the representation itself is still exact;
    /// consumers that must stay meaningful there compare against
    /// [`Self::scale`] and [`Self::sumsq`] instead.
    #[inline]
    fn finish(&self) -> R {
        self.scale * self.sumsq.sqrt()
    }

    /// The running maximum pushed magnitude — the `scale` half of the
    /// represented norm `scale * sqrt(sumsq)`.
    ///
    /// Finite whenever every pushed value was finite; zero exactly when
    /// nothing nonzero (and no NaN) has been pushed, which makes it the
    /// saturation-free zero test for the accumulation. Infinite after an
    /// infinite push, NaN (sticky) after a NaN push.
    #[inline]
    pub fn scale(&self) -> R {
        self.scale
    }

    /// The sum of squared ratios against [`Self::scale`] — the `sumsq`
    /// half of the represented norm `scale * sqrt(sumsq)`.
    ///
    /// With every pushed value finite this is bounded by the push count
    /// plus one (each ratio is at most 1, and the initial value is one),
    /// so it never saturates on its own. It is meaningful only alongside
    /// its `scale`: while `scale` is zero the initial one is unobservable
    /// by construction, and after an infinite or NaN push the pair no
    /// longer carries the represented norm (the degenerate `scale` alone
    /// does).
    #[inline]
    pub fn sumsq(&self) -> R {
        self.sumsq
    }
}

/// Norm `sqrt(Σ |xᵢ|²)` of the yielded values, computed with the scaled
/// (overflow-avoiding) accumulation.
///
/// Values enter by magnitude, so signs are ignored; callers with complex
/// elements map them to their modulus first. Any NaN yields NaN — even
/// alongside infinities — and an infinite value yields `inf` (the true
/// norm of an unbounded vector). An empty iterator yields zero.
pub fn scale_safe_norm<R: Float>(values: impl IntoIterator<Item = R>) -> R {
    let mut acc = NormAccumulator::new();
    for v in values {
        acc.push(v);
    }
    acc.finish()
}

/// Combines two already-finished norms into `sqrt(a² + b²)`.
///
/// Concatenating two vectors combines their norms by the Pythagorean
/// identity, and feeding a finished norm back in as a single element is
/// the same operation, so this routes through the same accumulation to
/// keep the overflow and NaN contracts identical to [`scale_safe_norm`].
pub fn combine_norms<R: Float>(a: R, b: R) -> R {
    scale_safe_norm([a, b])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn matches_naive_for_moderate_input() {
        // sqrt(3² + 4²) = 5, exactly representable.
        assert_eq!(scale_safe_norm([3.0_f64, 4.0]), 5.0);
        assert_eq!(scale_safe_norm([3.0_f32, 4.0]), 5.0);
    }

    #[test]
    fn sign_is_ignored() {
        assert_eq!(scale_safe_norm([-3.0_f64, 4.0]), 5.0);
    }

    #[test]
    fn leading_and_interior_zeros_are_skipped() {
        assert_eq!(scale_safe_norm([0.0_f64, 0.0, 3.0, 4.0]), 5.0);
        assert_eq!(scale_safe_norm([3.0_f64, 0.0, 4.0]), 5.0);
    }

    #[test]
    fn empty_and_zero_accumulations_finish_at_zero() {
        assert_eq!(scale_safe_norm(std::iter::empty::<f64>()), 0.0);
        assert_eq!(scale_safe_norm([0.0_f64, 0.0]), 0.0);
    }

    #[test]
    fn f32_extreme_magnitude_stays_finite() {
        // The naive `sqrt(Σ |x|²)` overflows here: 1e20² = 1e40 exceeds
        // f32::MAX (~3.4e38) and saturates to inf. The scaled algorithm
        // stays finite.
        let n = scale_safe_norm([1e20_f32, 2e20]);
        // sqrt(1e40 + 4e40) = sqrt(5) * 1e20 ≈ 2.2360680e20.
        assert_relative_eq!(n, 5.0_f32.sqrt() * 1e20, max_relative = 1e-6);
    }

    #[test]
    fn f64_extreme_magnitude_stays_finite() {
        // Mirror of the f32 case one exponent-range up: 1e200² = 1e400
        // exceeds f64::MAX (~1.8e308) and saturates to inf under the naive
        // sum.
        let n = scale_safe_norm([1e200_f64, 2e200]);
        assert_relative_eq!(n, 5.0_f64.sqrt() * 1e200, max_relative = 1e-12);
    }

    #[test]
    fn reciprocal_scale_accumulation_stays_finite() {
        // Reciprocals of tiny norms land near the top of the exponent
        // range; the scaled accumulation must survive where squaring
        // (1e300² = 1e600) cannot.
        let n = scale_safe_norm([1e-300_f64, 2e-300].map(f64::recip));
        assert_relative_eq!(n, 1.25_f64.sqrt() * 1e300, max_relative = 1e-12);
    }

    #[test]
    fn nan_propagates() {
        assert!(scale_safe_norm([1.0_f64, f64::NAN, 2.0]).is_nan());
        // Both interleavings with infinity: chained `hypot` would return
        // inf for at least one of them; the unified contract keeps NaN.
        assert!(scale_safe_norm([f64::NAN, f64::INFINITY]).is_nan());
        assert!(scale_safe_norm([f64::INFINITY, f64::NAN]).is_nan());
    }

    #[test]
    fn infinity_yields_infinite_norm() {
        // Single and repeated infinities both give inf (the true norm of
        // an unbounded vector is inf).
        assert!(scale_safe_norm([f64::INFINITY]).is_infinite());
        assert!(scale_safe_norm([f64::INFINITY, f64::INFINITY]).is_infinite());
        assert!(scale_safe_norm([1.0, f64::INFINITY, 2.0]).is_infinite());
        assert!(scale_safe_norm([f64::NEG_INFINITY, 3.0]).is_infinite());
    }

    #[test]
    fn accumulator_representation_matches_finish_on_moderate_input() {
        // (scale, sumsq) = (4, 1 + 9/16) after pushing 3 and 4, so the
        // represented norm scale * sqrt(sumsq) is exactly finish() = 5.
        let mut acc = NormAccumulator::new();
        acc.push(3.0_f64);
        acc.push(4.0);
        assert_eq!(acc.scale(), 4.0);
        assert_relative_eq!(acc.sumsq(), 25.0 / 16.0, max_relative = 1e-15);
        assert_eq!(acc.finish(), 5.0);
        assert_eq!(acc.scale() * acc.sumsq().sqrt(), acc.finish());
    }

    #[test]
    fn accumulator_pair_stays_finite_past_the_saturation_point() {
        // Four values whose true combined norm 2e308 exceeds f64::MAX
        // (~1.8e308): finish() saturates to inf, but the (scale, sumsq)
        // representation stays exact and bounded (sumsq <= pushes + 1).
        let mut acc = NormAccumulator::new();
        for _ in 0..4 {
            acc.push(1.0e308_f64);
        }
        assert!(acc.finish().is_infinite());
        assert_eq!(acc.scale(), 1.0e308);
        assert_relative_eq!(acc.sumsq(), 4.0, max_relative = 1e-15);
    }

    #[test]
    fn component_pushes_survive_modulus_overflow() {
        // A complex element with finite components can have an
        // unrepresentable modulus; the component-wise scalar push keeps
        // the accumulator's scale finite where pushing hypot(re, im)
        // would poison it with inf.
        let z = num_complex::Complex::new(1.5e308_f64, 1.5e308);
        assert!(z.re.hypot(z.im).is_infinite());
        let mut acc = NormAccumulator::new();
        acc.push_scalar(z);
        assert!(acc.scale().is_finite());
        assert_relative_eq!(acc.sumsq(), 2.0, max_relative = 1e-15);
    }

    #[test]
    fn accumulator_zero_state_reports_zero_scale() {
        // scale() == 0 is the saturation-free zero test: nothing nonzero
        // pushed keeps both the scale and the finished norm at zero.
        let mut acc = NormAccumulator::<f64>::new();
        assert_eq!(acc.scale(), 0.0);
        assert_eq!(acc.finish(), 0.0);
        acc.push(0.0);
        assert_eq!(acc.scale(), 0.0);
        assert_eq!(acc.finish(), 0.0);
    }

    #[test]
    fn combine_norms_matches_pythagorean_identity() {
        assert_eq!(combine_norms(3.0_f64, 4.0), 5.0);
        assert_eq!(combine_norms(0.0_f64, 7.0), 7.0);
        // Extreme scales survive combination as well.
        assert_relative_eq!(
            combine_norms(1e200_f64, 2e200),
            5.0_f64.sqrt() * 1e200,
            max_relative = 1e-12
        );
        // NaN propagates through combination even against infinity.
        assert!(combine_norms(f64::NAN, f64::INFINITY).is_nan());
        assert!(combine_norms(f64::INFINITY, f64::NAN).is_nan());
    }
}
