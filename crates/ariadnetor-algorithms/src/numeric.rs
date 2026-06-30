//! Small numeric helpers shared by multiple algorithm modules.
//!
//! Currently only houses `try_real_from_f64`, which converts a
//! plain-`f64` tolerance / scalar to the algorithm's `T::Real` type.
//! Callers that own a fallible API surface use the `try_` form
//! directly; callers that have already pre-validated their input can
//! `.expect(...)` on the returned `Option`.

use ariadnetor_core::Scalar;
use num_traits::{Float, NumCast};

/// Attempt to cast `x: f64` into the `T::Real` real type.
///
/// Returns `None` when `x` is not representable in `T::Real` as a
/// finite value. Two distinct overflow modes are guarded:
///
/// - `NumCast::from` itself may return `None` for unsupported
///   conversions (rare in practice for the `f32`/`f64` real types).
/// - For `T::Real == f32`, `NumCast::from` does NOT return `None`
///   on out-of-range inputs — it returns `Some(±inf)`. We
///   post-check `is_finite()` so an `f64` value like `1e300`
///   surfaces as `None` instead of silently becoming infinity.
///   Without this check a downstream comparison like
///   `abs_delta <= tol_real` would always be `true` and report
///   spurious convergence.
///
/// Callers should:
/// - Drive a fallible public API with `?`
///   (e.g. surface `Err(InvalidParams)` on `None`).
/// - Use `.expect(...)` after upstream validation guarantees the
///   value is representable, so an unrepresentable input is a bug,
///   not silent zero.
#[inline]
pub(crate) fn try_real_from_f64<T: Scalar>(x: f64) -> Option<T::Real> {
    let cast = <T::Real as NumCast>::from(x)?;
    if cast.is_finite() { Some(cast) } else { None }
}

#[cfg(test)]
mod tests {
    //! Contract: `try_real_from_f64::<T>` returns `Some(x)` only if
    //! `x` is finite in `T::Real`. Out-of-range f64 inputs (which
    //! `NumCast::from` for f32 maps to `Some(±inf)` rather than
    //! `None`) and non-finite inputs (NaN, ±inf) must surface as
    //! `None` so fallible callers can return `InvalidParams`
    //! instead of silently using a non-finite tolerance.
    use super::try_real_from_f64;
    use num_complex::Complex;

    #[test]
    fn f64_identity_within_range() {
        assert_eq!(try_real_from_f64::<f64>(0.0), Some(0.0));
        assert_eq!(try_real_from_f64::<f64>(1e-10), Some(1e-10));
        assert_eq!(try_real_from_f64::<f64>(1e300), Some(1e300));
        assert_eq!(try_real_from_f64::<f64>(-1e300), Some(-1e300));
    }

    #[test]
    fn f64_rejects_non_finite() {
        assert_eq!(try_real_from_f64::<f64>(f64::INFINITY), None);
        assert_eq!(try_real_from_f64::<f64>(f64::NEG_INFINITY), None);
        assert_eq!(try_real_from_f64::<f64>(f64::NAN), None);
    }

    #[test]
    fn f32_within_range_round_trips() {
        assert_eq!(try_real_from_f64::<f32>(0.0), Some(0.0_f32));
        assert_eq!(try_real_from_f64::<f32>(1e-10), Some(1e-10_f32));
    }

    #[test]
    fn f32_rejects_overflow_to_inf() {
        // The whole reason this helper post-checks `is_finite()`:
        // `NumCast::from(1e300_f64)` returns `Some(f32::INFINITY)`,
        // not `None`. Without the post-check, the downstream
        // comparison `abs_delta <= tol_real` would always be `true`
        // and report spurious convergence.
        assert_eq!(try_real_from_f64::<f32>(1e300_f64), None);
        assert_eq!(try_real_from_f64::<f32>(-1e300_f64), None);
        assert_eq!(try_real_from_f64::<f32>((f32::MAX as f64) * 2.0), None);
    }

    #[test]
    fn f32_rejects_non_finite() {
        assert_eq!(try_real_from_f64::<f32>(f64::INFINITY), None);
        assert_eq!(try_real_from_f64::<f32>(f64::NEG_INFINITY), None);
        assert_eq!(try_real_from_f64::<f32>(f64::NAN), None);
    }

    #[test]
    fn complex_real_part_inherits_constraints() {
        // T::Real for Complex<f32> is f32, so the same overflow /
        // non-finite gates apply.
        assert_eq!(try_real_from_f64::<Complex<f32>>(1e300_f64), None);
        assert_eq!(try_real_from_f64::<Complex<f32>>(f64::NAN), None);
        assert_eq!(try_real_from_f64::<Complex<f32>>(1e-10), Some(1e-10_f32));
        // T::Real for Complex<f64> is f64.
        assert_eq!(try_real_from_f64::<Complex<f64>>(1e300_f64), Some(1e300));
        assert_eq!(try_real_from_f64::<Complex<f64>>(f64::INFINITY), None);
    }
}
