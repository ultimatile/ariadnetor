//! Small numeric helpers shared by multiple algorithm modules.
//!
//! Currently only houses `try_real_from_f64`, which converts a
//! plain-`f64` tolerance / scalar to the algorithm's `T::Real` type.
//! Callers that own a fallible API surface use the `try_` form
//! directly; callers that have already pre-validated their input can
//! `.expect(...)` on the returned `Option`.

use arnet_core::Scalar;
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
