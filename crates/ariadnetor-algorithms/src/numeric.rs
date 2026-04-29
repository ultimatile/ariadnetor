//! Small numeric helpers shared by multiple algorithm modules.
//!
//! Currently only houses `try_real_from_f64`, which converts a
//! plain-`f64` tolerance / scalar to the algorithm's `T::Real` type.
//! Callers that own a fallible API surface use the `try_` form
//! directly; callers that have already pre-validated their input can
//! `.expect(...)` on the returned `Option`.

use arnet_core::Scalar;
use num_traits::NumCast;

/// Attempt to cast `x: f64` into the `T::Real` real type.
///
/// Returns `None` when `x` is not representable in `T::Real` — this
/// shows up in practice when `T::Real == f32` and the input exceeds
/// `f32::MAX` or is a non-finite value the caller did not screen out.
///
/// Callers should:
/// - Drive a fallible public API with `?`
///   (e.g. surface `Err(InvalidParams)` on `None`).
/// - Use `.expect(...)` after upstream validation guarantees the
///   value is representable, so an unrepresentable input is a bug,
///   not silent zero.
#[inline]
pub(crate) fn try_real_from_f64<T: Scalar>(x: f64) -> Option<T::Real> {
    <T::Real as NumCast>::from(x)
}
