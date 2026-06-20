//! Scalar trait for tensor element types
//!
//! Provides `Scalar` trait unifying real and complex floating-point types.
//! This auxiliary trait resolves E0592 (inherent impl overlap) that prevents
//! separate generic impls for `DenseTensorData<T: Float>` and
//! `DenseTensorData<Complex<T: Float>>`.

use num_complex::Complex;
use num_traits::{One, Zero};

use crate::backend::DispatchScalar;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// Scalar type for tensor elements (sealed trait).
///
/// See ADR-0003 for the design rationale (E0592 avoidance via sealed pattern).
pub trait Scalar:
    sealed::Sealed
    + Clone
    + Copy
    + 'static
    + Send
    + Sync
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Mul<Self::Real, Output = Self>
    + DispatchScalar
{
    /// The real part type. Always `Scalar + Float` — supports both tensor
    /// element storage and floating-point math (`sqrt`, `exp`, etc.).
    type Real: Scalar + num_traits::Float;
    /// The complex type having this scalar's real type as its components.
    type Complex: Scalar;
    /// Absolute value (modulus), as the real type.
    fn abs(self) -> Self::Real;
    /// Real part.
    fn re(self) -> Self::Real;
    /// Imaginary part (always zero for real scalars).
    fn im(self) -> Self::Real;
    /// Multiply by a real factor.
    fn scale_real(self, factor: Self::Real) -> Self;
    /// Complex conjugate (identity for real scalars).
    fn conj(self) -> Self;
    /// Widen into the corresponding complex type.
    fn into_complex(self) -> Self::Complex;
    /// Build from real and imaginary parts; for real scalars the
    /// imaginary part is ignored.
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self;
}

impl Scalar for f32 {
    type Real = f32;
    type Complex = Complex<f32>;
    #[inline]
    fn abs(self) -> Self::Real {
        self.abs()
    }
    #[inline]
    fn re(self) -> Self::Real {
        self
    }
    #[inline]
    fn im(self) -> Self::Real {
        0.0
    }
    #[inline]
    fn scale_real(self, factor: Self::Real) -> Self {
        self * factor
    }
    #[inline]
    fn conj(self) -> Self {
        self
    }
    #[inline]
    fn into_complex(self) -> Self::Complex {
        Complex::new(self, 0.0)
    }
    #[inline]
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self {
        let _ = im;
        re
    }
}

impl Scalar for f64 {
    type Real = f64;
    type Complex = Complex<f64>;
    #[inline]
    fn abs(self) -> Self::Real {
        self.abs()
    }
    #[inline]
    fn re(self) -> Self::Real {
        self
    }
    #[inline]
    fn im(self) -> Self::Real {
        0.0
    }
    #[inline]
    fn scale_real(self, factor: Self::Real) -> Self {
        self * factor
    }
    #[inline]
    fn conj(self) -> Self {
        self
    }
    #[inline]
    fn into_complex(self) -> Self::Complex {
        Complex::new(self, 0.0)
    }
    #[inline]
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self {
        let _ = im;
        re
    }
}

impl Scalar for Complex<f32> {
    type Real = f32;
    type Complex = Complex<f32>;
    #[inline]
    fn abs(self) -> Self::Real {
        self.norm()
    }
    #[inline]
    fn re(self) -> Self::Real {
        self.re
    }
    #[inline]
    fn im(self) -> Self::Real {
        self.im
    }
    #[inline]
    fn scale_real(self, factor: Self::Real) -> Self {
        Complex::new(self.re * factor, self.im * factor)
    }
    #[inline]
    fn conj(self) -> Self {
        Complex::conj(&self)
    }
    #[inline]
    fn into_complex(self) -> Self::Complex {
        self
    }
    #[inline]
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self {
        Complex::new(re, im)
    }
}

impl Scalar for Complex<f64> {
    type Real = f64;
    type Complex = Complex<f64>;
    #[inline]
    fn abs(self) -> Self::Real {
        self.norm()
    }
    #[inline]
    fn re(self) -> Self::Real {
        self.re
    }
    #[inline]
    fn im(self) -> Self::Real {
        self.im
    }
    #[inline]
    fn scale_real(self, factor: Self::Real) -> Self {
        Complex::new(self.re * factor, self.im * factor)
    }
    #[inline]
    fn conj(self) -> Self {
        Complex::conj(&self)
    }
    #[inline]
    fn into_complex(self) -> Self::Complex {
        self
    }
    #[inline]
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self {
        Complex::new(re, im)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify Scalar trait algebraic laws for any implementing type.
    /// Uses fully-qualified `Scalar::method(x)` calls to test trait impls,
    /// not inherent methods (e.g., f64::abs shadows Scalar::abs).
    fn assert_scalar_laws<S>(x: S, factor: S::Real)
    where
        S: Scalar + PartialEq + std::fmt::Debug,
        S::Real: PartialEq + std::fmt::Debug,
    {
        // abs is positive for non-zero input
        assert!(Scalar::abs(x) > S::Real::zero());
        // conj is involution
        assert_eq!(Scalar::conj(Scalar::conj(x)), x);
        // conj preserves re, negates im (real: 0 == -0, complex: real test)
        assert_eq!(Scalar::re(Scalar::conj(x)), Scalar::re(x));
        assert_eq!(Scalar::im(Scalar::conj(x)), S::Real::zero() - Scalar::im(x),);
        // scale_real identity
        assert_eq!(Scalar::scale_real(x, S::Real::one()), x);
        // scale_real with non-trivial factor
        let scaled = Scalar::scale_real(x, factor);
        assert_eq!(Scalar::re(scaled), Scalar::re(x) * factor);
        assert_eq!(Scalar::im(scaled), Scalar::im(x) * factor);
        // re/im round-trip
        assert_eq!(S::from_real_imag(Scalar::re(x), Scalar::im(x)), x);
    }

    #[test]
    fn test_scalar_laws() {
        assert_scalar_laws(2.5f32, 3.0);
        assert_scalar_laws(2.5f64, 3.0);
        assert_scalar_laws(Complex::new(3.0f32, 4.0), 2.0);
        assert_scalar_laws(Complex::new(3.0f64, 4.0), 2.0);
    }

    #[test]
    fn test_into_complex_f32() {
        let x = 2.5f32;
        let z = x.into_complex();
        assert_eq!(z, Complex::new(2.5f32, 0.0));
    }

    #[test]
    fn test_into_complex_f64() {
        let x = 3.0f64;
        let z = x.into_complex();
        assert_eq!(z, Complex::new(3.0f64, 0.0));
    }

    #[test]
    fn test_into_complex_already_complex() {
        let z = Complex::new(1.0f64, 2.0);
        assert_eq!(z.into_complex(), z);

        let z32 = Complex::new(1.0f32, 2.0);
        assert_eq!(z32.into_complex(), z32);
    }

    #[test]
    fn test_re_im_f64() {
        let x = 3.5f64;
        assert_eq!(x.re(), 3.5);
        assert_eq!(x.im(), 0.0);
    }

    #[test]
    fn test_re_im_f32() {
        let x = 2.5f32;
        assert_eq!(x.re(), 2.5);
        assert_eq!(x.im(), 0.0);
    }

    #[test]
    fn test_re_im_complex_f64() {
        let z = Complex::new(3.0f64, 4.0);
        assert_eq!(z.re(), 3.0);
        assert_eq!(z.im(), 4.0);
    }

    #[test]
    fn test_re_im_complex_f32() {
        let z = Complex::new(1.0f32, -2.0);
        assert_eq!(z.re(), 1.0);
        assert_eq!(z.im(), -2.0);
    }

    #[test]
    fn test_from_real_imag_complex_f64() {
        let z = Complex::<f64>::from_real_imag(3.0, 4.0);
        assert_eq!(z, Complex::new(3.0, 4.0));
    }

    #[test]
    fn test_from_real_imag_complex_f32() {
        let z = Complex::<f32>::from_real_imag(1.0, -2.0);
        assert_eq!(z, Complex::new(1.0, -2.0));
    }

    #[test]
    fn test_from_real_imag_f64() {
        let x = f64::from_real_imag(3.0, 999.0);
        assert_eq!(x, 3.0);
    }

    #[test]
    fn test_from_real_imag_f32() {
        let x = f32::from_real_imag(2.5, 999.0);
        assert_eq!(x, 2.5);
    }
}
