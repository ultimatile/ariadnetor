//! Scalar trait for tensor element types
//!
//! Provides `Scalar` trait unifying real and complex floating-point types.

use num_complex::Complex;
use num_traits::{One, Zero};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// Real-valued computation type for norm results and normalization factors.
///
/// Delegates to `num_traits::Float` for arithmetic operations.
/// Currently implemented for f32 and f64.
pub trait FloatCompute: num_traits::Float + 'static {}

impl FloatCompute for f32 {}
impl FloatCompute for f64 {}

/// Scalar type for tensor elements (sealed trait).
pub trait Scalar:
    sealed::Sealed
    + Clone
    + Copy
    + 'static
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
{
    type Real: FloatCompute;
    type Complex: Scalar;
    fn abs(self) -> Self::Real;
    fn re(self) -> Self::Real;
    fn im(self) -> Self::Real;
    fn scale_real(self, factor: Self::Real) -> Self;
    fn conj(self) -> Self;
    fn into_complex(self) -> Self::Complex;
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

    #[test]
    fn test_scalar_f64() {
        assert_eq!(3.0f64.abs(), 3.0);
        assert_eq!(3.0f64.scale_real(2.0), 6.0);
    }

    #[test]
    fn test_scalar_complex_f64() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.abs(), 5.0);
        assert_eq!(z.conj(), Complex::new(3.0, -4.0));
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

    #[test]
    fn test_round_trip_complex_f64() {
        let z = Complex::new(3.0f64, -4.0);
        let reconstructed = Complex::<f64>::from_real_imag(z.re(), z.im());
        assert_eq!(reconstructed, z);
    }

    #[test]
    fn test_round_trip_complex_f32() {
        let z = Complex::new(1.5f32, 2.5);
        let reconstructed = Complex::<f32>::from_real_imag(z.re(), z.im());
        assert_eq!(reconstructed, z);
    }

    #[test]
    fn test_round_trip_f64() {
        let x = 7.0f64;
        let reconstructed = f64::from_real_imag(x.re(), x.im());
        assert_eq!(reconstructed, x);
    }
}
