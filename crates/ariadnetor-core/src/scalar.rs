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
    fn abs(self) -> Self::Real;
    fn scale_real(self, factor: Self::Real) -> Self;
    fn conj(self) -> Self;
}

impl Scalar for f32 {
    type Real = f32;
    #[inline]
    fn abs(self) -> Self::Real {
        self.abs()
    }
    #[inline]
    fn scale_real(self, factor: Self::Real) -> Self {
        self * factor
    }
    #[inline]
    fn conj(self) -> Self {
        self
    }
}

impl Scalar for f64 {
    type Real = f64;
    #[inline]
    fn abs(self) -> Self::Real {
        self.abs()
    }
    #[inline]
    fn scale_real(self, factor: Self::Real) -> Self {
        self * factor
    }
    #[inline]
    fn conj(self) -> Self {
        self
    }
}

impl Scalar for Complex<f32> {
    type Real = f32;
    #[inline]
    fn abs(self) -> Self::Real {
        self.norm()
    }
    #[inline]
    fn scale_real(self, factor: Self::Real) -> Self {
        Complex::new(self.re * factor, self.im * factor)
    }
    #[inline]
    fn conj(self) -> Self {
        Complex::conj(&self)
    }
}

impl Scalar for Complex<f64> {
    type Real = f64;
    #[inline]
    fn abs(self) -> Self::Real {
        self.norm()
    }
    #[inline]
    fn scale_real(self, factor: Self::Real) -> Self {
        Complex::new(self.re * factor, self.im * factor)
    }
    #[inline]
    fn conj(self) -> Self {
        Complex::conj(&self)
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
}
