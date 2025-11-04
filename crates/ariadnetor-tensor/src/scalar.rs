//! Scalar trait for tensor element types
//!
//! This module provides the `Scalar` trait which unifies real and complex
//! floating-point types for tensor operations, particularly norm and normalize.
//!
//! # Design
//!
//! The `Scalar` trait uses a sealed pattern to prevent external implementations
//! and ensure type safety. It provides operations needed for norm computation:
//! - `abs()`: Absolute value / modulus
//! - `scale_real()`: Multiply by real scalar
//! - `conj()`: Complex conjugate
//!
//! For basic arithmetic (scale, linear_combine), use standard trait bounds
//! (Zero, One, Add, Mul) instead of Scalar.
//!
//! # Example
//!
//! ```
//! use arnet_tensor::{RawTensor, Scalar};
//!
//! fn compute_norm<T: Scalar>(tensor: &RawTensor<T>) -> T::Real {
//!     tensor.norm()
//! }
//! ```

use num_complex::Complex;
use num_traits::{One, Zero};

// ============================================================================
// Sealed trait pattern
// ============================================================================

mod sealed {
    /// Sealed trait to prevent external implementations of Scalar
    pub trait Sealed {}

    // Only these types can implement Scalar
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}

// ============================================================================
// FloatCompute trait
// ============================================================================

/// Real-valued computation type
///
/// This trait represents real floating-point types used for norm results
/// and normalization factors. It is a subset of the `FloatCompute` trait
/// planned in `future_dtype_system.md`.
///
/// # Implementations
///
/// Current: `f32`, `f64`
///
/// Future: `half::f16`, `half::bf16`
///
/// # Why not `num_traits::Float`?
///
/// - `half::f16` and `half::bf16` only implement `FloatCore`, not `Float`
/// - Using `Float` would block f16/bf16 extension
/// - Custom trait allows future Storage/Compute separation
pub trait FloatCompute: Copy + PartialOrd + 'static {
    /// Zero element
    fn zero() -> Self;

    /// One element
    fn one() -> Self;

    /// Square root
    fn sqrt(self) -> Self;

    /// Addition (for fold operations)
    fn add(self, rhs: Self) -> Self;

    /// Multiplication (for fold operations)
    fn mul(self, rhs: Self) -> Self;

    /// Division (for normalization)
    fn div(self, rhs: Self) -> Self;
}

impl FloatCompute for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }
}

impl FloatCompute for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }
}

// ============================================================================
// Scalar trait
// ============================================================================

/// Scalar type for tensor elements
///
/// This trait provides operations needed for norm computation and normalization.
/// For basic arithmetic (scale, linear_combine), use standard trait bounds
/// (`Zero`, `One`, `Add`, `Mul`).
///
/// This trait is sealed to prevent external implementations.
///
/// # Type Parameters
///
/// - `Real`: Real-valued component type
///   - For real numbers (f32/f64): same as `Self`
///   - For complex numbers: the underlying real type (f32 or f64)
///
/// # Examples
///
/// ```
/// use arnet_tensor::{RawTensor, Scalar};
/// use num_complex::Complex;
///
/// // Real tensor
/// let real_tensor = RawTensor::<f64>::ones(vec![3, 3]);
/// let real_norm: f64 = real_tensor.norm();
///
/// // Complex tensor
/// let complex_data = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
/// let complex_tensor = RawTensor::from_data(complex_data, vec![2]);
/// let complex_norm: f64 = complex_tensor.norm();  // Returns real-valued norm
/// ```
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
    /// Real-valued component type
    ///
    /// For real numbers (f32/f64): same as `Self`
    ///
    /// For complex numbers: the underlying real type (f32 or f64)
    type Real: FloatCompute;

    /// Absolute value / modulus (returns real value)
    ///
    /// For real: |x|
    ///
    /// For complex: |z| = sqrt(re² + im²)
    fn abs(self) -> Self::Real;

    /// Multiply by real scalar
    ///
    /// For real: x * factor
    ///
    /// For complex: (re * factor, im * factor)
    fn scale_real(self, factor: Self::Real) -> Self;

    /// Complex conjugate
    ///
    /// For real: identity (returns self)
    ///
    /// For complex: (re, -im)
    fn conj(self) -> Self;
}

// ============================================================================
// Scalar implementations for real types
// ============================================================================

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
        self // Identity for real numbers
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
        self // Identity for real numbers
    }
}

// ============================================================================
// Scalar implementations for complex types
// ============================================================================

impl Scalar for Complex<f32> {
    type Real = f32;

    #[inline]
    fn abs(self) -> Self::Real {
        self.norm() // |z| = sqrt(re² + im²)
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
        self.norm() // |z| = sqrt(re² + im²)
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
        let x = 3.0f64;
        assert_eq!(x.abs(), 3.0);
        assert_eq!(x.scale_real(2.0), 6.0);
        assert_eq!(x.conj(), 3.0);
    }

    #[test]
    fn test_scalar_f32() {
        let x = -2.5f32;
        assert_eq!(x.abs(), 2.5);
        assert_eq!(x.scale_real(3.0), -7.5);
        assert_eq!(x.conj(), -2.5);
    }

    #[test]
    fn test_scalar_complex_f64() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.abs(), 5.0); // |3+4i| = 5
        assert_eq!(z.scale_real(2.0), Complex::new(6.0, 8.0));
        assert_eq!(z.conj(), Complex::new(3.0, -4.0));
    }

    #[test]
    fn test_scalar_complex_f32() {
        let z = Complex::new(1.0f32, 1.0f32);
        let abs_val = z.abs();
        assert!((abs_val - 2.0f32.sqrt()).abs() < 1e-6);
        assert_eq!(z.scale_real(3.0), Complex::new(3.0, 3.0));
        assert_eq!(z.conj(), Complex::new(1.0, -1.0));
    }

    #[test]
    fn test_float_compute_f64() {
        assert_eq!(<f64 as FloatCompute>::zero(), 0.0);
        assert_eq!(<f64 as FloatCompute>::one(), 1.0);
        assert_eq!(FloatCompute::sqrt(9.0f64), 3.0);
        assert_eq!(FloatCompute::add(2.0, 3.0), 5.0);
        assert_eq!(FloatCompute::mul(2.0, 3.0), 6.0);
        assert_eq!(FloatCompute::div(6.0, 2.0), 3.0);
    }

    #[test]
    fn test_float_compute_f32() {
        assert_eq!(<f32 as FloatCompute>::zero(), 0.0);
        assert_eq!(<f32 as FloatCompute>::one(), 1.0);
        assert_eq!(FloatCompute::sqrt(4.0f32), 2.0);
        assert_eq!(FloatCompute::add(1.5, 2.5), 4.0);
        assert_eq!(FloatCompute::mul(2.0, 1.5), 3.0);
        assert_eq!(FloatCompute::div(9.0, 3.0), 3.0);
    }
}
