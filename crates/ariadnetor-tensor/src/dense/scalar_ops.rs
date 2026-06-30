//! Scalar-dependent operations for `DenseTensorData<T>` (conjugate,
//! norm, complex conversion).
//!
//! Element-wise transforms (`conj`, `real`, `imag`, `to_complex`)
//! preserve the layout; norm-related operations route through the
//! storage half.

use crate::DenseTensorData;

impl<T> DenseTensorData<T>
where
    T: ariadnetor_core::Scalar,
{
    /// Element-wise complex conjugate.
    pub fn conj(&self) -> Self {
        self.map(|x| x.conj())
    }

    /// Convert each element to its complex representation.
    pub fn to_complex(&self) -> DenseTensorData<T::Complex> {
        self.map(|x| x.into_complex())
    }

    /// Extract the real part of each element.
    pub fn real(&self) -> DenseTensorData<T::Real> {
        self.map(|x| x.re())
    }

    /// Extract the imaginary part of each element.
    pub fn imag(&self) -> DenseTensorData<T::Real> {
        self.map(|x| x.im())
    }

    /// Compute Frobenius norm: √(Σ |element|²).
    pub fn norm_frobenius(&self) -> T::Real {
        self.storage().norm_frobenius()
    }

    /// Compute Frobenius norm (alias for [`norm_frobenius`](Self::norm_frobenius)).
    pub fn norm(&self) -> T::Real {
        self.storage().norm()
    }

    /// Normalize to unit Frobenius norm (in-place).
    ///
    /// Returns the norm before normalization. Panics if the tensor
    /// has zero norm.
    pub fn normalize(&mut self) -> T::Real {
        self.storage_mut().normalize()
    }

    /// Normalize and return a new tensor (out-of-place).
    ///
    /// Returns `(normalized_tensor, original_norm)`. Panics if the
    /// tensor has zero norm.
    pub fn normalized(&self) -> (Self, T::Real) {
        let mut result = self.clone();
        let norm = result.normalize();
        (result, norm)
    }
}
