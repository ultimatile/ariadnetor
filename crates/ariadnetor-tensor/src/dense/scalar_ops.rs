//! Scalar-dependent operations (conjugate, norm, complex conversion).

use num_traits::{Float, One, Zero};

use super::DenseTensor;

impl<T> DenseTensor<T>
where
    T: arnet_core::scalar::Scalar,
{
    /// Element-wise complex conjugate.
    pub fn conj(&self) -> Self {
        self.map(|x| x.conj())
    }

    /// Convert each element to its complex representation.
    pub fn to_complex(&self) -> DenseTensor<T::Complex> {
        self.map(|x| x.into_complex())
    }

    /// Extract the real part of each element.
    pub fn real(&self) -> DenseTensor<T::Real> {
        self.map(|x| x.re())
    }

    /// Extract the imaginary part of each element.
    pub fn imag(&self) -> DenseTensor<T::Real> {
        self.map(|x| x.im())
    }

    /// Compute squared Frobenius norm: Σ |element|².
    fn norm_squared(&self) -> T::Real {
        let c = self.to_contiguous(self.memory_order());
        c.data()
            .iter()
            .map(|&x| {
                let a = x.abs();
                a * a
            })
            .fold(T::Real::zero(), |acc, x| acc + x)
    }

    /// Compute Frobenius norm: √(Σ |element|²).
    pub fn norm_frobenius(&self) -> T::Real {
        self.norm_squared().sqrt()
    }

    /// Normalize to unit Frobenius norm (in-place).
    ///
    /// Returns the norm before normalization.
    /// Panics if the tensor has zero norm.
    pub fn normalize_in_place(&mut self) -> T::Real {
        let norm = self.norm_frobenius();
        assert!(norm != T::Real::zero(), "Cannot normalize zero tensor");
        let inv_norm = T::Real::one() / norm;
        *self = self.to_contiguous(self.memory_order());
        let data = self.data_mut();
        for elem in data.iter_mut() {
            *elem = elem.scale_real(inv_norm);
        }
        norm
    }
}
