//! Scalar-dependent operations (conjugate, norm) for BlockSparse.

use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};
use num_traits::{Float, One, Zero};

use super::BlockSparse;
use crate::sector::Sector;

impl<T, S> BlockSparse<T, S>
where
    T: arnet_core::scalar::Scalar,
    S: Sector,
{
    /// Element-wise complex conjugate.
    pub fn conj(&self) -> Self {
        let new_data =
            AVec::<T, ConstAlign<64>>::from_iter(64, self.data.iter().copied().map(|x| x.conj()));
        Self {
            data: Arc::new(new_data),
            blocks: self.blocks.clone(),
            block_index: self.block_index.clone(),
            indices: self.indices.clone(),
            flux: self.flux.clone(),
            shape: self.shape.clone(),
        }
    }

    /// Compute squared Frobenius norm: Σ |element|².
    fn norm_squared(&self) -> T::Real {
        self.data
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

    /// Compute Frobenius norm (alias for [`norm_frobenius`](Self::norm_frobenius)).
    pub fn norm(&self) -> T::Real {
        self.norm_frobenius()
    }

    /// Normalize and return a new tensor (out-of-place).
    ///
    /// Returns `(normalized_tensor, original_norm)`.
    /// Panics if the tensor has zero norm.
    pub fn normalized(&self) -> (Self, T::Real) {
        let mut result = self.clone();
        let norm = result.normalize();
        (result, norm)
    }

    /// Normalize to unit Frobenius norm (in-place).
    ///
    /// Returns the norm before normalization.
    /// Panics if the tensor has zero norm.
    pub fn normalize(&mut self) -> T::Real {
        let norm = self.norm_frobenius();
        assert!(norm != T::Real::zero(), "Cannot normalize zero tensor");
        let inv_norm = T::Real::one() / norm;
        let data = Arc::make_mut(&mut self.data);
        for elem in data.iter_mut() {
            *elem = elem.scale_real(inv_norm);
        }
        norm
    }
}
