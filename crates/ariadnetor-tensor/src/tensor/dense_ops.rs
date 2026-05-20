//! Dense-specific inherent methods on `Tensor<DenseStorage<S>, DenseLayout, B>`.
//!
//! Covers element access, in-place fills / scales, linear combinations,
//! and Frobenius-norm-based normalization. These operations are
//! storage-local: they do not need the backend for dispatch, so they
//! work uniformly over any `B: ComputeBackend`.

use std::ops::{Add, Mul};
use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use num_traits::{One, Zero};

use super::Tensor;
use crate::{DenseLayout, DenseStorage, DenseTensorData};

// ============================================================================
// Dense-specific data access (all backends)
// ============================================================================

impl<S, B: ComputeBackend> Tensor<DenseStorage<S>, DenseLayout, B> {
    /// Get a reference to the underlying contiguous data buffer.
    pub fn data_slice(&self) -> &[S] {
        self.data.storage().data()
    }

    /// Get a mutable reference to the underlying data buffer
    /// (CoW-aware).
    pub fn data_slice_mut(&mut self) -> &mut [S]
    where
        S: Clone,
    {
        self.data.storage_mut().data_mut()
    }
}

impl<S: Scalar, B: ComputeBackend> Tensor<DenseStorage<S>, DenseLayout, B> {
    /// Memory order this tensor's flat data is laid out in.
    pub fn order(&self) -> arnet_core::backend::MemoryOrder {
        self.data.layout().order()
    }

    /// Get element at the given indices.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len() != rank` or any index exceeds the
    /// corresponding axis dimension.
    pub fn get(&self, indices: &[usize]) -> S {
        let shape = self.shape();
        assert_eq!(
            indices.len(),
            shape.len(),
            "Tensor::get: indices length {} doesn't match rank {}",
            indices.len(),
            shape.len(),
        );
        for (axis, (&idx, &dim)) in indices.iter().zip(shape).enumerate() {
            assert!(
                idx < dim,
                "Tensor::get: index {idx} out of bounds for axis {axis} with size {dim}",
            );
        }
        let order = self.order();
        let flat = crate::flat_index(indices, shape, order);
        self.data.storage().data()[flat]
    }

    /// Set element at the given indices.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len() != rank` or any index exceeds the
    /// corresponding axis dimension.
    pub fn set(&mut self, indices: &[usize], value: S) {
        let shape_owned: Vec<usize> = self.shape().to_vec();
        assert_eq!(
            indices.len(),
            shape_owned.len(),
            "Tensor::set: indices length {} doesn't match rank {}",
            indices.len(),
            shape_owned.len(),
        );
        for (axis, (&idx, &dim)) in indices.iter().zip(&shape_owned).enumerate() {
            assert!(
                idx < dim,
                "Tensor::set: index {idx} out of bounds for axis {axis} with size {dim}",
            );
        }
        let order = self.order();
        let flat = crate::flat_index(indices, &shape_owned, order);
        self.data.storage_mut().data_mut()[flat] = value;
    }

    /// Fill the tensor with a constant value.
    pub fn fill(&mut self, value: S) {
        for slot in self.data.storage_mut().data_mut().iter_mut() {
            *slot = value;
        }
    }
}

// ============================================================================
// Dense-specific arithmetic operations (all backends)
// ============================================================================

impl<S: Clone, B: ComputeBackend> Tensor<DenseStorage<S>, DenseLayout, B> {
    /// Scale every element by a factor (in-place).
    pub fn scale<F>(&mut self, factor: F)
    where
        S: Mul<F, Output = S>,
        F: Clone,
    {
        for slot in self.data.storage_mut().data_mut().iter_mut() {
            *slot = slot.clone() * factor.clone();
        }
    }

    /// Scale every element by a factor (out-of-place).
    pub fn scaled<F>(&self, factor: F) -> Self
    where
        S: Mul<F, Output = S>,
        F: Clone,
    {
        let new_data: Vec<S> = self
            .data
            .storage()
            .data()
            .iter()
            .map(|x| x.clone() * factor.clone())
            .collect();
        let shape = self.shape().to_vec();
        let order = self.data.layout().order();
        let td = DenseTensorData::from_raw_parts(new_data, shape, order);
        Self::with_backend(td, Arc::clone(&self.backend))
    }
}

impl<S, B: ComputeBackend> Tensor<DenseStorage<S>, DenseLayout, B>
where
    S: Clone + Zero + One + Add<Output = S> + Mul<Output = S>,
{
    /// Linear combination of tensors: `Σ_i coefs[i] * tensors[i]`.
    ///
    /// All inputs must have the same shape and memory order. The
    /// result inherits the order of the first input.
    pub fn linear_combine(
        tensors: &[&Tensor<DenseStorage<S>, DenseLayout, B>],
        coefs: &[S],
    ) -> Result<Tensor<DenseStorage<S>, DenseLayout, B>, String> {
        if tensors.is_empty() {
            return Err("Cannot combine empty tensor list".to_string());
        }
        if tensors.len() != coefs.len() {
            return Err(format!(
                "linear_combine: tensors.len() = {} != coefs.len() = {}",
                tensors.len(),
                coefs.len(),
            ));
        }
        let shape0 = tensors[0].shape().to_vec();
        let order0 = tensors[0].data.layout().order();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.shape() != shape0.as_slice() {
                return Err(format!(
                    "linear_combine: shape mismatch at index {i}: expected {shape0:?}, got {:?}",
                    t.shape(),
                ));
            }
            if t.data.layout().order() != order0 {
                return Err(format!(
                    "linear_combine: memory-order mismatch at index {i}",
                ));
            }
        }
        let len: usize = shape0.iter().product();
        let mut acc = vec![S::zero(); len];
        for (t, c) in tensors.iter().zip(coefs) {
            for (a, s) in acc.iter_mut().zip(t.data.storage().data()) {
                *a = a.clone() + s.clone() * c.clone();
            }
        }
        let td = DenseTensorData::from_raw_parts(acc, shape0, order0);
        Ok(Tensor::with_backend(
            td,
            Arc::clone(tensors[0].backend_arc()),
        ))
    }

    /// Add all tensors (coefficients all = 1).
    pub fn add_all(
        tensors: &[&Tensor<DenseStorage<S>, DenseLayout, B>],
    ) -> Result<Tensor<DenseStorage<S>, DenseLayout, B>, String> {
        let coefs = vec![S::one(); tensors.len()];
        Self::linear_combine(tensors, &coefs)
    }
}

// ============================================================================
// Dense-specific norm / normalization (all backends)
// ============================================================================

impl<S, B: ComputeBackend> Tensor<DenseStorage<S>, DenseLayout, B>
where
    S: Scalar,
{
    /// Frobenius norm.
    pub fn norm(&self) -> S::Real {
        let mut sq = S::Real::zero();
        for &x in self.data.storage().data() {
            let a = x.abs();
            sq = sq + a * a;
        }
        <S::Real as num_traits::Float>::sqrt(sq)
    }

    /// Normalize to unit norm (in-place). Returns the original norm.
    ///
    /// # Panics
    ///
    /// Panics if the tensor has zero norm.
    pub fn normalize(&mut self) -> S::Real {
        let norm = self.norm();
        assert!(norm != S::Real::zero(), "Cannot normalize zero tensor");
        let inv_norm = S::Real::one() / norm;
        for slot in self.data.storage_mut().data_mut().iter_mut() {
            *slot = slot.scale_real(inv_norm);
        }
        norm
    }

    /// Normalize and return a new tensor (out-of-place).
    pub fn normalized(&self) -> (Self, S::Real) {
        let mut clone = self.clone();
        let n = clone.normalize();
        (clone, n)
    }
}
