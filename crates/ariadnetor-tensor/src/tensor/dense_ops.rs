//! Dense-specific inherent methods on `Tensor<DenseStorage<S>, DenseLayout, B>`.
//!
//! Covers element access, in-place fills / scales, linear combinations,
//! Frobenius-norm-based normalization, and zero-copy reshape. These
//! operations are storage-local: they do not need the backend for
//! dispatch, so they work uniformly over any `B: ComputeBackend`.

use std::ops::Mul;
use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use num_traits::{One, Zero};

use super::Tensor;
use crate::{DenseLayout, DenseStorage, DenseTensorData, TensorData};

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

    /// Reshape to `new_shape` (zero-copy). Preserves the layout's memory
    /// order and the backend `Arc`. The flat data buffer is `Arc`-shared
    /// via `DenseStorage::Clone`, so the result aliases the same
    /// allocation as `self`.
    ///
    /// Under non-adjacent axis fusion the logical mapping differs
    /// between row-major and column-major; callers fusing such axes
    /// must reorder the flat buffer to the appropriate order first.
    ///
    /// # Panics
    ///
    /// Panics if `new_shape.iter().product() != self.len()`, via
    /// [`TensorData::new`]'s `storage.flat_len() == layout.storage_extent()`
    /// assert.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_layout = DenseLayout::new(new_shape, self.data.layout().order());
        let new_storage = self.data.storage().clone();
        Self::with_backend(
            TensorData::new(new_storage, new_layout),
            Arc::clone(&self.backend),
        )
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

    /// Element-wise complex conjugate. Result shares the input's
    /// backend `Arc`. Symmetric with [`BlockSparseTensor::conj`].
    pub fn conj(&self) -> Self {
        Self {
            data: self.data.conj(),
            backend: std::sync::Arc::clone(&self.backend),
        }
    }

    /// Return a tensor with flat data reordered to `to`. Result shares
    /// the input's backend `Arc`. When `self.data().order() == to`,
    /// the underlying buffer is shared via `Arc` rather than copied.
    pub fn reordered(&self, to: arnet_core::backend::MemoryOrder) -> Self {
        let reordered = crate::reorder::reorder_data(&self.data, to);
        Self {
            data: reordered,
            backend: std::sync::Arc::clone(&self.backend),
        }
    }
}
