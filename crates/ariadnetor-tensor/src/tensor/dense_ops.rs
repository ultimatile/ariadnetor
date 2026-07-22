//! Dense-specific inherent methods on `Tensor<DenseStorage<S>, DenseLayout>`.
//!
//! Covers element access, in-place fills / scales, Frobenius-norm-based
//! normalization, conjugation, zero-copy reshape, and reorder. These
//! operations are storage-local: they do not need a backend for dispatch.

use std::ops::{Mul, MulAssign};

use ariadnetor_core::Scalar;

use super::Tensor;
use crate::{DenseLayout, DenseStorage, DenseTensorData, TensorData};

// ============================================================================
// Dense-specific data access (all backends)
// ============================================================================

impl<S> Tensor<DenseStorage<S>, DenseLayout> {
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
    /// order. The flat data buffer is `Arc`-shared via
    /// `DenseStorage::Clone`, so the result aliases the same allocation
    /// as `self`.
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
        Self::from_data(TensorData::new(new_storage, new_layout))
    }
}

impl<S: Scalar> Tensor<DenseStorage<S>, DenseLayout> {
    /// Memory order this tensor's flat data is laid out in.
    pub fn order(&self) -> ariadnetor_core::backend::MemoryOrder {
        self.data.layout().order()
    }

    /// Get element at the given indices.
    ///
    /// `indices` accepts any `AsRef<[usize]>`, so an array literal can be
    /// passed without a borrow: `t.get([i, j])` as well as `t.get(&coords)`.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len() != rank` or any index exceeds the
    /// corresponding axis dimension.
    pub fn get(&self, indices: impl AsRef<[usize]>) -> S {
        let indices = indices.as_ref();
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
    /// `indices` accepts any `AsRef<[usize]>`, so an array literal can be
    /// passed without a borrow: `t.set([i, j], v)` as well as `t.set(&coords, v)`.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len() != rank` or any index exceeds the
    /// corresponding axis dimension.
    pub fn set(&mut self, indices: impl AsRef<[usize]>, value: S) {
        let indices = indices.as_ref();
        // Resolve the flat offset under an immutable borrow that ends before
        // the mutable storage borrow below, so no owned-shape copy is needed.
        let flat = {
            let shape = self.shape();
            assert_eq!(
                indices.len(),
                shape.len(),
                "Tensor::set: indices length {} doesn't match rank {}",
                indices.len(),
                shape.len(),
            );
            for (axis, (&idx, &dim)) in indices.iter().zip(shape).enumerate() {
                assert!(
                    idx < dim,
                    "Tensor::set: index {idx} out of bounds for axis {axis} with size {dim}",
                );
            }
            crate::flat_index(indices, shape, self.order())
        };
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

impl<S: Clone> Tensor<DenseStorage<S>, DenseLayout> {
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
        Self::from_data(td)
    }
}

// ============================================================================
// Scalar-multiplication operators on the joined DenseTensor surface
// ============================================================================
//
// Convenience aliases for `scale` / `scaled`, restricted to a same-type
// factor (`S` matches the element type). Cross-type factors keep using
// the named methods, since a single `Mul` impl cannot cover them without
// conflicting coherence.

impl<S> Mul<S> for Tensor<DenseStorage<S>, DenseLayout>
where
    S: Clone + Mul<Output = S>,
{
    type Output = Tensor<DenseStorage<S>, DenseLayout>;

    /// Scale by `rhs`, consuming `self`. Reuses the owned buffer in
    /// place (no extra allocation when the storage is uniquely owned;
    /// a buffer still shared via copy-on-write is cloned first).
    fn mul(mut self, rhs: S) -> Self::Output {
        self.scale(rhs);
        self
    }
}

impl<S> Mul<S> for &Tensor<DenseStorage<S>, DenseLayout>
where
    S: Clone + Mul<Output = S>,
{
    type Output = Tensor<DenseStorage<S>, DenseLayout>;

    /// Scale by `rhs`, leaving `self` untouched (out-of-place).
    fn mul(self, rhs: S) -> Self::Output {
        self.scaled(rhs)
    }
}

impl<S> MulAssign<S> for Tensor<DenseStorage<S>, DenseLayout>
where
    S: Clone + Mul<Output = S>,
{
    /// Scale every element by `rhs` in place.
    fn mul_assign(&mut self, rhs: S) {
        self.scale(rhs);
    }
}

// ============================================================================
// Dense-specific norm / normalization (all backends)
// ============================================================================

impl<S> Tensor<DenseStorage<S>, DenseLayout>
where
    S: Scalar,
{
    /// Frobenius norm.
    pub fn norm(&self) -> S::Real {
        self.data.storage().norm_frobenius()
    }

    /// Normalize to unit norm (in-place). Returns the original norm.
    ///
    /// # Panics
    ///
    /// Panics if the tensor has zero norm.
    pub fn normalize(&mut self) -> S::Real {
        // Delegate to the dense storage normalizer so the norm-and-divide
        // contract lives in one place rather than diverging across sites.
        self.data.normalize()
    }

    /// Normalize and return a new tensor (out-of-place).
    pub fn normalized(&self) -> (Self, S::Real) {
        let mut clone = self.clone();
        let n = clone.normalize();
        (clone, n)
    }

    /// Element-wise complex conjugate. Symmetric with
    /// [`BlockSparseTensor::conj`](crate::BlockSparseTensor::conj).
    pub fn conj(&self) -> Self {
        Self {
            data: self.data.conj(),
        }
    }

    /// Return a tensor with flat data reordered to `to`. When
    /// `self.data().order() == to`, the underlying buffer is shared via
    /// `Arc` rather than copied.
    ///
    /// This is a **workspace-internal escape hatch**, not a user entry
    /// point. The public `Tensor` surface hides memory layout: constructors
    /// take no order, and the linalg / algorithm layers normalize to the
    /// backend's preferred order internally. The only in-tree callers are
    /// that internal plumbing (and the order-mismatch rejection tests).
    /// End users should never need to choose a `MemoryOrder`; as an inherent
    /// method on a re-exported type it cannot be hidden from umbrella users,
    /// hence this note.
    pub fn reordered(&self, to: ariadnetor_core::backend::MemoryOrder) -> Self {
        let reordered = crate::reorder::reorder_data(&self.data, to);
        Self { data: reordered }
    }

    /// General logical (C-order) reshape to an arbitrary target shape,
    /// preserving the tensor's memory order. The buffer is routed
    /// through row-major so the logical axis grouping is independent of
    /// the physical layout, then restored to the original order; for a
    /// row-major tensor each step is a zero-copy `Arc` share, for a
    /// column-major tensor it costs one round-trip transpose.
    ///
    /// This is the low-level escape hatch for multi-leg regroupings that
    /// [`fuse_legs`] / [`split_leg`] cannot express in a single
    /// operation — e.g. fusing two disjoint leg groups at once. Prefer
    /// [`fuse_legs`] / [`split_leg`] for single-leg fuse / split: they
    /// constrain which axis changes and read as intent. Like
    /// [`reshape`], only the total element count is validated.
    ///
    /// [`fuse_legs`]: Self::fuse_legs
    /// [`split_leg`]: Self::split_leg
    /// [`reshape`]: Self::reshape
    ///
    /// # Panics
    ///
    /// Panics if `new_shape`'s total element count differs from the
    /// tensor's, via [`reshape`].
    pub fn reshape_logical(&self, new_shape: Vec<usize>) -> Self {
        let orig_order = self.order();
        self.reordered(ariadnetor_core::backend::MemoryOrder::RowMajor)
            .reshape(new_shape)
            .reordered(orig_order)
    }

    /// Fuse a contiguous range of axes into a single leg, grouping
    /// them in row-major (C-order) logical order regardless of the
    /// tensor's physical memory order.
    ///
    /// The fused leg's extent is the product of the fused axes'
    /// extents and its logical index runs fastest over the last fused
    /// axis. The result keeps `self`'s memory order. Use [`reshape`]
    /// instead when a raw, order-preserving buffer reinterpretation is
    /// wanted. Inverse of [`split_leg`] over the same range. Convenience
    /// over [`reshape_logical`] for the single-leg case; for multi-group
    /// regroupings call [`reshape_logical`] directly.
    ///
    /// [`reshape`]: Self::reshape
    /// [`split_leg`]: Self::split_leg
    /// [`reshape_logical`]: Self::reshape_logical
    ///
    /// # Panics
    ///
    /// Panics unless `range.start < range.end <= rank`.
    pub fn fuse_legs(&self, range: std::ops::Range<usize>) -> Self {
        let shape = self.shape();
        let rank = shape.len();
        assert!(
            range.start < range.end && range.end <= rank,
            "fuse_legs: range {range:?} out of bounds for rank {rank}",
        );
        let fused: usize = shape[range.clone()].iter().product();
        let mut new_shape = shape[..range.start].to_vec();
        new_shape.push(fused);
        new_shape.extend_from_slice(&shape[range.end..]);
        self.reshape_logical(new_shape)
    }

    /// Split one axis into multiple axes, distributing the extent in
    /// row-major (C-order) logical order regardless of the tensor's
    /// physical memory order.
    ///
    /// `into` lists the resulting extents from slowest- to
    /// fastest-varying. The result keeps `self`'s memory order.
    /// Inverse of [`fuse_legs`] for a contiguous range. Convenience over
    /// [`reshape_logical`] for the single-leg case; for multi-group
    /// regroupings call [`reshape_logical`] directly.
    ///
    /// [`fuse_legs`]: Self::fuse_legs
    /// [`reshape_logical`]: Self::reshape_logical
    ///
    /// # Panics
    ///
    /// Panics unless `axis < rank`, `into` is non-empty, and
    /// `into.iter().product() == shape[axis]`.
    pub fn split_leg(&self, axis: usize, into: &[usize]) -> Self {
        let shape = self.shape();
        let rank = shape.len();
        assert!(
            axis < rank,
            "split_leg: axis {axis} out of bounds for rank {rank}",
        );
        assert!(!into.is_empty(), "split_leg: `into` must be non-empty");
        let prod: usize = into.iter().product();
        assert_eq!(
            prod, shape[axis],
            "split_leg: product of {into:?} != axis {axis} extent {}",
            shape[axis],
        );
        let mut new_shape = shape[..axis].to_vec();
        new_shape.extend_from_slice(into);
        new_shape.extend_from_slice(&shape[axis + 1..]);
        self.reshape_logical(new_shape)
    }
}
