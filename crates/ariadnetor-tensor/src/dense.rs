//! Dense tensor storage with strides-based memory layout
//!
//! Provides a dense tensor with explicit strides and Arc-based shared ownership.
//! The tensor is self-describing regarding its memory layout — code should not
//! assume row-major or column-major without checking.

mod access;
mod constructors;
mod multi_tensor;
mod operations;
mod scalar_ops;
mod slice;

use aligned_vec::{AVec, ConstAlign};
use std::fmt;
use std::sync::Arc;

/// 64-byte alignment for SIMD (AVX-512)
type Align64 = ConstAlign<64>;

pub use access::{DenseTensorIter, StridedIter};
pub use arnet_core::MemoryOrder;

/// Dense tensor with shared ownership (Arc + Copy-on-Write)
///
/// # Memory Layout
///
/// Each tensor carries explicit `strides` and `offset` describing how logical
/// indices map to positions in the underlying data buffer. Constructors default
/// to row-major (C-contiguous) layout, but backends may produce other layouts.
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64)
#[derive(Clone)]
pub struct DenseTensor<T = f64> {
    /// Shared data buffer (64-byte aligned)
    data: Arc<AVec<T, Align64>>,
    /// Tensor shape
    shape: Vec<usize>,
    /// Strides for each axis (element count, signed for future negative strides)
    strides: Vec<isize>,
    /// Offset into the data buffer (element index of the first logical element)
    offset: usize,
    /// The memory order this tensor was created with.
    /// Needed to disambiguate layouts where strides alone are ambiguous
    /// (e.g., 1D tensors, tensors with size-1 dimensions).
    order: MemoryOrder,
}

// ============================================================================
// Strides computation helpers
// ============================================================================

/// Compute row-major (C-order) strides from shape.
/// Last axis has stride 1, each preceding axis has stride = product of subsequent dims.
pub fn row_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

/// Compute column-major (Fortran-order) strides from shape.
/// First axis has stride 1, each subsequent axis has stride = product of preceding dims.
pub fn column_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}

// ============================================================================
// Basic accessors
// ============================================================================

impl<T> DenseTensor<T> {
    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the rank (number of dimensions) of the tensor
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get shape as i64 slice for MLIR compatibility
    pub fn shape_i64(&self) -> Vec<i64> {
        self.shape.iter().map(|&s| s as i64).collect()
    }

    /// Get the total number of logical elements
    pub fn len(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the strides of the tensor
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Get the offset into the data buffer
    pub fn offset(&self) -> usize {
        self.offset
    }

    // ========================================================================
    // Layout queries
    // ========================================================================

    /// Check if the tensor is contiguous in any standard order.
    pub fn is_contiguous(&self) -> bool {
        self.is_row_major() || self.is_column_major()
    }

    /// Check if strides match row-major (C-order) layout.
    pub fn is_row_major(&self) -> bool {
        self.strides == row_major_strides(&self.shape)
    }

    /// Check if strides match column-major (Fortran-order) layout.
    pub fn is_column_major(&self) -> bool {
        self.strides == column_major_strides(&self.shape)
    }

    /// The memory order this tensor was created with.
    ///
    /// Unlike `is_row_major()` / `is_column_major()` which check strides,
    /// this returns the authoritative order that disambiguates cases where
    /// strides are ambiguous (e.g., 1D tensors or tensors with size-1 dims).
    pub fn memory_order(&self) -> MemoryOrder {
        self.order
    }

    /// Determine the memory order of this tensor, if contiguous.
    fn contiguous_order(&self) -> Option<MemoryOrder> {
        if !self.is_contiguous() {
            return None;
        }
        // Use the authoritative order field, not strides-based heuristic
        Some(self.order)
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Convert multi-dimensional indices to flat index using strides and offset.
    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "Number of indices {} doesn't match tensor rank {}",
            indices.len(),
            self.shape.len()
        );

        indices.iter().zip(&self.shape).for_each(|(&idx, &dim)| {
            assert!(
                idx < dim,
                "Index {} out of bounds for dimension {}",
                idx,
                dim
            )
        });

        let raw: isize = indices
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx as isize * stride)
            .sum();
        (self.offset as isize + raw) as usize
    }
}

// ============================================================================
// Display / Debug
// ============================================================================

impl<T> fmt::Debug for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DenseTensor(shape={:?}, strides={:?}, offset={}, elements={})",
            self.shape,
            self.strides,
            self.offset,
            self.len()
        )
    }
}

impl<T> fmt::Display for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenseTensor{:?}", self.shape)
    }
}

/// Compute row-major strides as usize (for internal use in expand/replace_slice).
fn compute_strides_usize(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute column-major strides as usize (for internal use in expand).
fn compute_strides_column_usize(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    strides
}
