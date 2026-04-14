//! Dense tensor storage with flat contiguous data
//!
//! Provides a dense tensor with Arc-based shared ownership.
//! Dense is pure storage — it holds `data` and `shape` only.
//! Memory layout interpretation (row-major vs column-major) is
//! delegated to the compute backend at operation time.

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

pub use arnet_core::MemoryOrder;

/// Dense tensor with shared ownership (Arc + Copy-on-Write)
///
/// Dense is pure contiguous storage: a flat data buffer plus its shape.
/// It carries no layout information (no strides, no offset, no memory order).
/// Layout interpretation is the responsibility of the compute backend,
/// mediated through `Tensor<Dense, B>` or explicit `MemoryOrder` parameters.
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64)
pub struct Dense<T = f64> {
    /// Shared data buffer (64-byte aligned)
    data: Arc<AVec<T, Align64>>,
    /// Tensor shape
    shape: Vec<usize>,
}

// Manual Clone impl: all fields are Clone regardless of T
// (Arc<AVec<T, _>> is Clone without T: Clone).
// #[derive(Clone)] would unnecessarily require T: Clone.
impl<T> Clone for Dense<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
        }
    }
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

impl<T> Dense<T> {
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
}

// ============================================================================
// Display / Debug
// ============================================================================

impl<T> fmt::Debug for Dense<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dense(shape={:?}, elements={})", self.shape, self.len())
    }
}

impl<T> fmt::Display for Dense<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dense{:?}", self.shape)
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
