//! Dense tensor storage with flat contiguous data
//!
//! Provides a dense tensor with Arc-based shared ownership.
//! `Dense<T>` carries a flat data buffer, its shape, and the memory
//! order the data is laid out in. The layout authority is the storage
//! itself: a downstream consumer that needs a specific layout should
//! reorder against `dense.order()` (typically via
//! [`normalize_to`](crate::normalize_to)) rather than assuming the data
//! matches a backend's preferred order.

mod access;
mod constructors;
mod layout;
mod multi_tensor;
mod operations;
mod scalar_ops;
mod slice;
mod storage;
mod tensor_data;

pub use layout::DenseLayout;
pub use storage::DenseStorage;
pub use tensor_data::DenseTensorData;

use aligned_vec::{AVec, ConstAlign};
use arnet_core::backend::MemoryOrder;
use std::fmt;
use std::sync::Arc;

/// 64-byte alignment for SIMD (AVX-512)
type Align64 = ConstAlign<64>;

/// Dense tensor with shared ownership (Arc + Copy-on-Write)
///
/// Dense holds a flat contiguous data buffer, its shape, and the
/// memory order the data is laid out in. The `order` field is the
/// storage authority for layout interpretation: a consumer that
/// needs a specific layout (e.g. a backend kernel) should normalize
/// against `dense.order()` at its entry point, rather than assuming
/// the data matches the backend's preferred order. Migration of
/// every linalg/mps/algorithms op to this discipline is in progress.
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64)
pub struct Dense<T = f64> {
    /// Shared data buffer (64-byte aligned)
    data: Arc<AVec<T, Align64>>,
    /// Tensor shape
    shape: Vec<usize>,
    /// Memory order the flat data is laid out in
    order: MemoryOrder,
}

// Manual Clone impl: all fields are Clone regardless of T
// (Arc<AVec<T, _>> is Clone without T: Clone).
// #[derive(Clone)] would unnecessarily require T: Clone.
impl<T> Clone for Dense<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
            order: self.order,
        }
    }
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

    /// Get the total number of logical elements
    pub fn len(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Memory order this tensor's flat data is laid out in.
    ///
    /// Operations and linalg kernels consult this to decide whether
    /// the data matches their expected layout.
    pub fn order(&self) -> MemoryOrder {
        self.order
    }
}

// ============================================================================
// Zero-cost conversion to/from `DenseTensorData<T>`
// ============================================================================
//
// `Dense<T>` and `DenseTensorData<T>` share the same flat data buffer
// representation (`Arc<AVec<T, Align64>>`). During the storage /
// layout split migration (issue #259) consumers cross this boundary
// at every linalg pub fn call; the converters below move the Arc
// directly so the boundary crossing is O(1) and allocation-free.
// The pair is `pub(crate)`-internal: the boundary crossing belongs to
// the migration scaffolding and is removed when the legacy `Dense<T>`
// type is deleted.

impl<T> Dense<T> {
    /// Consume `self` and return a `DenseTensorData<T>` sharing the
    /// same underlying buffer.
    pub fn into_tensor_data(self) -> DenseTensorData<T> {
        let storage = DenseStorage::from_arc(self.data);
        let layout = DenseLayout::new(self.shape, self.order);
        DenseTensorData::new(storage, layout)
    }

    /// Build a `Dense<T>` from an existing `DenseTensorData<T>`,
    /// reusing the Arc without reallocation.
    pub fn from_tensor_data(td: DenseTensorData<T>) -> Self {
        let (storage, layout) = td.into_parts();
        let (shape, order) = layout.into_parts();
        Self {
            data: storage.into_arc(),
            shape,
            order,
        }
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
