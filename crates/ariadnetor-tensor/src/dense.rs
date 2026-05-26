//! Dense tensor storage with flat contiguous data.
//!
//! Provides [`DenseTensorData<T>`] = [`TensorData<DenseStorage<T>, DenseLayout>`](crate::TensorData):
//! the joined storage+layout bundle that owns a 64-byte-aligned flat
//! buffer together with the logical shape and the memory order the
//! buffer is laid out in. The layout authority is the storage itself;
//! a downstream consumer that needs a specific layout should reorder
//! against `data.order()` (typically via
//! [`normalize_to_data`](crate::normalize_to_data)) rather than
//! assuming the buffer matches a backend's preferred order.

mod access;
mod constructors;
mod layout;
mod multi_tensor;
mod operations;
mod scalar_ops;
mod slice_data;
mod storage;
mod tensor_data;

pub use layout::DenseLayout;
pub use storage::DenseStorage;
pub use tensor_data::DenseTensorData;

use aligned_vec::ConstAlign;

/// 64-byte alignment for SIMD (AVX-512).
pub(crate) type Align64 = ConstAlign<64>;

/// Compute row-major strides as `usize` (for internal use in slice / expand / replace).
pub(crate) fn compute_strides_usize(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute column-major strides as `usize` (for internal use in slice / expand).
pub(crate) fn compute_strides_column_usize(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    strides
}
