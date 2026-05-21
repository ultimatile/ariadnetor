//! Crate-internal bridge helpers between the joined-form `Tensor`
//! surface and the kernel paths still used inside the mps internals.
//!
//! [`reorder_dense_tensor`] wraps `arnet::reorder_dense_data` so callers
//! holding a `DenseTensor<T, B>` stay on the joined surface.
//! [`dense_reshape`] still bridges through the legacy `Dense<T>`
//! representation pending a native `DenseTensorData::reshape`.

use std::sync::Arc;

use arnet::{ComputeBackend, DenseLayout, DenseStorage, DenseTensor, MemoryOrder, Scalar, Tensor};

/// Reorder a `DenseTensor`'s flat data to the requested memory order.
///
/// Delegates to the joined-surface `arnet::reorder_dense_data` and
/// re-wraps the result with the input's cached backend `Arc`. The
/// source layout is taken from `t.data().order()`, so callers do not
/// pass a redundant `from` argument that could disagree with the
/// stored layout.
pub(crate) fn reorder_dense_tensor<T, B>(
    t: &DenseTensor<T, B>,
    to: MemoryOrder,
) -> DenseTensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let backend_arc = Arc::clone(t.backend_arc());
    let reordered = arnet::reorder_dense_data(t.data(), to);
    Tensor::<DenseStorage<T>, DenseLayout, B>::with_backend(reordered, backend_arc)
}

/// Reshape a `DenseTensor` to a new shape (zero-copy, preserves
/// `order()` and backend). Wraps the legacy `Dense::reshape`.
pub(crate) fn dense_reshape<T, B>(t: &DenseTensor<T, B>, new_shape: Vec<usize>) -> DenseTensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let legacy = t.data().as_dense();
    let reshaped = legacy.reshape(new_shape);
    Tensor::<DenseStorage<T>, DenseLayout, B>::with_backend(
        reshaped.into_tensor_data(),
        Arc::clone(t.backend_arc()),
    )
}
