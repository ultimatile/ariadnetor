//! Crate-internal bridge helper between the joined `DenseTensor`
//! surface and the legacy `Dense<T>::reshape` path pending a native
//! `DenseTensorData::reshape`.

use std::sync::Arc;

use arnet::{ComputeBackend, DenseLayout, DenseStorage, DenseTensor, Scalar, Tensor};

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
