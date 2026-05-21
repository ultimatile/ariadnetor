//! Crate-internal bridge helpers between the joined-form `Tensor`
//! surface and the kernel paths still used inside the mps internals.
//!
//! [`reorder_dense_tensor`] wraps `arnet::reorder_dense_data` so callers
//! holding a `DenseTensor<T, B>` stay on the joined `DenseTensorData`
//! surface. Block-sparse `dagger` / `conj` are inherent on
//! `BlockSparseTensorData<T, S>`, so the corresponding wrappers
//! round-trip through `data()` and re-wrap with the cached backend.

use std::sync::Arc;

use arnet::{
    BlockSparseStorage, BlockSparseTensor, ComputeBackend, DenseLayout, DenseStorage, DenseTensor,
    MemoryOrder, Scalar, Sector, Tensor,
};

/// Reorder a `DenseTensor`'s flat data between memory orders.
///
/// Delegates to the joined-surface `arnet::reorder_dense_data` and
/// re-wraps the result with the input's cached backend `Arc`.
pub(crate) fn reorder_dense_tensor<T, B>(
    t: &DenseTensor<T, B>,
    from: MemoryOrder,
    to: MemoryOrder,
) -> DenseTensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let backend_arc = Arc::clone(t.backend_arc());
    let reordered = arnet::reorder_dense_data(t.data(), from, to);
    Tensor::<DenseStorage<T>, DenseLayout, B>::with_backend(reordered, backend_arc)
}

/// Hermitian adjoint of a `BlockSparseTensor`. Wraps the
/// `BlockSparseTensorData::dagger` joined-form method back into a
/// `BlockSparseTensor` sharing the input's backend Arc.
pub(crate) fn bsp_dagger<T, S, B>(t: &BlockSparseTensor<T, S, B>) -> BlockSparseTensor<T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let td = t.data().dagger();
    Tensor::<BlockSparseStorage<T>, _, B>::with_backend(td, Arc::clone(t.backend_arc()))
}

/// Element-wise complex conjugate of a `DenseTensor`.
///
/// Goes through the legacy `Dense<T>` `conj` (since `DenseTensorData`
/// has no inherent `conj`), preserving the input's layout and backend.
pub(crate) fn dense_conj<T, B>(t: &DenseTensor<T, B>) -> DenseTensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let legacy = t.data().as_dense();
    let conj = legacy.conj();
    Tensor::<DenseStorage<T>, DenseLayout, B>::with_backend(
        conj.into_tensor_data(),
        Arc::clone(t.backend_arc()),
    )
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
