//! Crate-internal bridge helpers between the joined-form `Tensor`
//! surface and the legacy `Dense<T>` / `BlockSparseTensorData` paths
//! still used inside the mps kernels.
//!
//! `arnet::reorder` operates on the legacy `Dense<T>` type; callers
//! holding a `DenseTensor<T, B>` go through [`reorder_dense_tensor`].
//! Block-sparse `dagger` / `conj` are inherent on
//! `BlockSparseTensorData<T, S>`, so the corresponding wrappers
//! round-trip through `data()` and re-wrap with the cached backend.

use std::sync::Arc;

use arnet::{
    BlockSparseStorage, BlockSparseTensor, ComputeBackend, DenseLayout, DenseStorage, DenseTensor,
    MemoryOrder, Scalar, Sector, Tensor,
};

/// Reorder a `DenseTensor`'s flat data between memory orders.
///
/// Bridges through the legacy `Dense<T>` representation that
/// `arnet::reorder` operates on. The returned `DenseTensor` shares the
/// same backend `Arc` as the input.
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
    let legacy = t.data().as_dense();
    let reordered = arnet::reorder(&legacy, from, to);
    Tensor::<DenseStorage<T>, DenseLayout, B>::with_backend(
        reordered.into_tensor_data(),
        backend_arc,
    )
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
