//! Internal helpers for the legacy [`Dense<T>`] / [`BlockSparse<T, S>`]
//! kernel representation ↔ user-facing
//! [`Tensor<St, L, B>`](arnet_tensor::Tensor) round-trip.
//!
//! Crate-internal only: every pub fn in `arnet-linalg` accepts
//! `&DenseTensor<T, B>` / `&BlockSparseTensor<T, S, B>` and immediately
//! views into the legacy storage form via these helpers. Kernel bodies
//! remain on `Dense<T>` / `BlockSparse<T, S>` during the #259 migration
//! window; once those legacy types are retired the helpers go with them.

use std::sync::Arc;

use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_tensor::{BlockSparse, BlockSparseTensor, Dense, DenseTensor, Sector};

/// Wrap a [`Dense<T>`] result into a [`DenseTensor<T, B>`] sharing the
/// reference tensor's backend Arc.
///
/// Used at the output boundary of every linalg pub fn: the legacy
/// kernel hands back a `Dense<T>`, which the joined `Tensor` form
/// adopts via [`Dense::into_tensor_data`].
pub(crate) fn wrap_dense<T, B>(dense: Dense<T>, backend: Arc<B>) -> DenseTensor<T, B>
where
    B: ComputeBackend,
{
    DenseTensor::with_backend(dense.into_tensor_data(), backend)
}

/// Wrap a [`Dense<T>`] result into a [`DenseTensor<U, B>`] where the
/// element type differs from the reference tensor (e.g. an `eigvalsh`
/// real-eigenvalue tensor wrapped against a complex input).
pub(crate) fn wrap_dense_as<U, B>(dense: Dense<U>, backend: Arc<B>) -> DenseTensor<U, B>
where
    B: ComputeBackend,
{
    DenseTensor::with_backend(dense.into_tensor_data(), backend)
}

/// Wrap a [`BlockSparse<T, S>`] result into a `BlockSparseTensor`
/// sharing the reference tensor's backend Arc and supplying the
/// missing memory order field.
pub(crate) fn wrap_block_sparse<T, S, B>(
    bsp: BlockSparse<T, S>,
    backend: Arc<B>,
    order: MemoryOrder,
) -> BlockSparseTensor<T, S, B>
where
    S: Sector,
    B: ComputeBackend,
{
    BlockSparseTensor::with_backend(bsp.into_tensor_data(order), backend)
}
