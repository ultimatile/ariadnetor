//! Tensor-keyed dispatch for diagonal scaling (`diagonal_scale`).
//!
//! [`LinalgScale`] is implemented on the concrete tensor types
//! ([`Tensor<DenseStorage<T>, DenseLayout>`](arnet_tensor::Tensor) and
//! [`Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>`](arnet_tensor::Tensor));
//! each implementation pairs a storage type via [`LinalgScale::Storage`], names
//! its weight input type via [`LinalgScale::Weights`], and routes to its
//! storage-specific kernel. Callers parameterized over `Tn: LinalgScale<T>`
//! issue one generic call that serves both flavors, mirroring
//! [`LinalgDecompose`](crate::LinalgDecompose).
//!
//! The trait is sealed through a crate-private [`Sealed`](crate::sealed::Sealed)
//! supertrait, so it cannot be implemented downstream and projects no storage /
//! layout taxonomy onto its public bound surface — `Storage` survives only as a
//! sealed associated type. The method is an associated function taking
//! `t: &Self` (not `&self`) so it does not collide, under method-call
//! resolution, with the identically-named `diagonal_scale` receiver method on
//! the [`DenseHostOps`](crate::DenseHostOps) /
//! [`BlockSparseHostOps`](crate::BlockSparseHostOps) extension traits.
//!
//! # Weight type
//!
//! The dense flat `&[T::Real]` and block-sparse sectored
//! `&BlockScalars<T::Real, S>` weight forms are absorbed by the
//! [`Weights`](LinalgScale::Weights) associated type, the same way
//! [`LinalgDecompose`](crate::LinalgDecompose) absorbs the dense `(U, S, Vt)`
//! and block-sparse `(U, BlockScalars, Vt)` SVD-output difference. The
//! block-sparse weight type is exactly the decomposition's singular-value
//! output type; the dense weight type mirrors it as a flat slice of the same
//! real scalars.
//!
//! # Operation authority
//!
//! The method takes its compute backend explicitly at the call site, bound by
//! [`OpsFor<Self::Storage>`](arnet_tensor::OpsFor) — the same capability gate
//! the rest of the linalg surface enforces. `diagonal_scale` is an
//! allocation-only op (the backend drives no counted kernel), so there is no
//! policy-explicit variant. Block-sparse dispatch enforces the layout-order
//! invariant against the supplied backend before the per-sector work; the dense
//! path self-normalizes via the backend's preferred order.

use arnet_core::Scalar;
use arnet_tensor::{
    BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, OpsFor, Sector, Storage,
    Tensor,
};

use crate::block_sparse_decomp::BlockScalars;
use crate::block_sparse_scale::diagonal_scale_block_sparse_dense;
use crate::error::LinalgError;
use crate::scalar_ops::diagonal_scale_dense;
use crate::sealed::Sealed;
use crate::tensor_bridge::check_bsp_data_layout_order_matches;

/// Sealed tensor-keyed dispatch trait for diagonal scaling.
///
/// Implemented for the concrete dense and block-sparse tensor types, each
/// pairing a storage type via [`Storage`](Self::Storage) and a weight input
/// type via [`Weights`](Self::Weights), and routing to its storage-specific
/// kernel.
pub trait LinalgScale<T: Scalar>: Sealed + Sized {
    /// Storage type paired with this tensor.
    type Storage: Storage;
    /// Weight input type for [`diagonal_scale`](Self::diagonal_scale). The
    /// `where Self: 'a` bound lets tensors whose sector type is non-`'static`
    /// borrow the weights for `'a`.
    type Weights<'a>
    where
        Self: 'a;

    /// Per-slice diagonal scaling along `axis`, using the supplied backend.
    fn diagonal_scale<B: OpsFor<Self::Storage>>(
        backend: &B,
        t: &Self,
        weights: Self::Weights<'_>,
        axis: usize,
    ) -> Result<Self, LinalgError>;
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T: Scalar> LinalgScale<T> for Tensor<DenseStorage<T>, DenseLayout> {
    type Storage = DenseStorage<T>;
    type Weights<'a> = &'a [T::Real];

    fn diagonal_scale<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        t: &Self,
        weights: &[T::Real],
        axis: usize,
    ) -> Result<Self, LinalgError> {
        let result = diagonal_scale_dense(backend, t.data(), weights, axis)?;
        Ok(Tensor::from_data(result))
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T: Scalar, S: Sector> LinalgScale<T> for Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>> {
    type Storage = BlockSparseStorage<T>;
    type Weights<'a>
        = &'a BlockScalars<T::Real, S>
    where
        Self: 'a;

    fn diagonal_scale<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        t: &Self,
        weights: &BlockScalars<T::Real, S>,
        axis: usize,
    ) -> Result<Self, LinalgError> {
        check_bsp_data_layout_order_matches(t.data(), backend, "diagonal_scale_block_sparse")?;
        let result = diagonal_scale_block_sparse_dense(backend, t.data(), weights, axis)?;
        Ok(Tensor::from_data(result))
    }
}

// ---------------------------------------------------------------------------
// Unified free function — type-erase the tensor into `Tn: LinalgScale<T>` so
// callers write `diagonal_scale(backend, &t, weights, axis)` without naming the
// storage. `Tn` resolves from the tensor argument and `T` through the impl, the
// same inference the decomposition free fns rely on.
// ---------------------------------------------------------------------------

/// Per-slice diagonal scaling along `axis`, using the supplied backend.
pub fn diagonal_scale<T, Tn, B>(
    backend: &B,
    t: &Tn,
    weights: Tn::Weights<'_>,
    axis: usize,
) -> Result<Tn, LinalgError>
where
    T: Scalar,
    Tn: LinalgScale<T>,
    B: OpsFor<Tn::Storage>,
{
    Tn::diagonal_scale(backend, t, weights, axis)
}
