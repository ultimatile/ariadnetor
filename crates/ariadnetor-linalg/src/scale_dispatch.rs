//! Layout-keyed dispatch for diagonal scaling (`diagonal_scale`).
//!
//! [`LinalgScale`] is implemented on the concrete layout types
//! ([`DenseLayout`] and [`BlockSparseLayout<S>`]); each implementation pairs a
//! storage type via [`LinalgScale::Storage`], names its weight input type via
//! [`LinalgScale::Weights`], and routes to its storage-specific kernel. Callers
//! parameterized over `L: LinalgScale<T>` issue one generic call that serves
//! both flavors, mirroring [`LinalgDecompose`](crate::LinalgDecompose).
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
    StorageFor, Tensor, TensorLayout,
};

use crate::block_sparse_decomp::BlockScalars;
use crate::block_sparse_scale::diagonal_scale_block_sparse_dense;
use crate::error::LinalgError;
use crate::scalar_ops::diagonal_scale_dense;
use crate::tensor_bridge::check_bsp_data_layout_order_matches;

/// Layout-keyed dispatch trait for diagonal scaling.
///
/// Implemented for [`DenseLayout`] and [`BlockSparseLayout<S>`], each pairing a
/// storage type via [`Storage`](Self::Storage) and a weight input type via
/// [`Weights`](Self::Weights), and routing to its storage-specific kernel.
pub trait LinalgScale<T: Scalar>: TensorLayout + Sized {
    /// Storage type paired with this layout.
    type Storage: Storage + StorageFor<Self>;
    /// Weight input type for [`diagonal_scale`](Self::diagonal_scale). The
    /// `where Self: 'a` bound lets layouts whose sector type is non-`'static`
    /// borrow the weights for `'a`.
    type Weights<'a>
    where
        Self: 'a;

    /// Per-slice diagonal scaling along `axis`, using the supplied backend.
    fn diagonal_scale<B: OpsFor<Self::Storage>>(
        backend: &B,
        t: &Tensor<Self::Storage, Self>,
        weights: Self::Weights<'_>,
        axis: usize,
    ) -> Result<Tensor<Self::Storage, Self>, LinalgError>;
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T: Scalar> LinalgScale<T> for DenseLayout {
    type Storage = DenseStorage<T>;
    type Weights<'a> = &'a [T::Real];

    fn diagonal_scale<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        t: &Tensor<DenseStorage<T>, DenseLayout>,
        weights: &[T::Real],
        axis: usize,
    ) -> Result<Tensor<DenseStorage<T>, DenseLayout>, LinalgError> {
        let result = diagonal_scale_dense(backend, t.data(), weights, axis)?;
        Ok(Tensor::from_data(result))
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T: Scalar, S: Sector> LinalgScale<T> for BlockSparseLayout<S> {
    type Storage = BlockSparseStorage<T>;
    type Weights<'a>
        = &'a BlockScalars<T::Real, S>
    where
        Self: 'a;

    fn diagonal_scale<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        t: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        weights: &BlockScalars<T::Real, S>,
        axis: usize,
    ) -> Result<Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>, LinalgError> {
        check_bsp_data_layout_order_matches(t.data(), backend, "diagonal_scale_block_sparse")?;
        let result = diagonal_scale_block_sparse_dense(backend, t.data(), weights, axis)?;
        Ok(Tensor::from_data(result))
    }
}

// ---------------------------------------------------------------------------
// Unified free function — type-erase the layout into `L: LinalgScale<T>` so
// callers write `diagonal_scale(backend, &t, weights, axis)` without naming the
// storage. `L` resolves from `Tensor`'s second argument and `T` through
// `L::Storage`, the same inference the decomposition free fns rely on.
// ---------------------------------------------------------------------------

/// Per-slice diagonal scaling along `axis`, using the supplied backend.
pub fn diagonal_scale<T, L, B>(
    backend: &B,
    t: &Tensor<L::Storage, L>,
    weights: L::Weights<'_>,
    axis: usize,
) -> Result<Tensor<L::Storage, L>, LinalgError>
where
    T: Scalar,
    L: LinalgScale<T>,
    B: OpsFor<L::Storage>,
{
    L::diagonal_scale(backend, t, weights, axis)
}
