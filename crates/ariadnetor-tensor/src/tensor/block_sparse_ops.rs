//! Pass-through inherent accessors on `BlockSparseTensor<T, S, B>`.
//!
//! The joined-form `Tensor` wraps a `TensorData<St, L>` plus a backend
//! Arc; for block-sparse tensors the same per-block / per-leg
//! introspection that `BlockSparseTensorData::block_data` and
//! `BlockSparseLayout::indices` already expose is needed by downstream
//! callers without forcing them to drill through `.data().layout().…`.
//! The pass-throughs below thread the joined form's accessors back up
//! to the `Tensor` surface so consumers can write `t.indices()`,
//! `t.block_data(&coord)`, etc.

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use num_traits::Float;

use super::Tensor;
use crate::{BlockCoord, BlockMeta, BlockSparseLayout, BlockSparseStorage, QNIndex, Sector};

impl<T, S, B> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, B>
where
    S: Sector,
    B: ComputeBackend,
{
    /// Per-leg `QNIndex` metadata (one entry per axis).
    pub fn indices(&self) -> &[QNIndex<S>] {
        self.data.layout().indices()
    }

    /// Overall flux label of the tensor.
    pub fn flux(&self) -> &S {
        self.data.layout().flux()
    }

    /// Number of flux-allowed blocks.
    pub fn num_blocks(&self) -> usize {
        self.data.layout().num_blocks()
    }

    /// Block metadata table (one entry per flux-allowed block).
    pub fn block_metas(&self) -> &[BlockMeta] {
        self.data.layout().block_metas()
    }

    /// Per-leg block shape for a stored block coordinate.
    ///
    /// Returns `None` if the coord is outside the layout's enumerated
    /// block set.
    pub fn block_shape(&self, coord: &BlockCoord) -> Option<Vec<usize>> {
        self.data.layout().block_shape(coord)
    }

    /// Data slice for a block identified by coordinate.
    ///
    /// Returns `None` if the block is not stored (zero by symmetry).
    pub fn block_data(&self, coord: &BlockCoord) -> Option<&[T]> {
        self.data.block_data(coord)
    }

    /// Mutable data slice for a block identified by coordinate
    /// (CoW-aware).
    pub fn block_data_mut(&mut self, coord: &BlockCoord) -> Option<&mut [T]>
    where
        T: Clone,
    {
        self.data.block_data_mut(coord)
    }
}

impl<T, S, B> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    /// Frobenius norm: `sqrt(Σ |x|^2)` over the packed flat buffer.
    pub fn norm(&self) -> T::Real {
        let mut sq = <T::Real as num_traits::Zero>::zero();
        for &x in self.data.storage().data() {
            let a = x.abs();
            sq = sq + a * a;
        }
        <T::Real as Float>::sqrt(sq)
    }

    /// Hermitian adjoint: element-wise conjugation, leg-direction flip,
    /// and flux dualization. Result shares the input's backend `Arc`.
    pub fn dagger(&self) -> Self {
        let td = self.data.dagger();
        Self {
            data: td,
            backend: Arc::clone(&self.backend),
        }
    }

    /// Element-wise complex conjugate. Result shares the input's
    /// backend `Arc`.
    pub fn conj(&self) -> Self {
        let td = self.data.conj();
        Self {
            data: td,
            backend: Arc::clone(&self.backend),
        }
    }
}
