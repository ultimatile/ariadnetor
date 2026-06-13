//! Pass-through inherent accessors on `BlockSparseTensor<T, S>`.
//!
//! The joined-form `Tensor` wraps a `TensorData<St, L>`; for block-sparse
//! tensors the same per-block / per-leg introspection that
//! `BlockSparseTensorData::block_data` and `BlockSparseLayout::indices`
//! already expose is needed by downstream callers without forcing them to
//! drill through `.data().layout().…`. The pass-throughs below thread the
//! joined form's accessors back up to the `Tensor` surface so consumers
//! can write `t.indices()`, `t.block_data(&coord)`, etc.

use arnet_core::Scalar;
use num_traits::Float;

use super::Tensor;
use crate::{BlockCoord, BlockMeta, BlockSparseLayout, BlockSparseStorage, QNIndex, Sector};

impl<T, S> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    S: Sector,
{
    /// Memory order this tensor's flat block data is laid out in.
    ///
    /// Mirror of `DenseTensor::order` — saves block-sparse callers
    /// from reaching through `.data().layout().order()` for a basic
    /// layout property.
    pub fn order(&self) -> arnet_core::backend::MemoryOrder {
        self.data.layout().order()
    }

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

impl<T, S> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    T: Scalar,
    S: Sector,
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
    /// and flux dualization.
    pub fn dagger(&self) -> Self {
        let td = self.data.dagger();
        Self { data: td }
    }

    /// Element-wise complex conjugate.
    pub fn conj(&self) -> Self {
        let td = self.data.conj();
        Self { data: td }
    }
}
