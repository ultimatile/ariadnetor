//! `BlockSparseStorage<T>`: pure-data half of the block-sparse tensor split.
//!
//! Carries only the packed flat element buffer. Block metadata,
//! sector indices, flux, shape, and memory order live on the paired
//! [`BlockSparseLayout<S>`](crate::BlockSparseLayout); the wrapper
//! [`BlockSparseTensorData<T, S>`](crate::BlockSparseTensorData) joins
//! the two.
//!
//! `T: Scalar`-bound scalar-only data operations (`stored_len`,
//! `norm`, `norm_frobenius`, `normalize`, `normalized`) live here
//! because their bodies touch only `data`; they require no layout
//! field, no symmetry sector, and no compute backend.

use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};
use num_traits::{Float, One, Zero};

use crate::{Sector, Storage, StorageFor};

/// Pure-data half of the block-sparse tensor split.
///
/// Holds a 64-byte-aligned packed buffer of allowed-block elements
/// with Arc-based shared ownership (Copy-on-Write via
/// [`Arc::make_mut`]). The interpretation (which slab maps to which
/// block coordinate) lives on
/// [`BlockSparseLayout<S>`](crate::BlockSparseLayout); the
/// `BlockSparseTensorData<T, S>` wrapper joins them.
///
/// Notice that `BlockSparseStorage<T>` is sector-agnostic — the
/// symmetry sector parameter `S` lives only on the layout, not on
/// the storage.
pub struct BlockSparseStorage<T> {
    data: Arc<AVec<T, ConstAlign<64>>>,
}

// Manual Clone: Arc::clone does not require T: Clone (same pattern as DenseStorage<T>).
impl<T> Clone for BlockSparseStorage<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
        }
    }
}

impl<T> BlockSparseStorage<T> {
    /// Construct from a `Vec<T>`, internally rebuilding into a
    /// 64-byte-aligned buffer.
    pub fn new(data: Vec<T>) -> Self
    where
        T: Clone,
    {
        let aligned = AVec::from_slice(64, &data);
        Self {
            data: Arc::new(aligned),
        }
    }

    /// Construct from an already-aligned `AVec` (zero-copy).
    pub(crate) fn from_aligned(data: AVec<T, ConstAlign<64>>) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    /// Reference to the packed flat buffer.
    pub fn data(&self) -> &[T] {
        &self.data[..]
    }

    /// Mutable reference to the packed flat buffer (triggers CoW if
    /// shared).
    pub fn data_mut(&mut self) -> &mut [T]
    where
        T: Clone,
    {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// Mutable access to the underlying `Arc` (for CoW-aware paths).
    pub(crate) fn arc_mut(&mut self) -> &mut Arc<AVec<T, ConstAlign<64>>> {
        &mut self.data
    }
}

impl<T> Storage for BlockSparseStorage<T> {
    type Element = T;

    fn flat_len(&self) -> usize {
        self.data.len()
    }
}

impl<T, S: Sector> StorageFor<crate::BlockSparseLayout<S>> for BlockSparseStorage<T> {}

// ---------------------------------------------------------------------------
// Scalar-only data operations
//
// These read only `self.data` (no block metadata, no flux, no backend),
// so they live on the storage half. The joined-form
// `BlockSparseTensorData` re-exposes them via thin forwarders.
// ---------------------------------------------------------------------------

impl<T> BlockSparseStorage<T>
where
    T: Clone,
{
    /// Scale every stored element by a scalar factor in place
    /// (triggers CoW if shared).
    pub fn scale<S>(&mut self, factor: S)
    where
        T: std::ops::Mul<S, Output = T>,
        S: Clone,
    {
        let data = Arc::make_mut(&mut self.data).as_mut_slice();
        for elem in data.iter_mut() {
            *elem = elem.clone() * factor.clone();
        }
    }
}

impl<T> BlockSparseStorage<T>
where
    T: arnet_core::Scalar,
{
    /// Total number of stored elements across all blocks (= length of
    /// the flat packed buffer).
    pub fn stored_len(&self) -> usize {
        self.data.len()
    }

    /// Squared Frobenius norm: Σ |element|².
    fn norm_squared(&self) -> T::Real {
        self.data
            .iter()
            .map(|&x| {
                let a = x.abs();
                a * a
            })
            .fold(T::Real::zero(), |acc, x| acc + x)
    }

    /// Frobenius norm: √(Σ |element|²).
    pub fn norm_frobenius(&self) -> T::Real {
        self.norm_squared().sqrt()
    }

    /// Frobenius norm (alias for [`norm_frobenius`](Self::norm_frobenius)).
    pub fn norm(&self) -> T::Real {
        self.norm_frobenius()
    }

    /// Normalize to unit Frobenius norm (in-place).
    ///
    /// Returns the norm before normalization.
    /// Panics if the tensor has zero norm.
    pub fn normalize(&mut self) -> T::Real {
        let norm = self.norm_frobenius();
        assert!(norm != T::Real::zero(), "Cannot normalize zero tensor");
        let inv_norm = T::Real::one() / norm;
        let data = Arc::make_mut(&mut self.data);
        for elem in data.iter_mut() {
            *elem = elem.scale_real(inv_norm);
        }
        norm
    }

    /// Normalize and return a new tensor (out-of-place).
    ///
    /// Returns `(normalized_storage, original_norm)`.
    /// Panics if the tensor has zero norm.
    pub fn normalized(&self) -> (Self, T::Real) {
        let mut result = self.clone();
        let norm = result.normalize();
        (result, norm)
    }
}
