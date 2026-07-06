//! `DenseStorage<T>`: pure-data half of the dense tensor split.
//!
//! Carries only the flat element buffer. Shape and memory order live
//! on the paired [`DenseLayout`](crate::DenseLayout); the wrapper
//! [`DenseTensorData<T>`](crate::DenseTensorData) joins the two.
//!
//! `T: Scalar`-bound scalar-only data operations (`norm`,
//! `norm_frobenius`, `normalize`) live here because their bodies
//! touch only `data`; they require no shape, no memory order, and no
//! compute backend. Symmetric with the BSp side.

use std::ops::Mul;
use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};
use num_traits::Zero;

use super::Align64;
use crate::{Storage, StorageFor};

/// Pure-data half of the dense tensor split.
///
/// Holds a 64-byte-aligned flat buffer with Arc-based shared
/// ownership (Copy-on-Write via [`Arc::make_mut`]). Shape and memory
/// order are not carried here — they live on
/// [`DenseLayout`](crate::DenseLayout). For the full
/// storage + layout bundle, use
/// [`DenseTensorData<T>`](crate::DenseTensorData).
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64)
pub struct DenseStorage<T = f64> {
    data: Arc<AVec<T, Align64>>,
}

// Manual Clone impl: Arc<AVec<T, _>> is Clone regardless of T.
// #[derive(Clone)] would unnecessarily require T: Clone.
impl<T> Clone for DenseStorage<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
        }
    }
}

impl<T> DenseStorage<T> {
    /// Construct from a `Vec<T>`, internally rebuilding into a
    /// 64-byte-aligned buffer.
    pub fn new(data: Vec<T>) -> Self {
        let len = data.len();
        let mut aligned: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, len);
        for elem in data {
            aligned.push(elem);
        }
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

    /// Get a reference to the underlying contiguous data.
    pub fn data(&self) -> &[T] {
        &self.data[..]
    }

    /// Get a mutable reference to the underlying data (triggers CoW
    /// if shared).
    pub fn data_mut(&mut self) -> &mut [T]
    where
        T: Clone,
    {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// Iterate over all stored elements in flat (storage) order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data[..].iter()
    }

    /// Get pointer to the underlying data for FFI.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get mutable pointer to the underlying data for FFI (triggers
    /// CoW if shared).
    pub fn as_mut_ptr(&mut self) -> *mut T
    where
        T: Clone,
    {
        Arc::make_mut(&mut self.data).as_mut_ptr()
    }
}

impl<T> Storage for DenseStorage<T> {
    type Element = T;

    fn flat_len(&self) -> usize {
        self.data.len()
    }
}

impl<T> StorageFor<crate::DenseLayout> for DenseStorage<T> {}

// ---------------------------------------------------------------------------
// Length-preserving data operations
//
// Bodies only touch `data`, so they live on the storage half.
// Mirrors the BSp storage pattern.
// ---------------------------------------------------------------------------

impl<T> DenseStorage<T>
where
    T: Clone,
{
    /// Fill every stored element with a constant value (triggers CoW
    /// if shared).
    pub(crate) fn fill(&mut self, value: T) {
        Arc::make_mut(&mut self.data).as_mut_slice().fill(value);
    }

    /// Apply a function to each stored element in place (triggers CoW
    /// if shared).
    ///
    /// Element ordering follows storage layout; the closure sees raw
    /// flat positions without coordinate context.
    pub(crate) fn map_mut<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        let data = Arc::make_mut(&mut self.data).as_mut_slice();
        for x in data.iter_mut() {
            *x = f(x);
        }
    }

    /// Scale every stored element by a scalar factor in place
    /// (triggers CoW if shared).
    pub(crate) fn scale<S>(&mut self, factor: S)
    where
        T: Mul<S, Output = T>,
        S: Clone,
    {
        let data = Arc::make_mut(&mut self.data).as_mut_slice();
        for elem in data.iter_mut() {
            *elem = elem.clone() * factor.clone();
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar-only data operations
//
// Symmetric with `BlockSparseStorage`: norm-related operations read
// only the flat buffer (no shape, no layout-internal metadata).
// ---------------------------------------------------------------------------

impl<T> DenseStorage<T>
where
    T: ariadnetor_core::Scalar,
{
    /// Frobenius norm: √(Σ |element|²).
    pub(crate) fn norm_frobenius(&self) -> T::Real {
        crate::norm::frobenius_norm(&self.data[..])
    }

    /// Frobenius norm (alias for [`norm_frobenius`](Self::norm_frobenius)).
    pub(crate) fn norm(&self) -> T::Real {
        self.norm_frobenius()
    }

    /// Normalize to unit Frobenius norm (in-place).
    ///
    /// Returns the norm before normalization. Panics if the tensor has
    /// zero norm.
    pub(crate) fn normalize(&mut self) -> T::Real {
        let norm = self.norm_frobenius();
        assert!(norm != T::Real::zero(), "Cannot normalize zero tensor");
        let data = Arc::make_mut(&mut self.data);
        for elem in data.iter_mut() {
            *elem = elem.div_real(norm);
        }
        norm
    }
}
