//! `DenseStorage<T>`: pure-data half of the dense tensor split.
//!
//! Carries only the flat element buffer. Shape and memory order live
//! on the paired [`DenseLayout`](crate::DenseLayout); the wrapper
//! [`DenseTensorData<T>`](crate::DenseTensorData) joins the two.

use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};

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

    /// Cheap O(1) clone of the underlying storage Arc.
    ///
    /// Internal kernel bridge for
    /// [`DenseTensorData::as_dense`](crate::DenseTensorData::as_dense)
    /// so the legacy [`Dense<T>`](crate::Dense) view can share the
    /// same aligned buffer without copying. Pub for cross-crate access
    /// from `arnet-linalg`; not a user-facing accessor.
    #[doc(hidden)]
    pub fn arc_clone(&self) -> Arc<AVec<T, Align64>> {
        Arc::clone(&self.data)
    }

    /// Construct directly from an already-aligned storage Arc.
    ///
    /// Internal kernel counterpart to [`arc_clone`](Self::arc_clone),
    /// used by [`Dense::into_tensor_data`](crate::Dense::into_tensor_data)
    /// to move ownership of the buffer without a copy. Pub for
    /// cross-crate access; not a user-facing constructor.
    #[doc(hidden)]
    pub fn from_arc(data: Arc<AVec<T, Align64>>) -> Self {
        Self { data }
    }
}

impl<T> Storage for DenseStorage<T> {
    type Element = T;

    fn flat_len(&self) -> usize {
        self.data.len()
    }
}

impl<T> StorageFor<crate::DenseLayout> for DenseStorage<T> {}
