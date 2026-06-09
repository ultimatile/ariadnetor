//! Backend-capability scaffolding.
//!
//! [`OpsFor<St>`] is the compile-time half of capability dispatch: a
//! backend implements it for each storage flavor whose operations it
//! actually supports — the Kokkos `SpaceAccessibility` analogue. It is
//! deliberately not sealed, so out-of-tree backends (e.g. a future GPU
//! backend) can declare their own capability by implementing it.
//!
//! [`Host`] aliases the default host backend, so signatures can name the
//! substrate through one stable alias instead of spelling the concrete
//! backend type; repointing the substrate is then a one-line change.

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_native::NativeBackend;

use crate::{BlockSparseStorage, DenseStorage};

/// Compile-time marker: backend `Self` supports operations on storage
/// flavor `St`. Implemented selectively per (backend, storage) pair, so a
/// backend that cannot operate on a given storage simply omits the impl.
pub trait OpsFor<St>: ComputeBackend {}

impl<T: Scalar> OpsFor<DenseStorage<T>> for NativeBackend {}
impl<T: Scalar> OpsFor<BlockSparseStorage<T>> for NativeBackend {}

/// The default host compute substrate, aliased so signatures name it
/// through one stable alias rather than spelling the concrete backend
/// type; repointing the substrate is then a one-line change.
pub type Host = NativeBackend;

#[cfg(test)]
mod tests {
    use super::*;

    /// `OpsFor` is a marker, so the meaningful assertion is a compile-time
    /// bound check: this fails to compile if an impl is missing or if
    /// `Host` stops resolving to a backend that satisfies it.
    #[test]
    fn native_and_host_declare_ops_for_both_storage_flavors() {
        fn assert_ops_for<St, B: OpsFor<St>>() {}

        assert_ops_for::<DenseStorage<f64>, NativeBackend>();
        assert_ops_for::<BlockSparseStorage<f64>, NativeBackend>();
        assert_ops_for::<DenseStorage<f64>, Host>();
        assert_ops_for::<BlockSparseStorage<f64>, Host>();
    }
}
