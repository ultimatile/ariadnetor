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

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::ComputeBackend;
use ariadnetor_native::NativeBackend;

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
///
/// The `pluggability-litmus` feature exercises exactly that one-line
/// repoint: it swaps the substrate to `AltHostBackend`, a distinct
/// stateful backend, so the whole host-pinned surface is proven to hold
/// against a substrate that is not the concrete native type. The litmus
/// build is a standalone check (`cargo make litmus`), not part of the
/// default gate.
#[cfg(not(feature = "pluggability-litmus"))]
pub type Host = NativeBackend;

/// Litmus substrate: under `pluggability-litmus`, `Host` resolves to the
/// alternate backend instead of `NativeBackend`, proving the substrate is
/// swappable in one line.
#[cfg(feature = "pluggability-litmus")]
pub type Host = alt_host::AltHostBackend;

/// Pluggability-litmus alternate host backend.
///
/// A distinct, stateful (non-zero-sized) backend that delegates every
/// kernel to an inner [`NativeBackend`] while counting dispatches. Made
/// the [`Host`] substrate by the feature-gated alias above, it proves
/// the call-site-backend design holds against a substrate that is not
/// the concrete native type: every `Host::shared()` call site, every
/// `host_order()` constructor, and every `&Host` / `Arc<Host>` /
/// `Host: OpsFor<…>` use type-checks and runs against this type. The
/// dispatch counter lets a routing test observe that host-ergonomic
/// paths actually reach the aliased substrate rather than a hard-coded
/// native handle.
#[cfg(feature = "pluggability-litmus")]
mod alt_host {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, OnceLock};

    use ariadnetor_core::Scalar;
    use ariadnetor_core::backend::{
        BackendError, ComputeBackend, DeviceType, EigDescriptor, EighDescriptor, ExecPolicy,
        GemmDescriptor, LqDescriptor, MemoryOrder, QrDescriptor, SolveDescriptor, SvdDescriptor,
        TransposeDescriptor, TridiagEighDescriptor,
    };
    use ariadnetor_native::NativeBackend;

    use crate::{BlockSparseStorage, DenseStorage, OpsFor};

    /// See the module docs: a stateful native delegate that counts kernel
    /// dispatches, used as the `Host` substrate under the litmus feature.
    pub struct AltHostBackend {
        inner: NativeBackend,
        kernel_calls: AtomicUsize,
    }

    impl AltHostBackend {
        fn new() -> Self {
            Self {
                inner: NativeBackend::new(),
                kernel_calls: AtomicUsize::new(0),
            }
        }

        /// Shared singleton, mirroring [`NativeBackend::shared`] so the
        /// `Host::shared()` call sites resolve unchanged under the alias.
        pub fn shared() -> Arc<AltHostBackend> {
            static INSTANCE: OnceLock<Arc<AltHostBackend>> = OnceLock::new();
            INSTANCE
                .get_or_init(|| Arc::new(AltHostBackend::new()))
                .clone()
        }

        /// Number of kernel dispatches routed through this backend.
        pub fn count(&self) -> usize {
            self.kernel_calls.load(Ordering::SeqCst)
        }

        fn bump(&self) {
            self.kernel_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl ComputeBackend for AltHostBackend {
        fn name(&self) -> &'static str {
            "alt-host"
        }

        fn device_type(&self) -> DeviceType {
            self.inner.device_type()
        }

        fn preferred_order(&self) -> MemoryOrder {
            self.inner.preferred_order()
        }

        fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
            self.bump();
            self.inner.gemm(desc)
        }

        fn transpose<T: Scalar>(
            &self,
            desc: TransposeDescriptor<'_, T>,
        ) -> Result<(), BackendError> {
            self.bump();
            self.inner.transpose(desc)
        }

        fn svd<T: Scalar>(&self, desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
            self.bump();
            self.inner.svd(desc)
        }

        fn qr<T: Scalar>(&self, desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
            self.bump();
            self.inner.qr(desc)
        }

        fn lq<T: Scalar>(&self, desc: LqDescriptor<'_, T>) -> Result<(), BackendError> {
            self.bump();
            self.inner.lq(desc)
        }

        fn eigh<T: Scalar>(&self, desc: EighDescriptor<'_, T>) -> Result<(), BackendError> {
            self.bump();
            self.inner.eigh(desc)
        }

        fn tridiag_eigh<T: Scalar>(
            &self,
            desc: TridiagEighDescriptor<'_, T>,
        ) -> Result<(), BackendError> {
            self.bump();
            self.inner.tridiag_eigh(desc)
        }

        fn eig<T: Scalar>(&self, desc: EigDescriptor<'_, T>) -> Result<(), BackendError> {
            self.bump();
            self.inner.eig(desc)
        }

        fn solve<T: Scalar>(&self, desc: SolveDescriptor<'_, T>) -> Result<(), BackendError> {
            self.bump();
            self.inner.solve(desc)
        }

        // Policy hooks carry no compute authority but must mirror the native
        // substrate's hardware-aware thresholds, so delegate all eight.
        fn par_for_svd(&self, m: usize, n: usize) -> ExecPolicy {
            self.inner.par_for_svd(m, n)
        }

        fn par_for_qr(&self, m: usize, n: usize) -> ExecPolicy {
            self.inner.par_for_qr(m, n)
        }

        fn par_for_lq(&self, m: usize, n: usize) -> ExecPolicy {
            self.inner.par_for_lq(m, n)
        }

        fn par_for_eigh(&self, n: usize) -> ExecPolicy {
            self.inner.par_for_eigh(n)
        }

        fn par_for_tridiag_eigh(&self, n: usize) -> ExecPolicy {
            self.inner.par_for_tridiag_eigh(n)
        }

        fn par_for_eig(&self, n: usize) -> ExecPolicy {
            self.inner.par_for_eig(n)
        }

        fn par_for_gemm(&self, m: usize, n: usize, k: usize) -> ExecPolicy {
            self.inner.par_for_gemm(m, n, k)
        }

        fn par_for_solve(&self, n: usize, nrhs: usize) -> ExecPolicy {
            self.inner.par_for_solve(n, nrhs)
        }

        fn par_for_transpose(&self, shape: &[usize]) -> ExecPolicy {
            self.inner.par_for_transpose(shape)
        }
    }

    // Delegating every kernel to a full `NativeBackend`, the litmus backend
    // genuinely supports both storage flavors and declares the capability
    // exactly as an out-of-tree backend would (`OpsFor` is unsealed), so it
    // can stand in for `Host` on the `OpsFor`-gated public surface.
    impl<T: Scalar> OpsFor<DenseStorage<T>> for AltHostBackend {}
    impl<T: Scalar> OpsFor<BlockSparseStorage<T>> for AltHostBackend {}
}

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
