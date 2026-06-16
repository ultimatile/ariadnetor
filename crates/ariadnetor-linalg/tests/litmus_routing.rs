//! Pluggability-litmus routing observation.
//!
//! Under `--features pluggability-litmus`, `Host` aliases the stateful
//! `AltHostBackend` instead of the concrete `NativeBackend`. This binary
//! holds exactly one test, so its process-wide `Host::shared()` singleton
//! counter is observed without interference from parallel tests in the
//! same binary — a before/after delta is then a sound proof that the
//! host-ergonomic op routed its kernel through the aliased substrate
//! rather than a hard-coded native handle.
//!
//! When the feature is off the whole file compiles to nothing.
#![cfg(feature = "pluggability-litmus")]

use arnet_core::backend::ComputeBackend;
use arnet_linalg::DenseHostOps;
use arnet_tensor::ComputeBackendTensorExt;
use arnet_tensor::{DenseTensor, Host};

#[test]
fn host_ergonomic_op_routes_through_aliased_substrate() {
    // Sanity: the one-line alias swap actually took effect, so a green
    // delta below cannot be a false pass against the native substrate.
    assert_eq!(
        Host::shared().name(),
        "alt-host",
        "Host did not resolve to the litmus substrate under the feature",
    );

    // A host-ergonomic kernel op (dense SVD) defaults its backend to
    // `Host::shared()`; if it routes through the alias its dispatch must
    // bump the singleton counter.
    let before = Host::shared().count();
    let a = Host::shared().dense(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]);
    let _ = a.svd(1).expect("host SVD should succeed on a 2x2");
    let after = Host::shared().count();

    assert!(
        after > before,
        "host SVD did not dispatch through the aliased Host substrate \
         (counter {before} -> {after})",
    );
}
