//! Umbrella-surface test for the `arnet::expert` per-call-policy layer.
//!
//! Verifies the expert layer is fully expressible from `arnet` alone: the five
//! `expert::*` entry points are reachable, and `ExecPolicy` — their by-argument
//! policy knob — is constructible through the umbrella, so an umbrella-only
//! consumer needs no direct `arnet-core` / `arnet-linalg` dependency to pin a
//! policy. Naming every path through `arnet::` is the load-bearing part: each
//! line fails to compile unless the re-export exists.

use arnet::{DenseTensor, ExecPolicy, NativeBackend};

#[test]
fn expert_layer_reachable_through_umbrella() {
    let backend = NativeBackend::new();

    // Execute one op end to end through the umbrella, constructing the policy
    // via `arnet::ExecPolicy`. The shape assertion is a structural smoke test;
    // the per-op numerics are covered in the linalg crate's own tests.
    let mut t = DenseTensor::<f64>::zeros(vec![2, 3]);
    for (i, v) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().enumerate() {
        t.set([i / 3, i % 3], v);
    }
    let tt = arnet::expert::transpose(&backend, &t, &[1, 0], ExecPolicy::Sequential)
        .expect("expert::transpose via umbrella");
    assert_eq!(tt.shape(), &[3, 2]);

    // Reference the remaining four entry points: naming each generic fn item
    // proves the umbrella re-export resolves and its bounds hold for the
    // `(f64, NativeBackend)` instantiation, without asserting per-op numerics.
    let _ = arnet::expert::contract::<f64, NativeBackend>;
    let _ = arnet::expert::solve::<f64, NativeBackend>;
    let _ = arnet::expert::eigh::<f64, NativeBackend>;
    let _ = arnet::expert::eig::<f64, NativeBackend>;
}
