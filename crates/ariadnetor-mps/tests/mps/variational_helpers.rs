//! Shared helpers for the dense and block-sparse variational-apply tests.

use ariadnetor_mps::{ApplyMethod, Mpo, Mps, MpsOps, TruncateParams, VariationalInit, inner};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{OpsFor, Storage, StorageFor, TensorLayout};

use super::helpers::apply_ok;

/// Variational fits of `op · psi` for checking where the sweep stops.
/// Returns `(fit A, fit B)`: A runs with a `tol` that cycle `stop` is the
/// first to fall within and a sweep cap well past `stop`; B runs with
/// `tol = 0` capped at `stop` sweeps.
///
/// `tol` is chosen from `tol = 0` fits of 1 to `stop + 1` sweeps: each
/// cycle must still raise the center overlap `‖φ_k‖²` (cycle `stop + 1`
/// included, so a fit that stops a cycle late returns a different state
/// than fit B), and `tol` is set to
/// three quarters of cycle `stop - 1`'s relative change, which must leave
/// every earlier cycle above it and cycle `stop` at or below it. `stop`
/// must be at least 3, so that cycle `stop - 1` has a change to measure.
pub(crate) fn variational_fit_at_first_converging_cycle<St, L>(
    op: &Mpo<St, L>,
    psi: &Mps<St, L>,
    params: &TruncateParams,
    stop: usize,
) -> (Mps<St, L>, Mps<St, L>)
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    Mps<St, L>: MpsOps<f64, Storage = St, Layout = L>,
    NativeBackend: OpsFor<St>,
{
    assert!(stop >= 3, "stop must be at least 3, got {stop}");
    let backend = NativeBackend::new();
    let fit = |max_sweeps: usize, tol: f64| {
        apply_ok::<f64, _, _, _>(
            &backend,
            op,
            psi,
            Some(params),
            ApplyMethod::Variational {
                init: VariationalInit::ZipUp,
                max_sweeps,
                tol,
            },
        )
    };

    let mut iterates: Vec<Mps<St, L>> = (1..=stop + 1).map(|k| fit(k, 0.0)).collect();
    let overlaps: Vec<f64> = iterates
        .iter()
        .map(|phi| inner::<f64, _, _, _>(&backend, phi, phi))
        .collect();
    for k in 1..overlaps.len() {
        assert!(
            overlaps[k] > overlaps[k - 1],
            "cycle {} must still raise the overlap at tol = 0: {} -> {}",
            k + 1,
            overlaps[k - 1],
            overlaps[k],
        );
    }
    // Relative change of the overlap over cycle `k` (1-based, k >= 2).
    let rel_change = |k: usize| (overlaps[k - 1] - overlaps[k - 2]) / overlaps[k - 1];
    let tol = 0.75 * rel_change(stop - 1);
    for k in 2..stop {
        assert!(
            rel_change(k) > tol,
            "cycle {k} must not converge at tol {tol}"
        );
    }
    assert!(
        rel_change(stop) <= tol,
        "cycle {stop}'s change {} must fall within tol {tol}",
        rel_change(stop),
    );

    (fit(50, tol), iterates.swap_remove(stop - 1))
}
