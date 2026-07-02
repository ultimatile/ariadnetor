//! ARPACK error-path coverage for the crate-internal `dmrg_2site_step`.
//! It drives the per-step entry point directly to pin the
//! `From<ArpackError>` forwarding, so it reaches an internal surface and
//! lives next to the code. The happy-path equivalence test runs the
//! public `dmrg_2site` driver and stays an integration test
//! (`tests/dmrg_arpack.rs`). Gated under the `arpack` feature via the
//! parent module declaration.

use std::error::Error;

use algorithms_fixtures::dense_fixtures::{heisenberg_mpo_f64, random_mps_unknown_f64};
use ariadnetor_native::NativeBackend;

use crate::dmrg::heff::dmrg_2site_step;
use crate::dmrg::{DmrgHeffError, LocalEigensolverParams};
use crate::krylov::{ArpackError, ArpackParams};
use ariadnetor_linalg::TruncSvdParams;
use ariadnetor_mps::BraketEnvs;

// ---------------------------------------------------------------------------
// Error path: ARPACK upstream `MaxIterReached` must round-trip through
// the `DmrgHeffError::Arpack(_)` variant.
// ---------------------------------------------------------------------------

#[test]
fn dmrg_arpack_max_iter_one_returns_arpack_error() {
    let n = 4;
    let chi = 4;
    let mut psi = random_mps_unknown_f64(n, 2, chi, 0xCAFEBABE);
    psi.canonicalize(&NativeBackend::new(), 0);
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let mut envs = BraketEnvs::build(&psi, &mpo, &psi).expect("envs build");
    // Walk the left env up so left(1) is populated for site=1.
    envs.advance_left(&psi, &mpo, &psi, 0)
        .expect("advance_left(0)");

    // max_iter=1 with a tight relative tol is structurally
    // insufficient for a Heisenberg local Heff of dim ≥ 16 — ARPACK
    // returns `MaxIterReached` and the heff entry forwards it as
    // `DmrgHeffError::Arpack(_)` via the `From<ArpackError>` impl.
    let bad = LocalEigensolverParams::Arpack(ArpackParams {
        tol: 1e-12,
        max_iter: 1,
        ncv: None,
    });
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };

    let result = dmrg_2site_step(&envs, &psi, &mpo, 1, &bad, &trunc);
    assert!(
        matches!(
            result,
            Err(DmrgHeffError::Arpack(ArpackError::MaxIterReached { .. }))
        ),
        "expected DmrgHeffError::Arpack(ArpackError::MaxIterReached), got {result:?}",
    );

    // Contract: the wrapper keeps its Display to its own layer and
    // exposes the ARPACK diagnostic (iters / nconv / n_matvec for
    // `MaxIterReached`) through `source()`, not by folding it into
    // the wrapper's Display — so a `source()`-walking reporter does
    // not print the diagnostic twice.
    let err = result.expect_err("error path verified above");
    let inner = match &err {
        DmrgHeffError::Arpack(inner) => format!("{inner}"),
        _ => unreachable!(),
    };
    let outer = format!("{err}");
    assert!(
        !outer.contains(&inner),
        "DmrgHeffError::Arpack Display must stay self-layer, not embed the wrapped error: \
         outer = {outer:?}, inner = {inner:?}",
    );
    let source = err
        .source()
        .expect("Arpack variant must expose its child via source()");
    assert_eq!(
        source.to_string(),
        inner,
        "source() must reach the ARPACK diagnostic unchanged",
    );
}
