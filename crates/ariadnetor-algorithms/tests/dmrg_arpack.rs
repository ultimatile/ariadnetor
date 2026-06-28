//! ARPACK-arm coverage for the 2-site DMRG step. Gated under the
//! `arpack` feature.
//!
//! - **Happy-path equivalence**: `dmrg_2site` with
//!   `LocalEigensolverParams::Arpack(...)` must converge to the same
//!   ground-state energy as the Lanczos arm on the same Heisenberg
//!   MPO + random MPS, within a tolerance comparable to both
//!   solvers' configured `tol`. This exercises the new ARPACK
//!   match arm in `dmrg_2site_step{,_block_sparse}` end-to-end.
//! - **Error-path forwarding**: `dmrg_2site_step` with ARPACK at
//!   `max_iter = 1` on a non-trivial Heff must surface
//!   `Err(DmrgHeffError::Arpack(ArpackError::MaxIterReached { .. }))`,
//!   pinning the new `From<ArpackError>` impl + the
//!   `DmrgHeffError::Arpack` variant.
//!
//! The Heisenberg MPO + random MPS builders come from the shared
//! `algorithms_fixtures::dense_fixtures` module.

#![cfg(feature = "arpack")]

use std::error::Error;

use algorithms_fixtures::dense_fixtures::{heisenberg_mpo_f64, random_mps_unknown_f64};
use approx::assert_abs_diff_eq;
use arnet_algorithms::dmrg::{
    DmrgEnvs, DmrgHeffError, DmrgSweepParams, LocalEigensolverParams, dmrg_2site, dmrg_2site_step,
};
use arnet_algorithms::krylov::{ArpackError, ArpackParams, LanczosParams};
use arnet_linalg::TruncSvdParams;
use arnet_native::NativeBackend;

// ---------------------------------------------------------------------------
// Happy path: ARPACK and Lanczos must agree on the converged energy.
// ---------------------------------------------------------------------------

#[test]
fn dmrg_arpack_matches_lanczos_on_heisenberg_n6() {
    let n = 6;
    let chi = 8;
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let psi0 = random_mps_unknown_f64(n, 2, chi, 0xDEAD_BEEF);

    let common = |eigensolver: LocalEigensolverParams| DmrgSweepParams {
        max_sweeps: 12,
        min_sweeps: 2,
        energy_tol: 1e-9,
        eigensolver,
        trunc: TruncSvdParams {
            chi_max: Some(chi),
            target_trunc_err: None,
        },
    };

    let p_lanczos = common(LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 200,
        tol: 1e-10,
        seed: Some(0xACED),
    }));
    let p_arpack = common(LocalEigensolverParams::Arpack(ArpackParams {
        tol: 1e-10,
        max_iter: 300,
        ncv: None,
    }));

    let (res_lan, _) = dmrg_2site(&mpo, &psi0, &p_lanczos).expect("lanczos run");
    let (res_arp, _) = dmrg_2site(&mpo, &psi0, &p_arpack).expect("arpack run");

    assert!(
        res_lan.converged,
        "Lanczos arm must converge on n=6 Heisenberg"
    );
    assert!(
        res_arp.converged,
        "ARPACK arm must report convergence on n=6 Heisenberg \
         (step-level flag is `Ok` from arpack_smallest, i.e. ARPACK's \
         relative-tol stopping criterion fired)"
    );
    // 1e-7 absorbs the per-step solver tolerances + sweep-truncation
    // residual at chi_max=8 without being so loose it would mask a
    // wired-wrong arm.
    assert_abs_diff_eq!(res_lan.energy, res_arp.energy, epsilon = 1e-7);
}

// ---------------------------------------------------------------------------
// Error path: ARPACK upstream `MaxIterReached` must round-trip through
// the new `DmrgHeffError::Arpack(_)` variant.
// ---------------------------------------------------------------------------

#[test]
fn dmrg_arpack_max_iter_one_returns_arpack_error() {
    let n = 4;
    let chi = 4;
    let mut psi = random_mps_unknown_f64(n, 2, chi, 0xCAFEBABE);
    psi.canonicalize(&NativeBackend::new(), 0);
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let mut envs = DmrgEnvs::build(&psi, &mpo).expect("envs build");
    // Walk the left env up so left(1) is populated for site=1.
    envs.advance_left(&psi, &mpo, 0).expect("advance_left(0)");

    // max_iter=1 with a tight relative tol is structurally
    // insufficient for a Heisenberg local Heff of dim ≥ 16 — ARPACK
    // returns `MaxIterReached` and the heff entry forwards it as
    // `DmrgHeffError::Arpack(_)` via the new `From<ArpackError>` impl.
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
