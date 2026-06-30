//! ARPACK happy-path equivalence for the 2-site DMRG driver. Gated
//! under the `arpack` feature.
//!
//! `dmrg_2site` with `LocalEigensolverParams::Arpack(...)` must
//! converge to the same ground-state energy as the Lanczos arm on the
//! same Heisenberg MPO + random MPS, within a tolerance comparable to
//! both solvers' configured `tol`. This exercises the ARPACK match arm
//! end-to-end through the public driver.
//!
//! Error-path forwarding is covered by an in-crate unit test, which
//! drives the crate-internal per-step entry point directly.
//!
//! The Heisenberg MPO + random MPS builders come from the shared
//! `algorithms_fixtures::dense_fixtures` module.

#![cfg(feature = "arpack")]

use algorithms_fixtures::dense_fixtures::{heisenberg_mpo_f64, random_mps_unknown_f64};
use approx::assert_abs_diff_eq;
use ariadnetor_algorithms::dmrg::{DmrgSweepParams, LocalEigensolverParams, dmrg_2site};
use ariadnetor_algorithms::krylov::{ArpackParams, LanczosParams};
use ariadnetor_linalg::TruncSvdParams;

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
