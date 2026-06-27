//! Behavioral tests for the high-level `dmrg_2site` wrapper.
//!
//! Coverage: wrapper-vs-manual equivalence (deterministic Lanczos
//! seed → bit-identical energies + diagnostic records), input
//! validation (`EmptyMps`, `LengthMismatch`), acceptance of arbitrary
//! caller canonical forms (including `CanonicalForm::Unknown`), and
//! defensive non-mutation of the caller's MPS.
//!
//! BlockSparse coverage is intentionally not duplicated here: the
//! wrapper has zero storage-specific logic (it is pure trait
//! dispatch over `DmrgOps + Clone`), so its correctness on the
//! BlockSparse path is already implied by the Dense equivalence
//! tests plus the BlockSparse `sweep_2site` validation suite.

use algorithms_fixtures::dense_fixtures::{heisenberg_mpo_f64, random_mps_unknown_f64};
use arnet_algorithms::dmrg::{
    DmrgEnvs, DmrgError, DmrgSweepParams, LocalEigensolverParams, dmrg_2site, sweep_2site,
};
use arnet_algorithms::krylov::LanczosParams;
use arnet_linalg::TruncSvdParams;
use arnet_mps::{CanonicalForm, Mpo, Mps, TensorChain, canonicalize};
use arnet_native::NativeBackend;
use arnet_tensor::{DenseLayout, DenseStorage};

fn small_params(seed: u64) -> DmrgSweepParams {
    DmrgSweepParams {
        max_sweeps: 4,
        min_sweeps: 1,
        energy_tol: 1e-10,
        eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 80,
            tol: 1e-10,
            seed: Some(seed),
        }),
        trunc: TruncSvdParams {
            chi_max: Some(16),
            target_trunc_err: None,
        },
    }
}

// ===========================================================================
// Equivalence: wrapper output is bit-identical to manual composition.
// ===========================================================================

#[test]
fn wrapper_dense_heisenberg_n4_matches_manual() {
    let n = 4;
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let psi0 = random_mps_unknown_f64(n, 2, 4, 0xC4F1);
    let params = small_params(0xACED);

    // Manual composition (mirroring the wrapper body).
    let mut psi_manual = psi0.clone();
    canonicalize(&NativeBackend::new(), &mut psi_manual, 0);
    let mut envs_manual = DmrgEnvs::build(&psi_manual, &mpo).expect("manual envs build");
    let result_manual =
        sweep_2site(&mut envs_manual, &mut psi_manual, &mpo, &params).expect("manual sweep");

    // Wrapper.
    let (result_wrapper, psi_wrapper) = dmrg_2site(&mpo, &psi0, &params).expect("wrapper");

    // Bit-identical scalars (Lanczos is deterministic with seeded RNG).
    assert_eq!(result_wrapper.energy, result_manual.energy);
    assert_eq!(result_wrapper.n_sweeps, result_manual.n_sweeps);
    assert_eq!(result_wrapper.converged, result_manual.converged);

    // Final canonical form contract.
    assert_eq!(
        *psi_wrapper.canonical_form(),
        CanonicalForm::Mixed { center: 0 },
    );

    // Per-step diagnostic records identical (sweep count, eigenvalues,
    // truncation errors, eigensolver iters / converged flag).
    assert_eq!(result_wrapper.sweeps.len(), result_manual.sweeps.len());
    for (sw_w, sw_m) in result_wrapper
        .sweeps
        .iter()
        .zip(result_manual.sweeps.iter())
    {
        assert_eq!(sw_w.sweep_energy, sw_m.sweep_energy);
        assert_eq!(sw_w.max_bond, sw_m.max_bond);
        assert_eq!(sw_w.steps.len(), sw_m.steps.len());
        for (st_w, st_m) in sw_w.steps.iter().zip(sw_m.steps.iter()) {
            assert_eq!(st_w.eigenvalue, st_m.eigenvalue);
            assert_eq!(st_w.bond_dim, st_m.bond_dim);
            assert_eq!(st_w.eigensolver_iters, st_m.eigensolver_iters);
            assert_eq!(st_w.eigensolver_converged, st_m.eigensolver_converged);
        }
    }

    // Per-site tensor data identical between wrapper output and manual run.
    for i in 0..n {
        assert_eq!(
            psi_wrapper.site(i).data_slice(),
            psi_manual.site(i).data_slice(),
            "site {i} tensor data diverged between wrapper and manual",
        );
    }
}

// ===========================================================================
// Acceptance: wrapper canonicalizes whatever caller form, including Unknown.
// ===========================================================================

#[test]
fn wrapper_accepts_unknown_canonical() {
    let n = 4;
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let psi0 = random_mps_unknown_f64(n, 2, 4, 0xC4F2);
    assert_eq!(*psi0.canonical_form(), CanonicalForm::Unknown);

    let params = small_params(0xACED);
    let (_result, psi_out) =
        dmrg_2site(&mpo, &psi0, &params).expect("wrapper should accept Unknown canonical form");
    assert_eq!(
        *psi_out.canonical_form(),
        CanonicalForm::Mixed { center: 0 },
    );
}

// ===========================================================================
// Validation: empty MPS surfaces before canonicalize would panic.
// ===========================================================================

#[test]
fn wrapper_rejects_empty_mps() {
    let mpo: Mpo<DenseStorage<f64>, DenseLayout> = Mpo::empty();
    let psi0: Mps<DenseStorage<f64>, DenseLayout> = Mps::empty();
    let params = small_params(0xACED);

    match dmrg_2site(&mpo, &psi0, &params) {
        Err(DmrgError::EmptyMps) => {}
        Err(e) => panic!("expected DmrgError::EmptyMps, got Err({e:?})"),
        Ok(_) => panic!("expected DmrgError::EmptyMps, got Ok(_)"),
    }
}

// ===========================================================================
// Validation: length mismatch caught before downstream layers.
// ===========================================================================

#[test]
fn wrapper_rejects_length_mismatch() {
    let mpo = heisenberg_mpo_f64(4, 1.0);
    let psi0 = random_mps_unknown_f64(3, 2, 4, 0xC4F3);
    let params = small_params(0xACED);

    match dmrg_2site(&mpo, &psi0, &params) {
        Err(DmrgError::LengthMismatch { mps: 3, mpo: 4 }) => {}
        Err(e) => panic!("expected LengthMismatch {{ mps: 3, mpo: 4 }}, got Err({e:?})"),
        Ok(_) => panic!("expected LengthMismatch {{ mps: 3, mpo: 4 }}, got Ok(_)"),
    }
}

// ===========================================================================
// Defensive clone: caller's `psi0` is not mutated by the wrapper.
// ===========================================================================

#[test]
fn wrapper_does_not_mutate_input() {
    let n = 4;
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let psi0 = random_mps_unknown_f64(n, 2, 4, 0xC4F4);
    let psi0_snapshot = psi0.clone();
    let params = small_params(0xACED);

    let _ = dmrg_2site(&mpo, &psi0, &params).expect("wrapper");

    assert_eq!(*psi0.canonical_form(), *psi0_snapshot.canonical_form());
    for i in 0..n {
        assert_eq!(
            psi0.site(i).data_slice(),
            psi0_snapshot.site(i).data_slice(),
            "site {i} tensor data was mutated by the wrapper",
        );
    }
}
