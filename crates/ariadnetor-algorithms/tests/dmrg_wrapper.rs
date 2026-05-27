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

use std::sync::Arc;

use arnet::TruncSvdParams;
use arnet::{ComputeBackend, DenseLayout, DenseStorage, DenseTensor, NativeBackend};
use arnet_algorithms::dmrg::{
    DmrgEnvs, DmrgError, DmrgSweepParams, LocalEigensolverParams, dmrg_2site, sweep_2site,
};
use arnet_algorithms::krylov::LanczosParams;
use arnet_mps::{CanonicalForm, Mpo, Mps, TensorChain, canonicalize};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

const D: usize = 2; // physical dim (spin-1/2)

// ---------------------------------------------------------------------------
// Minimal Heisenberg MPO + random MPS — inlined to keep the wrapper
// test file self-contained. Heisenberg is preferred over TFI here
// because the planned (post-this-issue) BlockSparse / U(1) wrapper
// path will reuse Heisenberg as the test Hamiltonian (TFI is not U(1)
// symmetric); the same builder body keeps the two test surfaces
// directly comparable when that follow-up lands.
// ---------------------------------------------------------------------------

type Op = fn(usize, usize) -> f64;

fn op_id(k: usize, b: usize) -> f64 {
    if k == b { 1.0 } else { 0.0 }
}

fn op_sz(k: usize, b: usize) -> f64 {
    if k == b {
        if k == 0 { 1.0 } else { -1.0 }
    } else {
        0.0
    }
}

// σ⁺ raises (|down⟩ → |up⟩); single non-zero element at (k_ket=1, b_bra=0).
fn op_sp(k: usize, b: usize) -> f64 {
    if k == 1 && b == 0 { 1.0 } else { 0.0 }
}

// σ⁻ lowers (|up⟩ → |down⟩); single non-zero element at (k_ket=0, b_bra=1).
fn op_sm(k: usize, b: usize) -> f64 {
    if k == 0 && b == 1 { 1.0 } else { 0.0 }
}

fn build_mpo_site_f64(
    w_l_dim: usize,
    w_r_dim: usize,
    cells: &[(usize, usize, Op, f64)],
) -> DenseTensor<f64> {
    let backend = NativeBackend::shared();
    let len = w_l_dim * D * D * w_r_dim;
    let mut data = vec![0.0_f64; len];
    for &(vl, vr, op, scale) in cells {
        for k in 0..D {
            for b in 0..D {
                let idx = vl + w_l_dim * (k + D * (b + D * vr));
                data[idx] += scale * op(k, b);
            }
        }
    }
    DenseTensor::from_raw_parts(
        data,
        vec![w_l_dim, D, D, w_r_dim],
        backend.preferred_order(),
        Arc::clone(&backend),
    )
}

fn heisenberg_mpo_f64(n: usize, j: f64) -> Mpo<DenseStorage<f64>, DenseLayout> {
    assert!(n >= 2, "heisenberg_mpo_f64 requires n >= 2");
    let mut sites = Vec::with_capacity(n);

    sites.push(build_mpo_site_f64(
        1,
        5,
        &[
            (0, 1, op_sm, 2.0 * j),
            (0, 2, op_sp, 2.0 * j),
            (0, 3, op_sz, j),
            (0, 4, op_id, 1.0),
        ],
    ));

    for _ in 1..n - 1 {
        sites.push(build_mpo_site_f64(
            5,
            5,
            &[
                (0, 0, op_id, 1.0),
                (1, 0, op_sp, 1.0),
                (2, 0, op_sm, 1.0),
                (3, 0, op_sz, 1.0),
                (4, 1, op_sm, 2.0 * j),
                (4, 2, op_sp, 2.0 * j),
                (4, 3, op_sz, j),
                (4, 4, op_id, 1.0),
            ],
        ));
    }

    sites.push(build_mpo_site_f64(
        5,
        1,
        &[
            (0, 0, op_id, 1.0),
            (1, 0, op_sp, 1.0),
            (2, 0, op_sm, 1.0),
            (3, 0, op_sz, 1.0),
        ],
    ));

    Mpo::from_sites(sites)
}

/// Build a random MPS with no canonical form set (`CanonicalForm::Unknown`).
/// The wrapper is responsible for canonicalizing internally.
fn random_mps_unknown_f64(n: usize, chi: usize, seed: u64) -> Mps<DenseStorage<f64>, DenseLayout> {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * D * r;
            let data: Vec<f64> = (0..len).map(|_| rng.random_range(-0.5_f64..0.5)).collect();
            DenseTensor::from_raw_parts(
                data,
                vec![l, D, r],
                backend.preferred_order(),
                Arc::clone(&backend),
            )
        })
        .collect();
    Mps::from_sites(storages)
}

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
    let psi0 = random_mps_unknown_f64(n, 4, 0xC4F1);
    let params = small_params(0xACED);

    // Manual composition (mirroring the wrapper body).
    let mut psi_manual = psi0.clone();
    canonicalize(&mut psi_manual, 0);
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
    let psi0 = random_mps_unknown_f64(n, 4, 0xC4F2);
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
    let backend = NativeBackend::shared();
    let mpo: Mpo<DenseStorage<f64>, DenseLayout> = Mpo::empty(Arc::clone(&backend));
    let psi0: Mps<DenseStorage<f64>, DenseLayout> = Mps::empty(Arc::clone(&backend));
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
    let psi0 = random_mps_unknown_f64(3, 4, 0xC4F3);
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
    let psi0 = random_mps_unknown_f64(n, 4, 0xC4F4);
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
