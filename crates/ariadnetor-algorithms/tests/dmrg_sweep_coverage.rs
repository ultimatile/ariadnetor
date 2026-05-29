//! Additional `sweep_2site` coverage focused on tight assertions
//! that the smoke / convergence tests in `dmrg_sweep.rs` do not
//! make:
//!
//! - **Renormalization invariant**: `sweep_energy * <psi|psi>` equals
//!   `<psi|H|psi>.re()` for the post-truncation MPS, exercised under a
//!   fixture (`n=2` Heisenberg, `chi_max=1`) where truncation
//!   intentionally drops weight so `<psi|psi>` is meaningfully `!= 1`.
//! - **`completed_sweeps` counter**: with `min_sweeps == max_sweeps`
//!   and `tol = 0`, `result.n_sweeps` must reach exactly `max_sweeps`.
//! - **No premature break on tight tol**: with an unsatisfiable
//!   `energy_tol`, the run must exhaust `max_sweeps` instead of
//!   short-circuiting on a sign-flipped abs.
//! - **`validate_params` boundaries on `target_trunc_err`**: zero
//!   accepted, negative rejected, NaN rejected, positive accepted.

use std::sync::Arc;

use approx::assert_abs_diff_eq;
use arnet::Scalar;
use arnet::TruncSvdParams;
use arnet::{DenseLayout, DenseStorage, DenseTensor, NativeBackend};
use arnet_algorithms::dmrg::{
    DmrgEnvs, DmrgSweepError, DmrgSweepParams, LocalEigensolverParams, sweep_2site,
};
use arnet_algorithms::krylov::LanczosParams;
use arnet_mps::{Mpo, Mps, braket, canonicalize, norm};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

const D: usize = 2;

// ---------------------------------------------------------------------------
// Heisenberg MPO + random MPS fixtures (inlined; mirrors `dmrg_wrapper.rs`).
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

fn op_sp(k: usize, b: usize) -> f64 {
    if k == 1 && b == 0 { 1.0 } else { 0.0 }
}

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
    DenseTensor::from_raw_parts(data, vec![w_l_dim, D, D, w_r_dim], Arc::clone(&backend))
}

/// Spin-1/2 Heisenberg `H = J Σ S_i · S_{i+1}` as a bond-dim-5 MPO.
fn heisenberg_mpo_f64(n: usize, j: f64) -> Mpo<DenseStorage<f64>, DenseLayout> {
    assert!(n >= 2);
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

fn random_mps_center_zero_f64(
    n: usize,
    chi: usize,
    seed: u64,
) -> Mps<DenseStorage<f64>, DenseLayout> {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * D * r;
            let data: Vec<f64> = (0..len).map(|_| rng.random_range(-0.5_f64..0.5)).collect();
            DenseTensor::from_raw_parts(data, vec![l, D, r], Arc::clone(&backend))
        })
        .collect();
    let mut mps = Mps::from_sites(storages);
    canonicalize(&mut mps, 0);
    mps
}

fn psd_local_mpo_f64(n: usize, seed: u64) -> Mpo<DenseStorage<f64>, DenseLayout> {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let eps = 0.5_f64;
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            let g: Vec<f64> = (0..D * D)
                .map(|_| rng.random_range(-1.0_f64..1.0))
                .collect();
            let mut h = vec![0.0_f64; D * D];
            for s in 0..D {
                for t in 0..D {
                    let mut acc = 0.0;
                    for k in 0..D {
                        acc += g[k * D + s] * g[k * D + t];
                    }
                    h[s + D * t] = acc;
                }
                h[s + D * s] += eps;
            }
            DenseTensor::from_raw_parts(h, vec![1, D, D, 1], Arc::clone(&backend))
        })
        .collect();
    Mpo::from_sites(storages)
}

fn base_params(chi_max: Option<usize>, seed: u64) -> DmrgSweepParams {
    DmrgSweepParams {
        max_sweeps: 1,
        min_sweeps: 1,
        energy_tol: 0.0,
        eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 100,
            tol: 1e-10,
            seed: Some(seed),
        }),
        trunc: TruncSvdParams {
            chi_max,
            target_trunc_err: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Renormalization invariant — kills `sweep_energy = bra_h_ket / nrm_sq`
// arithmetic mutations (`*` ↔ `/`).
// ---------------------------------------------------------------------------

#[test]
fn sweep_energy_renormalizes_post_truncation() {
    // n=2 Heisenberg with chi_max=1: the GS is the entangled singlet, so
    // chi=1 truncation drops one singular value and the post-sweep
    // norm² is meaningfully below 1. Without renormalization the
    // returned `sweep_energy` would equal `<psi|H|psi>` (no divisor),
    // not the variational energy `<psi|H|psi> / <psi|psi>`.
    let n = 2;
    let mut mps = random_mps_center_zero_f64(n, 2, 0xA1A1);
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let mut envs: DmrgEnvs<DenseStorage<f64>, DenseLayout> =
        DmrgEnvs::build(&mps, &mpo).expect("build");
    let params = base_params(Some(1), 0xB00B);

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");

    let nrm = norm(&mps);
    let nrm_sq = nrm * nrm;
    let bra_h_ket = braket(&mps, &mpo, &mps).re();
    let sweep_energy = result.sweeps[0].sweep_energy;

    // Fixture sanity: the truncation must have actually dropped weight,
    // otherwise `nrm_sq == 1` and the renormalization is invisible.
    assert!(
        (nrm_sq - 1.0).abs() > 0.1,
        "fixture failed: nrm_sq = {nrm_sq} ≈ 1, kill not exercised",
    );

    // Renormalization equation: sweep_energy * <psi|psi> = <psi|H|psi>.
    // Mutation `*` → `/` makes nrm_sq = 1 → sweep_energy = bra_h_ket;
    // mutation `/` → `*` makes sweep_energy = bra_h_ket * nrm_sq.
    // The equality fails in both cases when nrm_sq != 1.
    assert_abs_diff_eq!(sweep_energy * nrm_sq, bra_h_ket, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// `result.n_sweeps == max_sweeps` exact equality — pinning the
// `completed_sweeps = sweep_idx + 1` arithmetic against off-by-one.
// ---------------------------------------------------------------------------

#[test]
fn n_sweeps_reaches_max_when_min_locked() {
    let n = 4;
    let mut mps = random_mps_center_zero_f64(n, 3, 0xC0DE);
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let mut envs: DmrgEnvs<DenseStorage<f64>, DenseLayout> =
        DmrgEnvs::build(&mps, &mpo).expect("build");
    let params = DmrgSweepParams {
        max_sweeps: 3,
        min_sweeps: 3,
        energy_tol: 0.0,
        eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 100,
            tol: 1e-10,
            seed: Some(0xBEEF),
        }),
        trunc: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
    };

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");

    // `min_sweeps == max_sweeps` and `energy_tol = 0` block the
    // convergence break; the loop must complete every iteration. The
    // mutation `+` → `*` yields `sweep_idx * 1 = sweep_idx`, leaving
    // `n_sweeps = max_sweeps - 1` (off-by-one). The `<=` style check
    // existing tests use lets that pass; this strict equality does not.
    assert_eq!(result.n_sweeps, params.max_sweeps);
}

// ---------------------------------------------------------------------------
// Tight unsatisfiable tol must not converge — defends the
// `delta.abs()` convergence path. A buggy `abs` that lets a negative
// delta pass through unchanged would satisfy `<= tol` for any tight
// positive tol and break out early.
// ---------------------------------------------------------------------------

#[test]
fn no_premature_convergence_on_tight_tol() {
    let n = 4;
    let mut mps = random_mps_center_zero_f64(n, 3, 0xD0D0);
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let mut envs: DmrgEnvs<DenseStorage<f64>, DenseLayout> =
        DmrgEnvs::build(&mps, &mpo).expect("build");
    let params = DmrgSweepParams {
        max_sweeps: 5,
        min_sweeps: 2,
        energy_tol: 1e-15,
        eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 100,
            tol: 1e-10,
            seed: Some(0xFEED),
        }),
        trunc: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
    };

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");

    // Monotone-decreasing energy with `chi_max=2 < required_chi`
    // produces sweep deltas of size O(1e-3) or larger — far above
    // `1e-15`. A buggy abs computation that lets a negative delta
    // satisfy `<= tol` would converge at `n_sweeps = 2` instead.
    assert!(
        !result.converged,
        "should not converge under tight tol; energies = {:?}",
        result
            .sweeps
            .iter()
            .map(|s| s.sweep_energy)
            .collect::<Vec<_>>()
    );
    assert_eq!(result.n_sweeps, params.max_sweeps);
}

// ---------------------------------------------------------------------------
// `validate_params` `target_trunc_err` boundary — exercises both the
// `!te.is_finite()` (NaN / inf) and `te < 0.0` (sign) predicates with
// inputs on each side of every comparison threshold.
// ---------------------------------------------------------------------------

type ValidationFixture = (
    DmrgEnvs<DenseStorage<f64>, DenseLayout>,
    Mps<DenseStorage<f64>, DenseLayout>,
    Mpo<DenseStorage<f64>, DenseLayout>,
    DmrgSweepParams,
);

fn small_validation_setup(target_trunc_err: Option<f64>) -> ValidationFixture {
    let n = 2;
    let mps = random_mps_center_zero_f64(n, 2, 0xE1E1);
    let mpo = psd_local_mpo_f64(n, 0xE2E2);
    let envs: DmrgEnvs<DenseStorage<f64>, DenseLayout> =
        DmrgEnvs::build(&mps, &mpo).expect("build");
    let mut params = base_params(Some(1), 0xE3E3);
    params.trunc.target_trunc_err = target_trunc_err;
    (envs, mps, mpo, params)
}

#[test]
fn validate_target_trunc_err_zero_accepted() {
    // `te = 0.0` accepted by the original `te < 0.0` predicate.
    // Mutation `<` → `==` (rejects only te==0) and `<` → `<=` (rejects
    // te<=0) would both turn this into `Err(InvalidParams)`.
    let (mut envs, mut mps, mpo, params) = small_validation_setup(Some(0.0));
    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params);
    assert!(
        result.is_ok(),
        "target_trunc_err=Some(0.0) must be accepted, got {:?}",
        result.err()
    );
}

#[test]
fn validate_target_trunc_err_positive_accepted() {
    // `te = 0.5` accepted by original. Mutation `<` → `>` would reject
    // any positive te.
    let (mut envs, mut mps, mpo, params) = small_validation_setup(Some(0.5));
    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params);
    assert!(
        result.is_ok(),
        "target_trunc_err=Some(0.5) must be accepted, got {:?}",
        result.err()
    );
}

#[test]
fn validate_target_trunc_err_negative_rejected() {
    // `te = -0.1` rejected by original. Mutation `<` → `>` would
    // accept negative te.
    let (mut envs, mut mps, mpo, params) = small_validation_setup(Some(-0.1));
    let err = sweep_2site(&mut envs, &mut mps, &mpo, &params)
        .expect_err("target_trunc_err=Some(-0.1) must be rejected");
    assert!(matches!(err, DmrgSweepError::InvalidParams { .. }));
}

#[test]
fn validate_target_trunc_err_nan_rejected() {
    // `te = NaN` rejected by `if !te.is_finite()`. Mutation
    // `delete !` flips the predicate, so finite te would be rejected
    // and NaN would slip through.
    let (mut envs, mut mps, mpo, params) = small_validation_setup(Some(f64::NAN));
    let err = sweep_2site(&mut envs, &mut mps, &mpo, &params)
        .expect_err("target_trunc_err=Some(NaN) must be rejected");
    assert!(matches!(err, DmrgSweepError::InvalidParams { .. }));
}
