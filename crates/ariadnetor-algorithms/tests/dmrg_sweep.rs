//! Tests for the DMRG 2-site sweep driver (`sweep` module).
//!
//! Strategy mirrors `dmrg_heff.rs`: small chains with bond-dim-1
//! Hermitian product MPOs whose ground state is itself a product
//! state, plus targeted edge-case fixtures for n_sites=2, c64,
//! Lanczos-non-convergence, and error variants. The PSD-product
//! variant gives a closed-form ground-energy reference
//! (`prod_i min_eig(h_i)`) without a separate ED solver.

use approx::assert_abs_diff_eq;
use arnet_algorithms::dmrg::{
    DmrgEnvs, DmrgResult, DmrgSweepError, DmrgSweepParams, SweepDirection, dmrg_2site_sweep,
};
use arnet_algorithms::krylov::LanczosParams;
use arnet_linalg::{TruncSvdParams, eigh};
use arnet_mps::{CanonicalForm, Mpo, Mps, TensorChain, canonicalize};
use arnet_native::NativeBackend;
use arnet_tensor::{ComputeBackendTensorExt, Dense};
use num_complex::Complex;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Random-but-seeded MPS in `Mixed { center: 0 }` form (sweep driver's accepted entry state).
fn random_mps_center_zero_f64(n: usize, d: usize, chi: usize, seed: u64) -> Mps<Dense<f64>> {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<Dense<f64>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * d * r;
            let data: Vec<f64> = (0..len).map(|_| rng.random_range(-0.5_f64..0.5)).collect();
            backend.make_tensor(data, vec![l, d, r])
        })
        .collect();
    let mut mps = Mps::from_storages(storages);
    canonicalize(&mut mps, 0);
    mps
}

fn random_mps_center_zero_c64(
    n: usize,
    d: usize,
    chi: usize,
    seed: u64,
) -> Mps<Dense<Complex<f64>>> {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<Dense<Complex<f64>>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * d * r;
            let data: Vec<Complex<f64>> = (0..len)
                .map(|_| {
                    let re = rng.random_range(-0.5_f64..0.5);
                    let im = rng.random_range(-0.5_f64..0.5);
                    Complex::new(re, im)
                })
                .collect();
            backend.make_tensor(data, vec![l, d, r])
        })
        .collect();
    let mut mps = Mps::from_storages(storages);
    canonicalize(&mut mps, 0);
    mps
}

/// Per-site PSD Hermitian `h_i = G^† G + ε I`. `h` and MPO storage are
/// column-major (`make_tensor` reads `backend.preferred_order()`).
fn psd_local_mpo_f64(n: usize, d: usize, seed: u64) -> (Mpo<Dense<f64>>, Vec<Vec<f64>>) {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let eps = 0.5_f64;
    let mut hs: Vec<Vec<f64>> = Vec::with_capacity(n);
    let storages: Vec<Dense<f64>> = (0..n)
        .map(|_| {
            // `g` is internal scratch; only `h`'s layout has to match the backend.
            let g: Vec<f64> = (0..d * d)
                .map(|_| rng.random_range(-1.0_f64..1.0))
                .collect();
            // h = G^T G + eps * I (PSD, real-symmetric).
            let mut h = vec![0.0_f64; d * d];
            for s in 0..d {
                for t in 0..d {
                    let mut acc = 0.0;
                    for k in 0..d {
                        acc += g[k * d + s] * g[k * d + t];
                    }
                    h[s + d * t] = acc;
                }
                h[s + d * s] += eps;
            }
            hs.push(h.clone());
            backend.make_tensor(h, vec![1, d, d, 1])
        })
        .collect();
    (Mpo::from_storages(storages), hs)
}

type C64Mpo = Mpo<Dense<Complex<f64>>>;
type C64SiteMatrix = Vec<Complex<f64>>;

fn psd_local_mpo_c64(n: usize, d: usize, seed: u64) -> (C64Mpo, Vec<C64SiteMatrix>) {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let eps = Complex::new(0.5_f64, 0.0);
    let mut hs: Vec<Vec<Complex<f64>>> = Vec::with_capacity(n);
    let storages: Vec<Dense<Complex<f64>>> = (0..n)
        .map(|_| {
            let g: Vec<Complex<f64>> = (0..d * d)
                .map(|_| {
                    let re = rng.random_range(-1.0_f64..1.0);
                    let im = rng.random_range(-1.0_f64..1.0);
                    Complex::new(re, im)
                })
                .collect();
            // h = G^† G + eps I, stored column-major to match NativeBackend.
            let mut h = vec![Complex::new(0.0, 0.0); d * d];
            for s in 0..d {
                for t in 0..d {
                    let mut acc = Complex::new(0.0, 0.0);
                    for k in 0..d {
                        acc += g[k * d + s].conj() * g[k * d + t];
                    }
                    h[s + d * t] = acc;
                }
                h[s + d * s] += eps;
            }
            hs.push(h.clone());
            backend.make_tensor(h, vec![1, d, d, 1])
        })
        .collect();
    (Mpo::from_storages(storages), hs)
}

/// Hermitian (real-symmetric, mixed-sign) bond-dim-1 product MPO.
/// `m` is stored column-major to match NativeBackend.
fn hermitian_local_mpo_f64(n: usize, d: usize, seed: u64) -> Mpo<Dense<f64>> {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<Dense<f64>> = (0..n)
        .map(|_| {
            let r: Vec<f64> = (0..d * d)
                .map(|_| rng.random_range(-1.0_f64..1.0))
                .collect();
            let mut m = vec![0.0_f64; d * d];
            for s in 0..d {
                for t in 0..d {
                    m[s + d * t] = 0.5 * (r[s * d + t] + r[t * d + s]);
                }
            }
            backend.make_tensor(m, vec![1, d, d, 1])
        })
        .collect();
    Mpo::from_storages(storages)
}

/// Smallest eigenvalue of a real-symmetric `d×d` matrix; `h` is CM.
fn min_eig_real_sym(h: &[f64], d: usize) -> f64 {
    let backend = NativeBackend::new();
    let m = backend.make_tensor(h.to_vec(), vec![d, d]);
    let (eigvals, _eigvecs) = eigh(&backend, &m, 1).expect("eigh");
    eigvals.data().iter().cloned().fold(f64::INFINITY, f64::min)
}

/// Smallest eigenvalue of a complex Hermitian `d×d` matrix (CM, see above).
fn min_eig_complex_herm(h: &[Complex<f64>], d: usize) -> f64 {
    let backend = NativeBackend::new();
    let m = backend.make_tensor(h.to_vec(), vec![d, d]);
    let (eigvals, _eigvecs) = eigh(&backend, &m, 1).expect("eigh");
    eigvals.data().iter().cloned().fold(f64::INFINITY, f64::min)
}

fn standard_params_f64(seed: u64) -> DmrgSweepParams {
    DmrgSweepParams {
        max_sweeps: 20,
        min_sweeps: 1,
        energy_tol: 1e-10,
        lanczos: LanczosParams {
            max_iter: 200,
            tol: 1e-10,
            seed: Some(seed),
        },
        trunc: TruncSvdParams {
            chi_max: Some(1),
            target_trunc_err: None,
        },
    }
}

// ---------------------------------------------------------------------------
// T1 — convergence on the PSD-product fixture (closed-form reference)
// ---------------------------------------------------------------------------
#[test]
fn t1_psd_product_converges_to_product_of_min_eigs_f64() {
    let n = 4;
    let d = 3;
    let mut mps = random_mps_center_zero_f64(n, d, 4, 0xA1A1);
    let (mpo, hs) = psd_local_mpo_f64(n, d, 0x1234);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let params = standard_params_f64(0xB00B);

    let result = dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");

    let reference = hs.iter().map(|h| min_eig_real_sym(h, d)).product::<f64>();
    assert_abs_diff_eq!(result.energy, reference, epsilon = 1e-7);
    assert!(result.converged, "should converge well within max_sweeps");
    assert!(
        result.n_sweeps <= params.max_sweeps,
        "n_sweeps {} > max_sweeps {}",
        result.n_sweeps,
        params.max_sweeps
    );
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

// ---------------------------------------------------------------------------
// T2 — monotone non-increasing post-truncation energy
// ---------------------------------------------------------------------------
#[test]
fn t2_energy_monotone_nonincreasing_across_sweeps() {
    let n = 4;
    let d = 2;
    let mut mps = random_mps_center_zero_f64(n, d, 4, 0xC0DE);
    let mpo = hermitian_local_mpo_f64(n, d, 0xF00D);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let params = DmrgSweepParams {
        max_sweeps: 10,
        min_sweeps: 10, // force all sweeps to run for a full energy trace.
        energy_tol: 0.0,
        lanczos: LanczosParams {
            max_iter: 200,
            tol: 1e-12,
            seed: Some(0xBEEF),
        },
        trunc: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
    };

    let result = dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");
    assert_eq!(result.sweeps.len(), params.max_sweeps);
    let energies: Vec<f64> = result.sweeps.iter().map(|s| s.sweep_energy).collect();
    for w in energies.windows(2) {
        // Allow tiny upward noise; truly increasing energy is a regression.
        assert!(
            w[1] - w[0] <= 1e-10,
            "energy increased: {} -> {} (delta {:e})",
            w[0],
            w[1],
            w[1] - w[0]
        );
    }
}

// ---------------------------------------------------------------------------
// T3 — boundary sites covered (site=0 and site=N-2 per direction)
// ---------------------------------------------------------------------------
#[test]
fn t3_boundary_sites_covered_each_sweep() {
    let n = 4;
    let d = 2;
    let mut mps = random_mps_center_zero_f64(n, d, 3, 0x33);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0x44);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let params = DmrgSweepParams {
        max_sweeps: 3,
        min_sweeps: 3,
        energy_tol: 0.0,
        lanczos: LanczosParams {
            max_iter: 100,
            tol: 1e-10,
            seed: Some(0x55),
        },
        trunc: TruncSvdParams {
            chi_max: Some(1),
            target_trunc_err: None,
        },
    };

    let result = dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");
    for sweep in &result.sweeps {
        let mut saw = (false, false, false, false);
        for step in &sweep.steps {
            match (step.direction, step.site) {
                (SweepDirection::LeftToRight, 0) => saw.0 = true,
                (SweepDirection::LeftToRight, s) if s == n - 2 => saw.1 = true,
                (SweepDirection::RightToLeft, 0) => saw.2 = true,
                (SweepDirection::RightToLeft, s) if s == n - 2 => saw.3 = true,
                _ => {}
            }
        }
        assert!(
            saw.0 && saw.1 && saw.2 && saw.3,
            "sweep {} missing a boundary step: {:?}",
            sweep.sweep,
            saw
        );
    }
}

// ---------------------------------------------------------------------------
// T4 — env freshness via functional equivalence
// ---------------------------------------------------------------------------
#[test]
fn t4_envs_functionally_equivalent_to_fresh_rebuild() {
    let n = 4;
    let d = 2;
    let mut mps = random_mps_center_zero_f64(n, d, 3, 0x4444);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0x5555);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    // First, run one full cycle.
    let prep_params = DmrgSweepParams {
        max_sweeps: 1,
        min_sweeps: 1,
        energy_tol: 0.0,
        lanczos: LanczosParams {
            max_iter: 100,
            tol: 1e-10,
            seed: Some(0x42),
        },
        trunc: TruncSvdParams {
            chi_max: Some(1),
            target_trunc_err: None,
        },
    };
    dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &prep_params).expect("prep");

    // Snapshot the post-prep MPS and run the comparison sweep twice:
    // (a) with the incremental envs we just maintained,
    // (b) with a fresh DmrgEnvs::build from the snapshot.
    let mut mps_a = mps.clone();
    let mut envs_a = envs.clone();
    let mut mps_b = mps.clone();
    let mut envs_b: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps_b, &mpo).expect("rebuild");

    let cmp_params = DmrgSweepParams {
        max_sweeps: 1,
        min_sweeps: 1,
        energy_tol: 0.0,
        lanczos: LanczosParams {
            max_iter: 100,
            tol: 1e-10,
            // SEED PINNED so the two parallel runs draw identical
            // initial Lanczos vectors.
            seed: Some(0xDEAD),
        },
        trunc: TruncSvdParams {
            chi_max: Some(1),
            target_trunc_err: None,
        },
    };

    let res_a = dmrg_2site_sweep(&mut envs_a, &mut mps_a, &mpo, &cmp_params).expect("a");
    let res_b = dmrg_2site_sweep(&mut envs_b, &mut mps_b, &mpo, &cmp_params).expect("b");

    assert_abs_diff_eq!(res_a.energy, res_b.energy, epsilon = 1e-9);
    let steps_a = &res_a.sweeps[0].steps;
    let steps_b = &res_b.sweeps[0].steps;
    assert_eq!(steps_a.len(), steps_b.len());
    for (sa, sb) in steps_a.iter().zip(steps_b.iter()) {
        assert_abs_diff_eq!(sa.eigenvalue, sb.eigenvalue, epsilon = 1e-9);
        assert_abs_diff_eq!(sa.trunc_err, sb.trunc_err, epsilon = 1e-9);
    }
    // Final tensors element-wise close.
    for i in 0..n {
        let a = mps_a.storage(i).data();
        let b = mps_b.storage(i).data();
        assert_eq!(mps_a.storage(i).shape(), mps_b.storage(i).shape());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_abs_diff_eq!(*x, *y, epsilon = 1e-9);
        }
    }
}

// ---------------------------------------------------------------------------
// T5 — n_sites = 2 edge case
// ---------------------------------------------------------------------------
#[test]
fn t5_n_sites_two_edge_case() {
    let n = 2;
    let d = 2;
    let mut mps = random_mps_center_zero_f64(n, d, 2, 0x77);
    let (mpo, hs) = psd_local_mpo_f64(n, d, 0x88);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let params = standard_params_f64(0x99);

    let result = dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");
    let reference = hs.iter().map(|h| min_eig_real_sym(h, d)).product::<f64>();
    assert_abs_diff_eq!(result.energy, reference, epsilon = 1e-7);
    assert!(result.converged);
    // Each sweep at n=2 has exactly one L→R step (site=0) and one R→L step (site=0).
    for sweep in &result.sweeps {
        assert_eq!(sweep.steps.len(), 2);
    }
}

// ---------------------------------------------------------------------------
// T6 — error variants
// ---------------------------------------------------------------------------
#[test]
fn t6_length_mismatch_mps_vs_envs() {
    // Build envs against (mps_a, mpo_a) of length 4, then call sweep
    // with a length-3 mps — the env's n_sites disagrees with mps.len().
    let n_a = 4;
    let n_b = 3;
    let d = 2;
    let mut mps_a = random_mps_center_zero_f64(n_a, d, 2, 0x10);
    let (mpo_a, _) = psd_local_mpo_f64(n_a, d, 0x11);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps_a, &mpo_a).expect("build");
    let mut mps_b = random_mps_center_zero_f64(n_b, d, 2, 0x12);
    // We still need an MPO of *some* length; the function checks both
    // mps and mpo against envs.n_sites.
    let (mpo_b, _) = psd_local_mpo_f64(n_b, d, 0x13);

    // First sweep with the original (consistent) triple to make sure
    // the setup itself is healthy, then break the contract.
    let _ = dmrg_2site_sweep(&mut envs, &mut mps_a, &mpo_a, &standard_params_f64(0x14)).unwrap();

    // Now break it: pass mismatched chain to the same envs.
    let err = dmrg_2site_sweep(&mut envs, &mut mps_b, &mpo_b, &standard_params_f64(0x14))
        .expect_err("length mismatch should fail");
    matches!(
        err,
        DmrgSweepError::LengthMismatch {
            mps: 3,
            mpo: 3,
            envs: 4,
        }
    )
    .then_some(())
    .expect("got wrong variant");
}

#[test]
fn t6_too_few_sites() {
    let d = 2;
    let backend = NativeBackend::shared();
    let mps = Mps::from_storages(vec![backend.make_tensor(vec![1.0_f64, 0.0], vec![1, d, 1])]);
    let mpo = Mpo::from_storages(vec![
        backend.make_tensor(vec![1.0_f64, 0.0, 0.0, 1.0], vec![1, d, d, 1]),
    ]);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let mut mps2 = mps.clone();
    let err = dmrg_2site_sweep(&mut envs, &mut mps2, &mpo, &standard_params_f64(0x20))
        .expect_err("n=1 should fail");
    assert!(matches!(err, DmrgSweepError::TooFewSites { n_sites: 1 }));
}

#[test]
fn t6_invalid_params_max_sweeps_zero() {
    let n = 4;
    let d = 2;
    let mps = random_mps_center_zero_f64(n, d, 2, 0x21);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0x22);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let mut mps2 = mps.clone();
    let mut p = standard_params_f64(0x23);
    p.max_sweeps = 0;
    let err = dmrg_2site_sweep(&mut envs, &mut mps2, &mpo, &p).expect_err("max_sweeps=0");
    assert!(matches!(err, DmrgSweepError::InvalidParams { .. }));
}

#[test]
fn t6_invalid_params_min_exceeds_max() {
    let n = 4;
    let d = 2;
    let mps = random_mps_center_zero_f64(n, d, 2, 0x24);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0x25);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let mut mps2 = mps.clone();
    let mut p = standard_params_f64(0x26);
    p.min_sweeps = 10;
    p.max_sweeps = 5;
    let err = dmrg_2site_sweep(&mut envs, &mut mps2, &mpo, &p).expect_err("min>max");
    assert!(matches!(err, DmrgSweepError::InvalidParams { .. }));
}

#[test]
fn t6_invalid_params_chi_max_zero() {
    let n = 4;
    let d = 2;
    let mps = random_mps_center_zero_f64(n, d, 2, 0x27);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0x28);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let mut mps2 = mps.clone();
    let mut p = standard_params_f64(0x29);
    p.trunc.chi_max = Some(0);
    let err = dmrg_2site_sweep(&mut envs, &mut mps2, &mpo, &p).expect_err("chi=0");
    assert!(matches!(err, DmrgSweepError::InvalidParams { .. }));
}

#[test]
fn t6_invalid_params_energy_tol_negative() {
    let n = 4;
    let d = 2;
    let mps = random_mps_center_zero_f64(n, d, 2, 0x2A);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0x2B);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let mut mps2 = mps.clone();
    let mut p = standard_params_f64(0x2C);
    p.energy_tol = -1e-10;
    let err = dmrg_2site_sweep(&mut envs, &mut mps2, &mpo, &p).expect_err("neg tol");
    assert!(matches!(err, DmrgSweepError::InvalidParams { .. }));
}

#[test]
fn t6_canonical_form_left_rejected() {
    let n = 4;
    let d = 2;
    let mut mps = random_mps_center_zero_f64(n, d, 2, 0x2D);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0x2E);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    // Move center to N-1 (i.e., left-canonical at sites 0..N-1) — not
    // allowed for the sweep entry point.
    canonicalize(&mut mps, n - 1);
    let err = dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &standard_params_f64(0x2F))
        .expect_err("left canonical reject");
    assert!(matches!(err, DmrgSweepError::MpsNotRightCanonical { .. }));
}

#[test]
fn t6_canonical_form_unknown_rejected() {
    let n = 4;
    let d = 2;
    let mps_init = random_mps_center_zero_f64(n, d, 2, 0x30);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0x31);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps_init, &mpo).expect("build");
    // Construct a fresh MPS with `from_storages` (which sets
    // `Unknown`), then pass it without canonicalizing.
    let mut mps_unk = Mps::from_storages(mps_init.storages().to_vec());
    assert_eq!(*mps_unk.canonical_form(), CanonicalForm::Unknown);
    let err = dmrg_2site_sweep(&mut envs, &mut mps_unk, &mpo, &standard_params_f64(0x32))
        .expect_err("Unknown rejected");
    assert!(matches!(err, DmrgSweepError::MpsNotRightCanonical { .. }));
}

#[test]
fn t6_step_error_propagated() {
    // Construct an MPS / MPO with a physical-dim mismatch so the
    // dmrg_2site_step shape check fires inside the loop.
    let n = 3;
    let d_mps = 2;
    let d_mpo = 3; // <- mismatch
    let backend = NativeBackend::shared();
    // The bond dimension is 1 at every site for this minimal fixture
    // (single-product state); explicitly write `1` rather than the
    // edge / interior conditional used elsewhere because both branches
    // collapse to 1 here.
    let mps_storages: Vec<Dense<f64>> = (0..n)
        .map(|_| backend.make_tensor(vec![1.0_f64, 0.0], vec![1, d_mps, 1]))
        .collect();
    let mut mps = Mps::from_storages(mps_storages);
    canonicalize(&mut mps, 0);
    // Identity-ish MPO with d_mpo physical (mismatching mps's d=2).
    let mpo_storages: Vec<Dense<f64>> = (0..n)
        .map(|_| {
            let mut m = vec![0.0_f64; d_mpo * d_mpo];
            for k in 0..d_mpo {
                m[k * d_mpo + k] = 1.0;
            }
            backend.make_tensor(m, vec![1, d_mpo, d_mpo, 1])
        })
        .collect();
    let mpo = Mpo::from_storages(mpo_storages);
    // Build envs from a *separately* matching pair so envs.n_sites == 3.
    let env_mps = random_mps_center_zero_f64(n, d_mpo, 1, 0xB1);
    let mut env_mpo_storages = Vec::new();
    for _ in 0..n {
        let mut m = vec![0.0_f64; d_mpo * d_mpo];
        for k in 0..d_mpo {
            m[k * d_mpo + k] = 1.0;
        }
        env_mpo_storages.push(backend.make_tensor(m, vec![1, d_mpo, d_mpo, 1]));
    }
    let env_mpo = Mpo::from_storages(env_mpo_storages);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&env_mps, &env_mpo).expect("build");
    let err = dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &standard_params_f64(0xB2))
        .expect_err("step shape mismatch");
    assert!(matches!(
        err,
        DmrgSweepError::Step {
            sweep: 0,
            site: 0,
            ..
        }
    ));
}

// ---------------------------------------------------------------------------
// T7 — c64 dtype-genericity (PSD-product reference)
// ---------------------------------------------------------------------------
#[test]
fn t7_c64_psd_product_converges() {
    let n = 3;
    let d = 2;
    let mut mps = random_mps_center_zero_c64(n, d, 3, 0xC1);
    let (mpo, hs) = psd_local_mpo_c64(n, d, 0xC2);
    let mut envs: DmrgEnvs<Dense<Complex<f64>>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let params = DmrgSweepParams {
        max_sweeps: 20,
        min_sweeps: 1,
        energy_tol: 1e-10,
        lanczos: LanczosParams {
            max_iter: 200,
            tol: 1e-10,
            seed: Some(0xC3),
        },
        trunc: TruncSvdParams {
            chi_max: Some(1),
            target_trunc_err: None,
        },
    };

    let result: DmrgResult<f64> =
        dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");
    let reference: f64 = hs.iter().map(|h| min_eig_complex_herm(h, d)).product();
    assert_abs_diff_eq!(result.energy, reference, epsilon = 1e-7);
    assert!(result.converged);
}

// ---------------------------------------------------------------------------
// T8 — lanczos_converged propagation
// ---------------------------------------------------------------------------
#[test]
fn t8_lanczos_nonconvergence_blocks_dmrg_convergence() {
    let n = 4;
    let d = 2;
    let mut mps = random_mps_center_zero_f64(n, d, 2, 0xE1);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0xE2);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    // Force Lanczos to fail convergence with max_iter=1 and an
    // unreasonably tight tolerance.
    let params = DmrgSweepParams {
        max_sweeps: 5,
        min_sweeps: 1,
        energy_tol: 1.0, // very loose — energy delta will satisfy it.
        lanczos: LanczosParams {
            max_iter: 1,
            tol: 1e-15,
            seed: Some(0xE3),
        },
        trunc: TruncSvdParams {
            chi_max: Some(1),
            target_trunc_err: None,
        },
    };

    let result = dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");
    assert!(
        !result.converged,
        "DmrgResult.converged must be false when any step has lanczos_converged=false"
    );
    let any_step_failed = result
        .sweeps
        .iter()
        .flat_map(|s| s.steps.iter())
        .any(|s| !s.lanczos_converged);
    assert!(
        any_step_failed,
        "test fixture should produce at least one non-converged Lanczos step"
    );
}

// T9 — diagnostics-field consistency for the fields T1–T8 do not assert on.
#[test]
fn t9_diagnostics_fields_consistent() {
    let n = 4;
    let d = 2;
    let mut mps = random_mps_center_zero_f64(n, d, 3, 0xD0);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0xD1);
    let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
    let params = DmrgSweepParams {
        max_sweeps: 3,
        min_sweeps: 3,
        energy_tol: 0.0,
        lanczos: LanczosParams {
            max_iter: 100,
            tol: 1e-10,
            seed: Some(0xD2),
        },
        trunc: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
    };

    let result = dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");

    for sweep in &result.sweeps {
        // Per-step contract.
        for step in &sweep.steps {
            assert!(
                step.residual >= 0.0,
                "step.residual = {} must be non-negative",
                step.residual
            );
            assert!(
                step.lanczos_iters >= 1,
                "step.lanczos_iters = {} must be >= 1 (Lanczos always runs at least once)",
                step.lanczos_iters
            );
            assert!(
                step.bond_dim >= 1,
                "step.bond_dim = {} must be >= 1",
                step.bond_dim
            );
            if let Some(chi_cap) = params.trunc.chi_max {
                assert!(
                    step.bond_dim <= chi_cap,
                    "step.bond_dim = {} exceeded chi_max = {}",
                    step.bond_dim,
                    chi_cap
                );
            }
            assert!(step.trunc_err >= 0.0, "step.trunc_err must be non-negative");
        }

        // Per-sweep aggregations match per-step values.
        let expected_min: f64 = sweep
            .steps
            .iter()
            .map(|s| s.eigenvalue)
            .fold(f64::INFINITY, f64::min);
        let expected_max_te: f64 = sweep
            .steps
            .iter()
            .map(|s| s.trunc_err)
            .fold(f64::NEG_INFINITY, f64::max);
        let expected_all_ok = sweep.steps.iter().all(|s| s.lanczos_converged);

        assert_abs_diff_eq!(sweep.min_step_eigenvalue, expected_min, epsilon = 0.0);
        assert_abs_diff_eq!(sweep.max_trunc_err, expected_max_te, epsilon = 0.0);
        assert_eq!(sweep.all_lanczos_converged, expected_all_ok);
    }

    // sweep.max_bond on the *last* sweep matches the actual MPS bond
    // dimensions after the driver returns.
    let final_max_bond = mps.max_bond_dim();
    let last_sweep = result.sweeps.last().expect("at least one sweep");
    assert_eq!(
        last_sweep.max_bond, final_max_bond,
        "last sweep.max_bond {} must equal post-sweep MPS max_bond_dim {}",
        last_sweep.max_bond, final_max_bond
    );
}

// T10 — post-sweep DmrgEnvs staleness contract: every populated slot
// must match a fresh `DmrgEnvs::build` against the post-sweep MPS,
// pinning the n=2 boundary case codex flagged.
#[test]
fn t10_post_sweep_envs_have_no_stale_some_slots() {
    fn check(side: &str, j: usize, n: usize, a: Option<&Dense<f64>>, b: Option<&Dense<f64>>) {
        match (a, b) {
            (Some(a), Some(b)) => {
                assert_eq!(a.shape(), b.shape(), "{side}[{j}] shape (n={n})");
                for (x, y) in a.data().iter().zip(b.data().iter()) {
                    assert_abs_diff_eq!(*x, *y, epsilon = 1e-10);
                }
            }
            (Some(_), None) => {
                panic!("post-sweep envs.{side}({j}) Some but fresh build None — stale-Some (n={n})")
            }
            _ => {}
        }
    }
    for &n in &[2usize, 4] {
        let d = 2;
        let mut mps = random_mps_center_zero_f64(n, d, 2, 0xF0 ^ n as u64);
        let (mpo, _) = psd_local_mpo_f64(n, d, 0xF1 ^ n as u64);
        let mut envs: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("build");
        let params = DmrgSweepParams {
            max_sweeps: 1,
            min_sweeps: 1,
            energy_tol: 0.0,
            lanczos: LanczosParams {
                max_iter: 100,
                tol: 1e-10,
                seed: Some(0xF2),
            },
            trunc: TruncSvdParams {
                chi_max: Some(1),
                target_trunc_err: None,
            },
        };
        dmrg_2site_sweep(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");
        let fresh: DmrgEnvs<Dense<f64>> = DmrgEnvs::build(&mps, &mpo).expect("rebuild");
        for j in 0..=n {
            check("left", j, n, envs.left(j), fresh.left(j));
            check("right", j, n, envs.right(j), fresh.right(j));
        }
    }
}
