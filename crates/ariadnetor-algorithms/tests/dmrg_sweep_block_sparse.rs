//! BlockSparse 2-site DMRG sweep driver integration tests.
//!
//! Mirrors `dmrg_sweep.rs` for the BlockSparse / U(1) path. Reuses
//! the XY hopping fixtures from the `algorithms-fixtures` crate's `fixtures`
//! module.
//! The XY chain `H = J (S+_a S-_{a+1} + S-_a S+_{a+1})` is
//! U(1)-symmetric — every sweep step preserves the chain's total
//! flux, so a fixture built in U(1)=1 stays in U(1)=1 throughout.
//!
//! T1 anchors absolute energy on the n=2 closed form (GS energy
//! `-J` in U(1)=1). T2–T4 use the n=3 fixture for monotone /
//! boundary / env-equivalence behavior. T5 / T7 exercise the
//! n=2 edge case and c64 dtype path. T6 covers error surface.
//! T8 verifies Lanczos non-convergence blocks the driver's
//! convergence flag.

use algorithms_fixtures::fixtures;

use ariadnetor_algorithms::dmrg::{
    DmrgEnvs, DmrgSweepError, DmrgSweepParams, LocalEigensolverParams, SweepDirection, sweep_2site,
};
use ariadnetor_algorithms::krylov::LanczosParams;
use ariadnetor_linalg::TruncSvdParams;
use ariadnetor_mps::{CanonicalForm, Mpo, Mps, TensorChain, braket};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::test_fixtures::legs;
use ariadnetor_tensor::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, Direction, Sector,
    U1Sector,
};
use num_complex::Complex;

use fixtures::{
    make_n2_mpo_c64, make_n2_mpo_f64, make_n2_mps_c64, make_n2_mps_f64, make_n3_mpo_f64,
    make_n3_mps_f64,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn standard_params(seed: u64) -> DmrgSweepParams {
    DmrgSweepParams {
        max_sweeps: 8,
        min_sweeps: 1,
        energy_tol: 1e-10,
        eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 200,
            tol: 1e-12,
            seed: Some(seed),
        }),
        trunc: TruncSvdParams {
            chi_max: Some(8),
            target_trunc_err: None,
        },
    }
}

/// Build envs from a freshly-canonicalized copy of `mps`. The
/// driver's input contract is `Right` or `Mixed { center: 0 }`;
/// `from_storages` produces `Unknown`, so position the orthogonality
/// center at 0 first, then build envs.
fn setup_f64(
    mps: &mut Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>,
    mpo: &Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>,
) -> DmrgEnvs<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    mps.canonicalize(&NativeBackend::new(), 0);
    DmrgEnvs::build(mps, mpo).expect("envs build")
}

fn setup_c64(
    mps: &mut Mps<BlockSparseStorage<Complex<f64>>, BlockSparseLayout<U1Sector>>,
    mpo: &Mpo<BlockSparseStorage<Complex<f64>>, BlockSparseLayout<U1Sector>>,
) -> DmrgEnvs<BlockSparseStorage<Complex<f64>>, BlockSparseLayout<U1Sector>> {
    mps.canonicalize(&NativeBackend::new(), 0);
    DmrgEnvs::build(mps, mpo).expect("envs build")
}

// ---------------------------------------------------------------------------
// T1 — absolute energy convergence (n=2 XY closed form)
// ---------------------------------------------------------------------------

#[test]
fn bsp_sweep_n2_xy_converges_to_minus_j() {
    // n=2 OBC XY at J=1 in U(1)=1 sector: H restricted to
    // {|01⟩, |10⟩} is J·[[0,1],[1,0]] with eigenvalues ±J. GS = -J.
    let j = 1.0_f64;
    let mut mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(j);
    let mut envs = setup_f64(&mut mps, &mpo);
    let params = DmrgSweepParams {
        max_sweeps: 6,
        min_sweeps: 2,
        ..standard_params(42)
    };

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep");

    assert!(result.converged, "expected convergence, got {result:?}");
    assert!(
        (result.energy - (-1.0)).abs() < 1e-8,
        "expected energy ≈ -1.0, got {}",
        result.energy
    );
    assert!(result.n_sweeps >= params.min_sweeps);
    assert_eq!(mps.canonical_form(), &CanonicalForm::Mixed { center: 0 });
}

// ---------------------------------------------------------------------------
// T2 — energy monotone non-increasing across cycles
// ---------------------------------------------------------------------------

#[test]
fn bsp_sweep_n3_energy_monotone_nonincreasing() {
    let mut mps = make_n3_mps_f64();
    let mpo = make_n3_mpo_f64(1.0);
    let mut envs = setup_f64(&mut mps, &mpo);
    // Force several cycles so monotonicity is observable across
    // multiple records, not just a single converged result.
    let params = DmrgSweepParams {
        max_sweeps: 5,
        min_sweeps: 5,
        energy_tol: 0.0,
        ..standard_params(7)
    };

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep");

    assert_eq!(result.sweeps.len(), params.max_sweeps);
    let tol = 1e-10;
    for w in result.sweeps.windows(2) {
        assert!(
            w[1].sweep_energy <= w[0].sweep_energy + tol,
            "sweep_energy increased: {:?} → {:?}",
            w[0].sweep_energy,
            w[1].sweep_energy
        );
    }
}

// ---------------------------------------------------------------------------
// T3 — boundary sites covered each sweep
// ---------------------------------------------------------------------------

#[test]
fn bsp_sweep_n3_covers_boundary_sites_each_cycle() {
    let mut mps = make_n3_mps_f64();
    let mpo = make_n3_mpo_f64(1.0);
    let mut envs = setup_f64(&mut mps, &mpo);
    let params = DmrgSweepParams {
        max_sweeps: 3,
        min_sweeps: 3,
        energy_tol: 0.0,
        ..standard_params(11)
    };

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep");

    let n_sites = mps.len();
    for sweep in &result.sweeps {
        // L→R sites 0..n-1, then R→L sites n-2..=0.
        let l2r: Vec<usize> = sweep
            .steps
            .iter()
            .filter(|s| s.direction == SweepDirection::LeftToRight)
            .map(|s| s.site)
            .collect();
        let r2l: Vec<usize> = sweep
            .steps
            .iter()
            .filter(|s| s.direction == SweepDirection::RightToLeft)
            .map(|s| s.site)
            .collect();
        assert_eq!(l2r, (0..n_sites - 1).collect::<Vec<_>>());
        assert_eq!(r2l, (0..n_sites - 1).rev().collect::<Vec<_>>());
    }
}

// ---------------------------------------------------------------------------
// T4 — post-sweep envs functionally equivalent to fresh build
// ---------------------------------------------------------------------------

#[test]
fn bsp_sweep_envs_equivalent_to_fresh_rebuild_after_sweep() {
    // Run a prep sweep that mutates `envs` in place, then run an
    // identical (seed-pinned) comparison sweep twice: once from the
    // maintained envs, once from a fresh DmrgEnvs::build of the
    // post-prep MPS snapshot. If the maintained envs drifted from
    // what a fresh build would produce, the two paths solve
    // different local Heff problems and diverge — caught by the
    // per-step eigenvalue and final-energy comparisons.
    let mut mps = make_n3_mps_f64();
    let mpo = make_n3_mpo_f64(1.0);
    let mut envs = setup_f64(&mut mps, &mpo);
    let prep_params = DmrgSweepParams {
        max_sweeps: 1,
        min_sweeps: 1,
        energy_tol: 0.0,
        ..standard_params(0x4242)
    };
    sweep_2site(&mut envs, &mut mps, &mpo, &prep_params).expect("prep");

    let mut mps_a = mps.clone();
    let mut envs_a = envs.clone();
    let mut mps_b = mps.clone();
    let mut envs_b = DmrgEnvs::build(&mps_b, &mpo).expect("rebuild");

    let cmp_params = DmrgSweepParams {
        max_sweeps: 1,
        min_sweeps: 1,
        energy_tol: 0.0,
        ..standard_params(0xDEAD)
    };

    let res_a = sweep_2site(&mut envs_a, &mut mps_a, &mpo, &cmp_params).expect("a");
    let res_b = sweep_2site(&mut envs_b, &mut mps_b, &mpo, &cmp_params).expect("b");

    assert!((res_a.energy - res_b.energy).abs() < 1e-9);
    assert_eq!(res_a.sweeps[0].steps.len(), res_b.sweeps[0].steps.len());
    for (sa, sb) in res_a.sweeps[0]
        .steps
        .iter()
        .zip(res_b.sweeps[0].steps.iter())
    {
        assert!((sa.eigenvalue - sb.eigenvalue).abs() < 1e-9);
        assert!((sa.trunc_err - sb.trunc_err).abs() < 1e-9);
    }
    // Final post-sweep energy and norm must agree across paths.
    let e_a = braket(&NativeBackend::new(), &mps_a, &mpo, &mps_a);
    let e_b = braket(&NativeBackend::new(), &mps_b, &mpo, &mps_b);
    assert!((e_a - e_b).abs() < 1e-9);
    let n_a = mps_a.norm(&NativeBackend::new());
    let n_b = mps_b.norm(&NativeBackend::new());
    assert!((n_a - n_b).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// T5 — n=2 edge case (sweep terminates cleanly)
// ---------------------------------------------------------------------------

#[test]
fn bsp_sweep_n2_edge_case_runs() {
    let mut mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(0.5);
    let mut envs = setup_f64(&mut mps, &mpo);
    let params = standard_params(3);

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep");

    assert!(result.n_sweeps >= 1);
    // Per cycle: one L→R step at site 0, one R→L step at site 0.
    for sweep in &result.sweeps {
        assert_eq!(sweep.steps.len(), 2);
    }
}

// ---------------------------------------------------------------------------
// T6 — error paths
// ---------------------------------------------------------------------------

#[test]
fn bsp_sweep_error_length_mismatch() {
    let mut mps = make_n2_mps_f64();
    let mpo3 = make_n3_mpo_f64(1.0);
    let mpo2 = make_n2_mpo_f64(1.0);
    // Build envs against the matching n=2 MPO so envs.n_sites = 2,
    // then pass mpo3 (length 3) to the sweep.
    let mut envs = setup_f64(&mut mps, &mpo2);
    let params = standard_params(0);

    let err = sweep_2site(&mut envs, &mut mps, &mpo3, &params).expect_err("err");
    assert!(matches!(err, DmrgSweepError::LengthMismatch { .. }));
}

#[test]
fn bsp_sweep_error_invalid_params_max_sweeps_zero() {
    let mut mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.0);
    let mut envs = setup_f64(&mut mps, &mpo);
    let params = DmrgSweepParams {
        max_sweeps: 0,
        ..standard_params(0)
    };
    let err = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect_err("err");
    assert!(matches!(err, DmrgSweepError::InvalidParams { .. }));
}

#[test]
fn bsp_sweep_error_invalid_params_min_exceeds_max() {
    let mut mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.0);
    let mut envs = setup_f64(&mut mps, &mpo);
    let params = DmrgSweepParams {
        max_sweeps: 2,
        min_sweeps: 5,
        ..standard_params(0)
    };
    let err = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect_err("err");
    assert!(matches!(err, DmrgSweepError::InvalidParams { .. }));
}

#[test]
fn bsp_sweep_error_invalid_params_chi_max_zero() {
    let mut mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.0);
    let mut envs = setup_f64(&mut mps, &mpo);
    let mut params = standard_params(0);
    params.trunc.chi_max = Some(0);
    let err = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect_err("err");
    assert!(matches!(err, DmrgSweepError::InvalidParams { .. }));
}

#[test]
fn bsp_sweep_error_invalid_params_energy_tol_nan() {
    let mut mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.0);
    let mut envs = setup_f64(&mut mps, &mpo);
    let params = DmrgSweepParams {
        energy_tol: f64::NAN,
        ..standard_params(0)
    };
    let err = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect_err("err");
    assert!(matches!(err, DmrgSweepError::InvalidParams { .. }));
}

#[test]
fn bsp_sweep_error_canonical_form_unknown_rejected() {
    let mut mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.0);
    let mut envs = DmrgEnvs::build(&mps, &mpo).expect("envs build");
    // Skip canonicalize → form stays Unknown.
    let params = standard_params(0);
    let err = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect_err("err");
    assert!(matches!(err, DmrgSweepError::MpsNotRightCanonical { .. }));
}

#[test]
fn bsp_sweep_error_canonical_form_left_rejected() {
    let mut mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.0);
    let mut envs = setup_f64(&mut mps, &mpo);
    // Force a non-accepted form post-setup.
    mps.set_canonical_form(CanonicalForm::Left);
    let params = standard_params(0);
    let err = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect_err("err");
    assert!(matches!(err, DmrgSweepError::MpsNotRightCanonical { .. }));
}

// ---------------------------------------------------------------------------
// T7 — c64 dtype path (n=2 XY closed form)
// ---------------------------------------------------------------------------

#[test]
fn bsp_sweep_n2_xy_c64_converges_to_minus_j() {
    let j = 1.0_f64;
    let mut mps = make_n2_mps_c64();
    let mpo = make_n2_mpo_c64(j);
    let mut envs = setup_c64(&mut mps, &mpo);
    let params = DmrgSweepParams {
        max_sweeps: 6,
        min_sweeps: 2,
        ..standard_params(17)
    };

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep");

    assert!(result.converged);
    assert!(
        (result.energy - (-1.0)).abs() < 1e-8,
        "expected energy ≈ -1.0, got {}",
        result.energy
    );
}

// ---------------------------------------------------------------------------
// T8 — Lanczos non-convergence blocks DmrgResult convergence
// ---------------------------------------------------------------------------

#[test]
fn bsp_sweep_lanczos_nonconvergence_blocks_dmrg_convergence() {
    let mut mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.0);
    let mut envs = setup_f64(&mut mps, &mpo);
    // tol=0 → Lanczos's true-residual test cannot fire, so each step
    // returns `converged = false`. The driver's `all_eigensolver_converged
    // && |delta_E| <= tol` check should then never set the result's
    // converged flag, even if the energy stabilizes numerically.
    let params = DmrgSweepParams {
        max_sweeps: 4,
        min_sweeps: 2,
        energy_tol: 1e-10,
        eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 1,
            tol: 0.0,
            seed: Some(123),
        }),
        trunc: TruncSvdParams {
            chi_max: Some(8),
            target_trunc_err: None,
        },
    };

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep");

    assert!(
        !result.converged,
        "expected non-convergence, got {result:?}"
    );
    for sweep in &result.sweeps {
        assert!(!sweep.all_eigensolver_converged);
    }
}

// ---------------------------------------------------------------------------
// T6 — TooFewSites and Step error propagation
// ---------------------------------------------------------------------------

type BspMpsF64 = Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>;
type BspMpoF64 = Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>;

/// Build a 1-site BlockSparse MPS / MPO pair with trivial dim-1
/// sectors so `DmrgEnvs::build` succeeds at n=1; the sweep must
/// then reject n_sites < 2 with `TooFewSites`.
fn make_n1_trivial_f64() -> (BspMpsF64, BspMpoF64) {
    let trivial = vec![(U1Sector(0), 1)];
    let phys = vec![(U1Sector(0), 1)];
    let mut mps_site = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (trivial.clone(), Direction::Out),
            (phys.clone(), Direction::Out),
            (trivial.clone(), Direction::In),
        ]),
        U1Sector::identity(),
    );
    mps_site
        .block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .expect("a")[0] = 1.0;
    let mut mpo_site = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (trivial.clone(), Direction::Out),
            (phys.clone(), Direction::In),
            (phys, Direction::Out),
            (trivial, Direction::In),
        ]),
        U1Sector::identity(),
    );
    mpo_site
        .block_data_mut(&BlockCoord(vec![0, 0, 0, 0]))
        .expect("b")[0] = 1.0;
    (
        Mps::from_sites(vec![mps_site]),
        Mpo::from_sites(vec![mpo_site]),
    )
}

#[test]
fn bsp_sweep_error_too_few_sites() {
    let (mut mps, mpo) = make_n1_trivial_f64();
    let mut envs = DmrgEnvs::build(&mps, &mpo).expect("build n=1");
    let params = standard_params(0);
    let err = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect_err("err");
    assert!(matches!(err, DmrgSweepError::TooFewSites { n_sites: 1 }));
}

#[test]
fn bsp_sweep_error_step_error_propagated() {
    // Build envs from a valid n=2 fixture so envs.n_sites = 2 and
    // sweep-level validation passes, then call sweep with an MPO
    // whose W[0] carries non-identity flux. The step's
    // identity-flux precondition fires inside the loop, surfacing
    // as DmrgSweepError::Step wrapping DmrgHeffError::QnMismatch.
    let mut mps = make_n2_mps_f64();
    let mpo_good = make_n2_mpo_f64(1.0);
    let mut envs = setup_f64(&mut mps, &mpo_good);

    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let trivial = vec![(U1Sector(0), 1)];
    let xy_bond = vec![(U1Sector(-1), 1), (U1Sector(1), 1)];
    let bad_w0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (trivial, Direction::Out),
            (phys.clone(), Direction::In),
            (phys, Direction::Out),
            (xy_bond, Direction::In),
        ]),
        U1Sector(2),
    );
    let bad_mpo = Mpo::from_sites(vec![bad_w0, mpo_good.site(1).clone()]);

    let params = standard_params(0);
    let err = sweep_2site(&mut envs, &mut mps, &bad_mpo, &params).expect_err("err");
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
// T9 — post-sweep envs slot-vs-fresh-rebuild equality (n=2 boundary
// case + n=3 bulk case). Catches stale-Some env slots that survive
// the sweep without being consumed — the failure mode the
// twin-sweep T4 cannot detect.
// ---------------------------------------------------------------------------

fn assert_bsp_close(
    a: &BlockSparseTensor<f64, U1Sector>,
    b: &BlockSparseTensor<f64, U1Sector>,
    eps: f64,
    ctx: &str,
) {
    assert_eq!(a.indices().len(), b.indices().len(), "{ctx}: rank");
    for ax in 0..a.indices().len() {
        let ia = &a.indices()[ax];
        let ib = &b.indices()[ax];
        assert_eq!(ia.direction(), ib.direction(), "{ctx}: axis {ax} dir");
        assert_eq!(
            ia.blocks().to_vec(),
            ib.blocks().to_vec(),
            "{ctx}: axis {ax} sectors"
        );
    }
    assert_eq!(a.flux(), b.flux(), "{ctx}: flux");
    for meta in a.block_metas() {
        let coord = &meta.coord;
        let da = a.block_data(coord).expect("a block");
        let db = b
            .block_data(coord)
            .unwrap_or_else(|| panic!("{ctx}: stale-Some block at {:?}", coord));
        assert_eq!(da.len(), db.len(), "{ctx}: block size at {:?}", coord);
        for (i, (x, y)) in da.iter().zip(db.iter()).enumerate() {
            assert!(
                (x - y).abs() < eps,
                "{ctx}: block {:?}[{i}] {x} vs {y}",
                coord
            );
        }
    }
    for meta in b.block_metas() {
        a.block_data(&meta.coord)
            .unwrap_or_else(|| panic!("{ctx}: extra block in b at {:?}", meta.coord));
    }
}

#[test]
fn bsp_sweep_post_sweep_envs_have_no_stale_some_slots() {
    for label in ["n2", "n3"] {
        let (mut mps, mpo) = if label == "n2" {
            (make_n2_mps_f64(), make_n2_mpo_f64(1.0))
        } else {
            (make_n3_mps_f64(), make_n3_mpo_f64(1.0))
        };
        let mut envs = setup_f64(&mut mps, &mpo);
        let params = DmrgSweepParams {
            max_sweeps: 1,
            min_sweeps: 1,
            energy_tol: 0.0,
            ..standard_params(0xF2)
        };
        sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep");
        let fresh = DmrgEnvs::build(&mps, &mpo).expect("rebuild");
        let n = mps.len();
        for j in 0..=n {
            if let Some(maintained) = envs.left(j) {
                let f = fresh
                    .left(j)
                    .unwrap_or_else(|| panic!("{label}: stale-Some left[{j}]"));
                assert_bsp_close(maintained, f, 1e-10, &format!("{label} left[{j}]"));
            }
            if let Some(maintained) = envs.right(j) {
                let f = fresh
                    .right(j)
                    .unwrap_or_else(|| panic!("{label}: stale-Some right[{j}]"));
                assert_bsp_close(maintained, f, 1e-10, &format!("{label} right[{j}]"));
            }
        }
    }
}
