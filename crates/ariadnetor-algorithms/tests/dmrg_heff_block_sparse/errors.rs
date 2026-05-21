//! Error-path tests for `dmrg_2site_step_block_sparse` and a
//! complex-storage smoke test.

use arnet::TruncSvdParams;
use arnet::{
    BlockCoord, BlockSparseTensor, DenseTensor, Direction, MemoryOrder, NativeBackend, QNIndex,
    Sector, U1Sector,
};
use arnet_algorithms::dmrg::{
    DmrgEnvs, DmrgHeffError, EffectiveHamiltonian2SiteBlockSparse, LocalEigensolverParams,
    dmrg_2site_step_block_sparse,
};
use arnet_algorithms::krylov::{LanczosParams, LinearOp};
use arnet_mps::{Mpo, TensorChain};
use num_complex::Complex;

use arnet_mps::Mps;

use super::fixtures::{
    build_envs_n2_f64, make_n2_mpo_c64, make_n2_mpo_f64, make_n2_mps_c64, make_n2_mps_f64,
    make_n3_mpo_f64, make_n3_mps_f64,
};
use super::helpers::densify_bsp_c64;

#[test]
fn bsp_heff_step_error_paths_invalid_site() {
    let mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo);
    let params = LocalEigensolverParams::Lanczos(LanczosParams::default());
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    // site = n-1 = 1 is invalid (need site + 1 < n, so site < 1).
    let r = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 1, &params, &trunc);
    assert!(matches!(r, Err(DmrgHeffError::InvalidSite { .. })));
    let r2 = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, usize::MAX, &params, &trunc);
    assert!(matches!(r2, Err(DmrgHeffError::InvalidSite { .. })));
}

#[test]
fn bsp_heff_step_error_paths_invalid_eigensolver_params() {
    let mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo);
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };

    let bad_iter = LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 0,
        tol: 1e-10,
        seed: None,
    });
    let r = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &bad_iter, &trunc);
    assert!(matches!(
        r,
        Err(DmrgHeffError::InvalidEigensolverParams { .. })
    ));

    let bad_nan = LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 200,
        tol: f64::NAN,
        seed: None,
    });
    let r = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &bad_nan, &trunc);
    assert!(matches!(
        r,
        Err(DmrgHeffError::InvalidEigensolverParams { .. })
    ));

    let bad_neg = LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 200,
        tol: -1.0,
        seed: None,
    });
    let r = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &bad_neg, &trunc);
    assert!(matches!(
        r,
        Err(DmrgHeffError::InvalidEigensolverParams { .. })
    ));
}

#[test]
fn bsp_heff_step_error_paths_qn_mismatch_mpo_flux() {
    let mps = make_n2_mps_f64();
    let mpo_good = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo_good);

    // Replace W[0] with a BlockSparse carrying non-identity flux.
    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let trivial = vec![(U1Sector(0), 1)];
    let xy_bond = vec![(U1Sector(-1), 1), (U1Sector(1), 1)];
    let bad_w0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(trivial, Direction::Out),
            QNIndex::new(phys.clone(), Direction::In),
            QNIndex::new(phys, Direction::Out),
            QNIndex::new(xy_bond, Direction::In),
        ],
        U1Sector(2),
    );
    let bad_mpo = Mpo::from_sites(vec![bad_w0, mpo_good.site(1).clone()]);

    let params = LocalEigensolverParams::Lanczos(LanczosParams::default());
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let r = dmrg_2site_step_block_sparse(&envs, &mps, &bad_mpo, 0, &params, &trunc);
    assert!(
        matches!(r, Err(DmrgHeffError::QnMismatch { field, .. }) if field.contains("flux")),
        "expected QnMismatch on flux, got {:?}",
        r.as_ref().err().map(|e| format!("{e}"))
    );
}

#[test]
fn bsp_heff_step_error_paths_empty_psi_template() {
    // Construct an MPS pair where each individual site has allowed
    // blocks AND env construction succeeds, but the derived
    // 2-site psi template flux has no flux-allowed tuple in the
    // outer-axis sector lattice. Without the dedicated guard,
    // `lanczos_smallest`'s `assert!(dim >= 1)` would panic.
    //
    // Construction:
    //   MPS[0]: left=(0,1) Out, phys=(0,1) Out, right=(0,1) In, flux=identity
    //   MPS[1]: left=(0,1) Out, phys=(0,1) Out, right=(2,1) In, flux=U1(2)
    //
    // MPS[0] has block (0,0,0). MPS[1] needs q_l + q_p - q_r = 2,
    // i.e. 0 + 0 - 2 = -2 ≠ 2 — so MPS[1] in fact has no allowed
    // blocks; envs.build would still succeed because boundary
    // helpers only require dim-1 / single-sector edge bonds, but
    // the derived psi_flux = identity + U1(2) = U1(2) cannot be
    // satisfied on a `(0,1)`-everywhere outer-axis lattice. Pre-
    // guard, this triggers the Lanczos panic. With the guard, it
    // surfaces as `QnMismatch { field: "psi_template", .. }`.
    let phys = vec![(U1Sector(0), 1)];
    let trivial = vec![(U1Sector(0), 1)];
    let charged_right = vec![(U1Sector(2), 1)];

    let mut mps0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(trivial.clone(), Direction::Out),
            QNIndex::new(phys.clone(), Direction::Out),
            QNIndex::new(trivial.clone(), Direction::In),
        ],
        U1Sector::identity(),
    );
    mps0.block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .expect("(0,0,0)")[0] = 1.0;

    let mps1 = BlockSparseTensor::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(trivial.clone(), Direction::Out),
            QNIndex::new(phys.clone(), Direction::Out),
            QNIndex::new(charged_right, Direction::In),
        ],
        U1Sector(2),
    );

    let mps = arnet_mps::Mps::from_sites(vec![mps0, mps1]);

    // Minimal MPO with dim-1 / single-sector edge bonds satisfying
    // the Phase 6.1 boundary contract. Identity propagator on the
    // single phys sector.
    let mut w0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(trivial.clone(), Direction::Out),
            QNIndex::new(phys.clone(), Direction::In),
            QNIndex::new(phys.clone(), Direction::Out),
            QNIndex::new(trivial.clone(), Direction::In),
        ],
        U1Sector::identity(),
    );
    w0.block_data_mut(&BlockCoord(vec![0, 0, 0, 0])).expect("I")[0] = 1.0;
    let mut w1 = BlockSparseTensor::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(trivial.clone(), Direction::Out),
            QNIndex::new(phys.clone(), Direction::In),
            QNIndex::new(phys, Direction::Out),
            QNIndex::new(trivial, Direction::In),
        ],
        U1Sector::identity(),
    );
    w1.block_data_mut(&BlockCoord(vec![0, 0, 0, 0])).expect("I")[0] = 1.0;
    let mpo = arnet_mps::Mpo::from_sites(vec![w0, w1]);

    let envs = DmrgEnvs::build(&mps, &mpo).expect("envs build");

    let params = LocalEigensolverParams::Lanczos(LanczosParams::default());
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let r = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &params, &trunc);
    assert!(
        matches!(r, Err(DmrgHeffError::QnMismatch { field, .. }) if field == "psi_template"),
        "expected QnMismatch on psi_template (empty flux-allowed set), got {:?}",
        r.as_ref().err().map(|e| format!("{e}"))
    );
}

#[test]
fn bsp_heff_step_error_paths_qn_mismatch_mpo_bra_ket() {
    // Replace W[0] with one whose bra direction is `In` instead of
    // `Out` — violates the bra/ket duality + the bra-vs-MPS-phys
    // direction match (both checks fire on this single defect).
    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let trivial = vec![(U1Sector(0), 1)];
    let xy_bond = vec![(U1Sector(-1), 1), (U1Sector(1), 1)];
    let bad_w0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(trivial, Direction::Out),
            QNIndex::new(phys.clone(), Direction::In),
            QNIndex::new(phys, Direction::In), // <-- WRONG: should be Out
            QNIndex::new(xy_bond, Direction::In),
        ],
        U1Sector::identity(),
    );
    let mps = make_n2_mps_f64();
    let mpo_good = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo_good);
    let bad_mpo = Mpo::from_sites(vec![bad_w0, mpo_good.site(1).clone()]);

    let params = LocalEigensolverParams::Lanczos(LanczosParams::default());
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let r = dmrg_2site_step_block_sparse(&envs, &mps, &bad_mpo, 0, &params, &trunc);
    assert!(
        matches!(r, Err(DmrgHeffError::QnMismatch { .. })),
        "expected QnMismatch, got {:?}",
        r.as_ref().err().map(|e| format!("{e}"))
    );
}

#[test]
fn bsp_heff_complex_path() {
    // Smoke + numerical agreement: build the complex BlockSparse
    // Heff and verify the matvec produces a Hermitian flat matrix
    // and a sensible eigenvalue via the public step API.
    let mps = make_n2_mps_c64();
    let mpo = make_n2_mpo_c64(1.5);
    let envs = DmrgEnvs::build(&mps, &mpo).expect("c64 envs");
    let backend = NativeBackend::shared();
    let bsp_heff = EffectiveHamiltonian2SiteBlockSparse::new(
        envs.left(0).expect("left"),
        mpo.site(0),
        mpo.site(1),
        envs.right(2).expect("right"),
        mps.site(0),
        mps.site(1),
        backend,
    );

    let dim = bsp_heff.dim();
    let mut h_data = vec![Complex::new(0.0, 0.0); dim * dim];
    for j in 0..dim {
        let mut e_j = vec![Complex::new(0.0, 0.0); dim];
        e_j[j] = Complex::new(1.0, 0.0);
        let out = bsp_heff.apply(&DenseTensor::from_raw_parts(
            e_j,
            vec![dim],
            MemoryOrder::ColumnMajor,
            NativeBackend::shared(),
        ));
        for i in 0..dim {
            h_data[i + dim * j] = out.data_slice()[i];
        }
    }
    for i in 0..dim {
        // Diagonal entries of a Hermitian matrix must be real.
        let diag_im = h_data[i + dim * i].im.abs();
        assert!(
            diag_im <= 1e-10,
            "complex matvec not Hermitian: H[{i},{i}].im = {diag_im} (expected ≈ 0)"
        );
        for j in (i + 1)..dim {
            let a = h_data[i + dim * j];
            let b = h_data[j + dim * i].conj();
            let diff = (a - b).norm();
            assert!(
                diff <= 1e-10,
                "complex matvec not Hermitian at ({i},{j}): a={a:?}, conj(H[{j},{i}])={b:?}, |diff|={diff}"
            );
        }
    }

    let params = LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 200,
        tol: 1e-12,
        seed: Some(42),
    });
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let result = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &params, &trunc).expect("step");
    assert!(result.eigenvalue.is_finite());
    assert!(result.converged);
    let _ = densify_bsp_c64(&result.u);
}

// Asymmetric length mismatch: exactly one of `mps.len()` and
// `mpo.len()` matches `envs.n_sites()`. The original `||` predicate
// returns `LengthMismatch` either way; mutating to `&&` suppresses
// the asymmetric branch and continues into the rank checks. Binding
// the explicit `mps`/`mpo`/`envs` values pins the variant.
#[test]
fn bsp_validate_inputs_asymmetric_length_mismatch() {
    let mps_n2 = make_n2_mps_f64();
    let mpo_n2 = make_n2_mpo_f64(1.5);
    let envs_n2 = build_envs_n2_f64(&mps_n2, &mpo_n2);
    let mps_n3 = make_n3_mps_f64();
    let mpo_n3 = make_n3_mpo_f64(1.5);

    let params = LocalEigensolverParams::Lanczos(LanczosParams::default());
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };

    let result = dmrg_2site_step_block_sparse(&envs_n2, &mps_n3, &mpo_n2, 0, &params, &trunc);
    assert!(
        matches!(
            result,
            Err(DmrgHeffError::LengthMismatch {
                mps: 3,
                mpo: 2,
                envs: 2,
            })
        ),
        "expected LengthMismatch {{ mps: 3, mpo: 2, envs: 2 }}, got {:?}",
        result.as_ref().err().map(|e| format!("{e}")),
    );

    let result = dmrg_2site_step_block_sparse(&envs_n2, &mps_n2, &mpo_n3, 0, &params, &trunc);
    assert!(
        matches!(
            result,
            Err(DmrgHeffError::LengthMismatch {
                mps: 2,
                mpo: 3,
                envs: 2,
            })
        ),
        "expected LengthMismatch {{ mps: 2, mpo: 3, envs: 2 }}, got {:?}",
        result.as_ref().err().map(|e| format!("{e}")),
    );
}

// StaleEnv reports `index: site + 2`. For site = 0 the original
// reports `index = 2`; the `+ → *` mutant reports `index = 0`. The
// binding on `index: 2` distinguishes them. Setting up a stale
// `right[2]` requires the n=3 fixture: `advance_left(0)` clears
// `right[1]`, `advance_left(1)` clears `right[2]` (interior since
// `1 + 1 < 3`).
#[test]
fn bsp_validate_inputs_stale_right_index_pinpoint() {
    let mps = make_n3_mps_f64();
    let mpo = make_n3_mpo_f64(1.5);
    let mut envs = DmrgEnvs::build(&mps, &mpo).expect("envs build n=3");

    envs.advance_left(&mps, &mpo, 0).expect("advance_left(0)");
    envs.advance_left(&mps, &mpo, 1).expect("advance_left(1)");

    let params = LocalEigensolverParams::Lanczos(LanczosParams::default());
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let result = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &params, &trunc);
    assert!(
        matches!(
            result,
            Err(DmrgHeffError::StaleEnv {
                side: "right",
                index: 2,
            })
        ),
        "expected StaleEnv {{ side: \"right\", index: 2 }}, got {:?}",
        result.as_ref().err().map(|e| format!("{e}")),
    );
}

// Isolate `check_qn_pair`'s sector-list comparison. Build envs from
// the standard n=2 fixture, then call the step with an alternate MPS
// whose `mps_alt[0].indices()[0]` declares a different sector list
// (with the same total dim and the same direction as the original).
// All earlier checks pass — rank, total-dim equality, the inline
// W-bra/MPS-phys direction equalities, and flux-identity. The first
// `check_qn_pair("left.bot_ket vs psi.axis 0 (MPS[i].left_bond)", _)`
// fails on sector-list mismatch. Under the stub `Ok(())` mutation,
// that call returns success, validation continues, and the surfaced
// error has a different `field` string. Exact-match `field` binding
// distinguishes original from mutation.
#[test]
fn bsp_validate_inputs_qn_mismatch_contracted_axis_sectors() {
    let mps_envs = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps_envs, &mpo);

    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let alt_left = vec![(U1Sector(2), 1)];
    let alt_mid = vec![(U1Sector(2), 1), (U1Sector(3), 1)];
    let trivial = vec![(U1Sector(0), 1)];

    let mps_alt0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(alt_left, Direction::Out),
            QNIndex::new(phys.clone(), Direction::Out),
            QNIndex::new(alt_mid.clone(), Direction::In),
        ],
        U1Sector::identity(),
    );
    let mps_alt1 = BlockSparseTensor::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(alt_mid, Direction::Out),
            QNIndex::new(phys, Direction::Out),
            QNIndex::new(trivial, Direction::In),
        ],
        U1Sector(2),
    );
    let mps_alt = Mps::from_sites(vec![mps_alt0, mps_alt1]);

    let params = LocalEigensolverParams::Lanczos(LanczosParams::default());
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let result = dmrg_2site_step_block_sparse(&envs, &mps_alt, &mpo, 0, &params, &trunc);
    assert!(
        matches!(
            &result,
            Err(DmrgHeffError::QnMismatch { field, .. })
                if *field == "left.bot_ket vs psi.axis 0 (MPS[i].left_bond)"
        ),
        "expected QnMismatch on \"left.bot_ket vs psi.axis 0 (MPS[i].left_bond)\", got {:?}",
        result.as_ref().err().map(|e| format!("{e}")),
    );
}
