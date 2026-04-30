//! Error-path tests for `dmrg_2site_step_block_sparse` and a
//! complex-storage smoke test.

use arnet_algorithms::dmrg::{
    DmrgEnvs, DmrgHeffError, EffectiveHamiltonian2SiteBlockSparse, dmrg_2site_step_block_sparse,
};
use arnet_algorithms::krylov::{LanczosParams, LinearOp};
use arnet_linalg::TruncSvdParams;
use arnet_mps::{Mpo, TensorChain};
use arnet_native::NativeBackend;
use arnet_tensor::{BlockSparse, Dense, Direction, QNIndex, Sector, U1Sector};
use num_complex::Complex;

use super::fixtures::{
    build_envs_n2_f64, make_n2_mpo_c64, make_n2_mpo_f64, make_n2_mps_c64, make_n2_mps_f64,
};
use super::helpers::densify_bsp_c64;

#[test]
fn bsp_heff_step_error_paths_invalid_site() {
    let mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo);
    let params = LanczosParams::default();
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
fn bsp_heff_step_error_paths_invalid_lanczos_params() {
    let mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo);
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };

    let bad_iter = LanczosParams {
        max_iter: 0,
        tol: 1e-10,
        seed: None,
    };
    let r = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &bad_iter, &trunc);
    assert!(matches!(r, Err(DmrgHeffError::InvalidLanczosParams { .. })));

    let bad_nan = LanczosParams {
        max_iter: 200,
        tol: f64::NAN,
        seed: None,
    };
    let r = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &bad_nan, &trunc);
    assert!(matches!(r, Err(DmrgHeffError::InvalidLanczosParams { .. })));

    let bad_neg = LanczosParams {
        max_iter: 200,
        tol: -1.0,
        seed: None,
    };
    let r = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &bad_neg, &trunc);
    assert!(matches!(r, Err(DmrgHeffError::InvalidLanczosParams { .. })));
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
    let bad_w0 = BlockSparse::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(trivial, Direction::Out),
            QNIndex::new(phys.clone(), Direction::In),
            QNIndex::new(phys, Direction::Out),
            QNIndex::new(xy_bond, Direction::In),
        ],
        U1Sector(2),
    );
    let bad_mpo = Mpo::from_storages(vec![bad_w0, mpo_good.storage(1).clone()]);

    let params = LanczosParams::default();
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
fn bsp_heff_step_error_paths_qn_mismatch_mpo_bra_ket() {
    // Replace W[0] with one whose bra direction is `In` instead of
    // `Out` — violates the bra/ket duality + the bra-vs-MPS-phys
    // direction match (both checks fire on this single defect).
    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let trivial = vec![(U1Sector(0), 1)];
    let xy_bond = vec![(U1Sector(-1), 1), (U1Sector(1), 1)];
    let bad_w0 = BlockSparse::<f64, U1Sector>::zeros(
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
    let bad_mpo = Mpo::from_storages(vec![bad_w0, mpo_good.storage(1).clone()]);

    let params = LanczosParams::default();
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
        mpo.storage(0),
        mpo.storage(1),
        envs.right(2).expect("right"),
        mps.storage(0),
        mps.storage(1),
        backend,
    );

    let dim = bsp_heff.dim();
    let mut h_data = vec![Complex::new(0.0, 0.0); dim * dim];
    for j in 0..dim {
        let mut e_j = vec![Complex::new(0.0, 0.0); dim];
        e_j[j] = Complex::new(1.0, 0.0);
        let out = bsp_heff.apply(&Dense::new(e_j, vec![dim]));
        for i in 0..dim {
            h_data[i + dim * j] = out.data()[i];
        }
    }
    for i in 0..dim {
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

    let params = LanczosParams {
        max_iter: 200,
        tol: 1e-12,
        seed: Some(42),
    };
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let result = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &params, &trunc).expect("step");
    assert!(result.eigenvalue.is_finite());
    assert!(result.converged);
    let _ = densify_bsp_c64(&result.u);
}
