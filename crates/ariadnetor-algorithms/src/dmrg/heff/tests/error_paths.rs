//! Error-path coverage for the crate-internal `dmrg_2site_step`. Kept
//! in its own module from `step.rs` to stay under the per-test-file
//! line cap.

use super::{identity_mpo, product_state_mps};
use crate::dmrg::heff::dmrg_2site_step;
use crate::dmrg::{DmrgHeffError, LocalEigensolverParams};
use crate::krylov::LanczosParams;
use ariadnetor_linalg::TruncSvdParams;
use ariadnetor_mps::BraketEnvs;

#[test]
fn heff_error_paths() {
    let n = 4;
    let d = 2;
    let mps = product_state_mps(n, d);
    let mpo = identity_mpo(n, d);
    let mut envs = BraketEnvs::build(&mps, &mpo).expect("build");
    let eigensolver = LocalEigensolverParams::Lanczos(LanczosParams::default());
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };

    // site + 1 == n → boundary, invalid for two-site.
    let bad = dmrg_2site_step(&envs, &mps, &mpo, n - 1, &eigensolver, &trunc);
    assert!(
        matches!(bad, Err(DmrgHeffError::InvalidSite { site, n_sites })
        if site == n - 1 && n_sites == n)
    );

    // Stale right env: advance_left twice so right[2] is invalidated.
    envs.advance_left(&mps, &mpo, 0).expect("advance_left(0)");
    envs.advance_left(&mps, &mpo, 1).expect("advance_left(1)");
    let stale = dmrg_2site_step(&envs, &mps, &mpo, 0, &eigensolver, &trunc);
    assert!(
        matches!(
            stale,
            Err(DmrgHeffError::StaleEnv {
                side: "right",
                index: 2
            })
        ),
        "got {:?}",
        stale
    );

    // LengthMismatch.
    let envs_4 = BraketEnvs::build(&product_state_mps(n, d), &identity_mpo(n, d)).expect("build");
    let mps_3 = product_state_mps(3, d);
    let mpo_3 = identity_mpo(3, d);
    let mismatch = dmrg_2site_step(&envs_4, &mps_3, &mpo_3, 0, &eigensolver, &trunc);
    assert!(
        matches!(
            mismatch,
            Err(DmrgHeffError::LengthMismatch {
                mps: 3,
                mpo: 3,
                envs: 4
            })
        ),
        "got {:?}",
        mismatch
    );

    // ShapeMismatch: feed in an MPO whose physical dim differs from
    // the MPS the envs were built against.
    let mps_d2 = product_state_mps(n, 2);
    let mpo_d2 = identity_mpo(n, 2);
    let envs_d2 = BraketEnvs::build(&mps_d2, &mpo_d2).expect("build envs(d=2)");
    let mpo_d3 = identity_mpo(n, 3);
    let bad_shape = dmrg_2site_step(&envs_d2, &mps_d2, &mpo_d3, 0, &eigensolver, &trunc);
    assert!(
        matches!(bad_shape, Err(DmrgHeffError::ShapeMismatch { site: 0, .. })),
        "got {:?}",
        bad_shape
    );

    // InvalidSite: site = usize::MAX must not overflow the +1 check.
    let overflow = dmrg_2site_step(&envs, &mps, &mpo, usize::MAX, &eigensolver, &trunc);
    assert!(
        matches!(
            overflow,
            Err(DmrgHeffError::InvalidSite { site, n_sites })
                if site == usize::MAX && n_sites == n
        ),
        "got {:?}",
        overflow
    );

    // InvalidEigensolverParams: max_iter = 0 / NaN / negative tol all
    // assert inside lanczos_smallest. The standard path must catch
    // them at entry instead of panicking.
    let bad_iter = dmrg_2site_step(
        &envs,
        &mps,
        &mpo,
        0,
        &LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 0,
            ..LanczosParams::default()
        }),
        &trunc,
    );
    assert!(
        matches!(
            bad_iter,
            Err(DmrgHeffError::InvalidEigensolverParams { .. })
        ),
        "got {:?}",
        bad_iter
    );
    let bad_nan = dmrg_2site_step(
        &envs,
        &mps,
        &mpo,
        0,
        &LocalEigensolverParams::Lanczos(LanczosParams {
            tol: f64::NAN,
            ..LanczosParams::default()
        }),
        &trunc,
    );
    assert!(
        matches!(bad_nan, Err(DmrgHeffError::InvalidEigensolverParams { .. })),
        "got {:?}",
        bad_nan
    );
    let bad_neg = dmrg_2site_step(
        &envs,
        &mps,
        &mpo,
        0,
        &LocalEigensolverParams::Lanczos(LanczosParams {
            tol: -1.0,
            ..LanczosParams::default()
        }),
        &trunc,
    );
    assert!(
        matches!(bad_neg, Err(DmrgHeffError::InvalidEigensolverParams { .. })),
        "got {:?}",
        bad_neg
    );

    // Contract: trunc_svd rejects `chi_max = Some(0)`.
    let envs_fresh = BraketEnvs::build(&mps, &mpo).expect("fresh build");
    let bad_trunc = TruncSvdParams {
        chi_max: Some(0),
        target_trunc_err: None,
    };
    let bad_contract = dmrg_2site_step(&envs_fresh, &mps, &mpo, 0, &eigensolver, &bad_trunc);
    assert!(
        matches!(bad_contract, Err(DmrgHeffError::Contract(_))),
        "got {:?}",
        bad_contract
    );
}
