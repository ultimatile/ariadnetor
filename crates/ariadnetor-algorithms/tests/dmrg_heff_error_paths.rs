//! Error-path coverage for `dmrg_2site_step`, split out from
//! `dmrg_heff.rs` to keep the per-test-file line cap. The fixtures
//! mirror the ones in `dmrg_heff.rs` (product-state Dense MPS, identity
//! MPO) so each integration-test binary is self-contained.

use arnet_algorithms::dmrg::{DmrgEnvs, DmrgHeffError, LocalEigensolverParams, dmrg_2site_step};
use arnet_algorithms::krylov::LanczosParams;
use arnet_linalg::TruncSvdParams;
use arnet_mps::{Mpo, Mps};
use arnet_tensor::{ComputeBackendTensorExt, DenseLayout, DenseStorage, DenseTensor, Host};

fn product_state_mps(n: usize, d: usize) -> Mps<DenseStorage<f64>, DenseLayout> {
    let sites: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            let mut data = vec![0.0_f64; d];
            data[0] = 1.0;
            Host::shared().dense(data, vec![1, d, 1])
        })
        .collect();
    Mps::from_sites(sites)
}

fn identity_mpo(n: usize, d: usize) -> Mpo<DenseStorage<f64>, DenseLayout> {
    let sites: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            let mut data = vec![0.0_f64; d * d];
            for k in 0..d {
                data[k + d * k] = 1.0;
            }
            Host::shared().dense(data, vec![1, d, d, 1])
        })
        .collect();
    Mpo::from_sites(sites)
}

#[test]
fn heff_error_paths() {
    let n = 4;
    let d = 2;
    let mps = product_state_mps(n, d);
    let mpo = identity_mpo(n, d);
    let mut envs = DmrgEnvs::build(&mps, &mpo).expect("build");
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
    let envs_4 = DmrgEnvs::build(&product_state_mps(n, d), &identity_mpo(n, d)).expect("build");
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
    let envs_d2 = DmrgEnvs::build(&mps_d2, &mpo_d2).expect("build envs(d=2)");
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
    let envs_fresh = DmrgEnvs::build(&mps, &mpo).expect("fresh build");
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
