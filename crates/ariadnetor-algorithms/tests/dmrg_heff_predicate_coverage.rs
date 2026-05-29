//! Tests targeting the validation prelude of `dmrg_2site_step`.
//!
//! Lives in a separate file from `dmrg_heff.rs` because that file is
//! near the project's per-test-file line cap. The fixtures here are
//! the minimum needed to drive the predicate paths — no oracles or
//! cross-checks.

use std::sync::Arc;

use arnet::{DenseLayout, DenseStorage, DenseTensor, NativeBackend, TruncSvdParams};
use arnet_algorithms::dmrg::{DmrgEnvs, DmrgHeffError, LocalEigensolverParams, dmrg_2site_step};
use arnet_algorithms::krylov::LanczosParams;
use arnet_mps::{Mpo, Mps};

fn product_state_mps(n: usize, d: usize) -> Mps<DenseStorage<f64>, DenseLayout> {
    let backend: Arc<NativeBackend> = NativeBackend::shared();
    let sites: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            let mut data = vec![0.0_f64; d];
            data[0] = 1.0;
            DenseTensor::from_raw_parts(data, vec![1, d, 1], Arc::clone(&backend))
        })
        .collect();
    Mps::from_sites(sites)
}

fn identity_mpo(n: usize, d: usize) -> Mpo<DenseStorage<f64>, DenseLayout> {
    let backend: Arc<NativeBackend> = NativeBackend::shared();
    let sites: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            let mut data = vec![0.0_f64; d * d];
            for k in 0..d {
                data[k + d * k] = 1.0;
            }
            DenseTensor::from_raw_parts(data, vec![1, d, d, 1], Arc::clone(&backend))
        })
        .collect();
    Mpo::from_sites(sites)
}

#[test]
fn heff_2site_step_asymmetric_length_and_zero_tol() {
    let n = 4;
    let d = 2;
    let mps_4 = product_state_mps(n, d);
    let mpo_4 = identity_mpo(n, d);
    let envs_4 =
        DmrgEnvs::<DenseStorage<f64>, DenseLayout, NativeBackend>::build::<f64>(&mps_4, &mpo_4)
            .expect("build envs n=4");

    let mps_3 = product_state_mps(3, d);
    let mpo_3 = identity_mpo(3, d);

    let lan_default = LocalEigensolverParams::Lanczos(LanczosParams::default());
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };

    // mpo matches envs (4), mps does not (3).
    let result = dmrg_2site_step(&envs_4, &mps_3, &mpo_4, 0, &lan_default, &trunc);
    assert!(
        matches!(
            result,
            Err(DmrgHeffError::LengthMismatch {
                mps: 3,
                mpo: 4,
                envs: 4,
            })
        ),
        "expected LengthMismatch {{ mps: 3, mpo: 4, envs: 4 }}, got {result:?}",
    );

    // mps matches envs (4), mpo does not (3).
    let result = dmrg_2site_step(&envs_4, &mps_4, &mpo_3, 0, &lan_default, &trunc);
    assert!(
        matches!(
            result,
            Err(DmrgHeffError::LengthMismatch {
                mps: 4,
                mpo: 3,
                envs: 4,
            })
        ),
        "expected LengthMismatch {{ mps: 4, mpo: 3, envs: 4 }}, got {result:?}",
    );

    // tol = 0.0 must be accepted by the `< 0.0` predicate.
    let zero_tol = LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 1,
        tol: 0.0,
        seed: Some(0),
    });
    let result = dmrg_2site_step(&envs_4, &mps_4, &mpo_4, 0, &zero_tol, &trunc);
    if let Err(DmrgHeffError::InvalidEigensolverParams { detail }) = &result {
        assert_ne!(
            *detail, "lanczos.tol must be non-negative",
            "tol = 0.0 must not surface the non-negative rejection (got {result:?})",
        );
    }
}
