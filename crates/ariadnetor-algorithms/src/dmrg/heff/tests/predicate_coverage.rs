//! Tests targeting the validation prelude of the crate-internal
//! `dmrg_2site_step`.
//!
//! Kept in its own module from `step.rs` to stay under the project's
//! per-test-file line cap. The fixtures are the minimum needed to drive
//! the predicate paths — no oracles or cross-checks.

use super::{identity_mpo, product_state_mps};
use crate::dmrg::heff::dmrg_2site_step;
use crate::dmrg::{DmrgHeffError, LocalEigensolverParams};
use crate::krylov::LanczosParams;
use ariadnetor_linalg::TruncSvdParams;
use ariadnetor_mps::BraketEnvs;
use ariadnetor_tensor::{DenseLayout, DenseStorage};

#[test]
fn heff_2site_step_asymmetric_length_and_zero_tol() {
    let n = 4;
    let d = 2;
    let mps_4 = product_state_mps(n, d);
    let mpo_4 = identity_mpo(n, d);
    let envs_4 = BraketEnvs::<DenseStorage<f64>, DenseLayout>::build::<f64>(&mps_4, &mpo_4, &mps_4)
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
