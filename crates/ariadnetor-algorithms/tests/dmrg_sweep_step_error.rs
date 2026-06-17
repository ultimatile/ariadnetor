//! Step-error propagation case for `sweep_2site`, split out from
//! `dmrg_sweep.rs` to keep the per-test-file line cap.

use arnet_algorithms::dmrg::{
    DmrgEnvs, DmrgSweepError, DmrgSweepParams, LocalEigensolverParams, sweep_2site,
};
use arnet_algorithms::krylov::LanczosParams;
use arnet_linalg::TruncSvdParams;
use arnet_mps::{Mpo, Mps, canonicalize};
use arnet_native::NativeBackend;
use arnet_tensor::{ComputeBackendTensorExt, DenseLayout, DenseStorage, DenseTensor, Host};
use test_utils::dense_fixtures::random_mps_center_zero_f64;

fn standard_params_f64(seed: u64) -> DmrgSweepParams {
    DmrgSweepParams {
        max_sweeps: 5,
        min_sweeps: 1,
        energy_tol: 1e-10,
        eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 100,
            tol: 1e-12,
            seed: Some(seed),
        }),
        trunc: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
    }
}

#[test]
fn t6_step_error_propagated() {
    // Construct an MPS / MPO with a physical-dim mismatch so the
    // dmrg_2site_step shape check fires inside the loop.
    let n = 3;
    let d_mps = 2;
    let d_mpo = 3; // <- mismatch
    let mps_storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| Host::shared().dense(vec![1.0_f64, 0.0], vec![1, d_mps, 1]))
        .collect();
    let mut mps = Mps::from_sites(mps_storages);
    canonicalize(&NativeBackend::new(), &mut mps, 0);
    let mpo_storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            let mut m = vec![0.0_f64; d_mpo * d_mpo];
            for k in 0..d_mpo {
                m[k * d_mpo + k] = 1.0;
            }
            Host::shared().dense(m, vec![1, d_mpo, d_mpo, 1])
        })
        .collect();
    let mpo = Mpo::from_sites(mpo_storages);
    let env_mps = random_mps_center_zero_f64(n, d_mpo, 1, 0xB1);
    let mut env_mpo_storages = Vec::new();
    for _ in 0..n {
        let mut m = vec![0.0_f64; d_mpo * d_mpo];
        for k in 0..d_mpo {
            m[k * d_mpo + k] = 1.0;
        }
        env_mpo_storages.push(Host::shared().dense(m, vec![1, d_mpo, d_mpo, 1]));
    }
    let env_mpo = Mpo::from_sites(env_mpo_storages);
    let mut envs: DmrgEnvs<DenseStorage<f64>, DenseLayout> =
        DmrgEnvs::build(&env_mps, &env_mpo).expect("build");
    let err = sweep_2site(&mut envs, &mut mps, &mpo, &standard_params_f64(0xB2))
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
