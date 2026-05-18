//! Diagnostic-surface tests for the DMRG 2-site sweep driver:
//! eigensolver-convergence propagation (T8) and env-equivalence
//! cross-check (T9).

use approx::assert_abs_diff_eq;
use arnet_algorithms::dmrg::{DmrgEnvs, DmrgSweepParams, LocalEigensolverParams, sweep_2site};
use arnet_algorithms::krylov::LanczosParams;
use arnet_linalg::TruncSvdParams;
use arnet_mps::{Mpo, Mps, canonicalize};
use arnet_native::NativeBackend;
use arnet_tensor::{ComputeBackendTensorExt, DenseLayout, DenseStorage, DenseTensorData};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

type F64Mps = Mps<DenseStorage<f64>, DenseLayout>;
type F64Mpo = Mpo<DenseStorage<f64>, DenseLayout>;
type F64Envs = DmrgEnvs<DenseStorage<f64>, DenseLayout>;

fn random_mps_center_zero_f64(n: usize, d: usize, chi: usize, seed: u64) -> F64Mps {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensorData<f64>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * d * r;
            let data: Vec<f64> = (0..len).map(|_| rng.random_range(-0.5_f64..0.5)).collect();
            backend.make_tensor_data(data, vec![l, d, r])
        })
        .collect();
    let mut mps = Mps::from_sites(storages);
    canonicalize(&mut mps, 0);
    mps
}

fn psd_local_mpo_f64(n: usize, d: usize, seed: u64) -> (F64Mpo, Vec<Vec<f64>>) {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let eps = 0.5_f64;
    let mut hs: Vec<Vec<f64>> = Vec::with_capacity(n);
    let storages: Vec<DenseTensorData<f64>> = (0..n)
        .map(|_| {
            let g: Vec<f64> = (0..d * d)
                .map(|_| rng.random_range(-1.0_f64..1.0))
                .collect();
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
            backend.make_tensor_data(h, vec![1, d, d, 1])
        })
        .collect();
    (Mpo::from_sites(storages), hs)
}

// ---------------------------------------------------------------------------
// T8 — eigensolver_converged propagation
// ---------------------------------------------------------------------------
#[test]
fn t8_lanczos_nonconvergence_blocks_dmrg_convergence() {
    let n = 4;
    let d = 2;
    let mut mps = random_mps_center_zero_f64(n, d, 2, 0xE1);
    let (mpo, _) = psd_local_mpo_f64(n, d, 0xE2);
    let mut envs: F64Envs = DmrgEnvs::build(&mps, &mpo).expect("build");
    // Force Lanczos to under-iterate so every step reports
    // `eigensolver_converged = false`. The sweep driver propagates that
    // to `DmrgResult.converged = false` even when the energy delta
    // satisfies `energy_tol`.
    let params = DmrgSweepParams {
        max_sweeps: 3,
        min_sweeps: 1,
        energy_tol: 1e-2,
        eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 1,
            tol: 1e-30,
            seed: Some(0xE3),
        }),
        trunc: TruncSvdParams {
            chi_max: Some(4),
            target_trunc_err: None,
        },
    };
    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");
    let any_step_failed = result.sweeps.iter().any(|s| !s.all_eigensolver_converged);
    assert!(
        any_step_failed,
        "test setup should force at least one step's local eigensolver to fail"
    );
    assert!(
        !result.converged,
        "DmrgResult.converged must reflect step-level eigensolver_converged"
    );
}

// ---------------------------------------------------------------------------
// T9 — envs post-sweep are functionally equivalent to a fresh rebuild
// ---------------------------------------------------------------------------
#[test]
fn t9_envs_invariant_post_sweep_matches_fresh_build() {
    fn check(
        side: &'static str,
        j: usize,
        n: usize,
        post: Option<&DenseTensorData<f64>>,
        fresh: Option<&DenseTensorData<f64>>,
    ) {
        match (post, fresh) {
            (Some(a), Some(b)) => {
                assert_eq!(
                    a.shape(),
                    b.shape(),
                    "post-sweep envs.{side}({j}) shape mismatch (n={n})"
                );
                assert_eq!(
                    a.data().len(),
                    b.data().len(),
                    "post-sweep envs.{side}({j}) len mismatch (n={n})"
                );
                for (k, (av, bv)) in a.data().iter().zip(b.data().iter()).enumerate() {
                    assert_abs_diff_eq!(*av, *bv, epsilon = 1e-9);
                    let _ = k;
                }
            }
            (None, None) => {}
            (Some(_), None) => {
                panic!("post-sweep envs.{side}({j}) Some but fresh build None — stale-Some (n={n})")
            }
            (None, Some(_)) => {
                panic!("post-sweep envs.{side}({j}) None but fresh rebuild Some — missing (n={n})")
            }
        }
    }
    for &n in &[2usize, 3, 4] {
        let d = 2;
        let mut mps = random_mps_center_zero_f64(n, d, 2, 0xF0 ^ n as u64);
        let (mpo, _) = psd_local_mpo_f64(n, d, 0xF1 ^ n as u64);
        let mut envs: F64Envs = DmrgEnvs::build(&mps, &mpo).expect("build");
        let params = DmrgSweepParams {
            max_sweeps: 1,
            min_sweeps: 1,
            energy_tol: 0.0,
            eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
                max_iter: 100,
                tol: 1e-10,
                seed: Some(0xF2),
            }),
            trunc: TruncSvdParams {
                chi_max: Some(1),
                target_trunc_err: None,
            },
        };
        sweep_2site(&mut envs, &mut mps, &mpo, &params).expect("sweep ok");
        let fresh: F64Envs = DmrgEnvs::build(&mps, &mpo).expect("rebuild");
        for j in 0..=n {
            check("left", j, n, envs.left(j), fresh.left(j));
            check("right", j, n, envs.right(j), fresh.right(j));
        }
    }
}
