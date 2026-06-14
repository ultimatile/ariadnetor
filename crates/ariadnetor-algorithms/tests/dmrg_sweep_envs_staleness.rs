//! Post-sweep DmrgEnvs staleness contract (t10), split out from
//! `dmrg_sweep.rs` to keep the per-test-file line cap. Pins the n=2
//! boundary case codex flagged: every populated slot must match a
//! fresh `DmrgEnvs::build` against the post-sweep MPS.

use approx::assert_abs_diff_eq;
use arnet_algorithms::dmrg::{DmrgEnvs, DmrgSweepParams, LocalEigensolverParams, sweep_2site};
use arnet_algorithms::krylov::LanczosParams;
use arnet_linalg::TruncSvdParams;
use arnet_mps::{Mpo, Mps, canonicalize};
use arnet_native::NativeBackend;
use arnet_tensor::{DenseLayout, DenseStorage, DenseTensor};

fn random_mps_center_zero_f64(
    n: usize,
    d: usize,
    chi: usize,
    seed: u64,
) -> Mps<DenseStorage<f64>, DenseLayout> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let sites: Vec<DenseTensor<f64>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * d * r;
            let data: Vec<f64> = (0..len).map(|_| rng.random_range(-0.5_f64..0.5)).collect();
            DenseTensor::from_raw_parts(data, vec![l, d, r])
        })
        .collect();
    let mut mps = Mps::from_sites(sites);
    canonicalize(&NativeBackend::new(), &mut mps, 0);
    mps
}

/// Identity-Hermitian PSD-product MPO. Each site is `h_i ⊗ I ⊗ I ...`,
/// where `h_i` is a random PSD matrix `R R^T` (`R` is the seeded random
/// `d × d`). Bond dim is 1; the chain's `<psi|H|psi>` factors as a
/// product per site.
fn psd_local_mpo_f64(n: usize, d: usize, seed: u64) -> Mpo<DenseStorage<f64>, DenseLayout> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let sites: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            let mut r = vec![0.0_f64; d * d];
            for entry in r.iter_mut() {
                *entry = rng.random_range(-1.0_f64..1.0);
            }
            // h = R R^T (column-major flat at [i + d*j]).
            let mut h = vec![0.0_f64; d * d];
            for i in 0..d {
                for j in 0..d {
                    let mut acc = 0.0_f64;
                    for k in 0..d {
                        acc += r[i + d * k] * r[j + d * k];
                    }
                    h[i + d * j] = acc;
                }
            }
            DenseTensor::from_raw_parts(h, vec![1, d, d, 1])
        })
        .collect();
    Mpo::from_sites(sites)
}

#[test]
fn t10_post_sweep_envs_have_no_stale_some_slots() {
    fn check(
        side: &str,
        j: usize,
        n: usize,
        a: Option<&DenseTensor<f64>>,
        b: Option<&DenseTensor<f64>>,
    ) {
        match (a, b) {
            (Some(a), Some(b)) => {
                assert_eq!(a.shape(), b.shape(), "{side}[{j}] shape (n={n})");
                for (x, y) in a.data_slice().iter().zip(b.data_slice().iter()) {
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
        let mpo = psd_local_mpo_f64(n, d, 0xF1 ^ n as u64);
        let mut envs: DmrgEnvs<DenseStorage<f64>, DenseLayout> =
            DmrgEnvs::build(&mps, &mpo).expect("build");
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
        let fresh: DmrgEnvs<DenseStorage<f64>, DenseLayout> =
            DmrgEnvs::build(&mps, &mpo).expect("rebuild");
        for j in 0..=n {
            check("left", j, n, envs.left(j), fresh.left(j));
            check("right", j, n, envs.right(j), fresh.right(j));
        }
    }
}
