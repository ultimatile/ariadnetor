//! End-to-end 2-site DMRG bench comparing the in-tree Lanczos solver
//! against the ARPACK-NG-backed solver on identical
//! Heisenberg-MPO / random-MPS inputs.
//!
//! Build / run:
//!   cargo bench -p ariadnetor-algorithms --bench dmrg_local_eigensolver
//!     → Lanczos arm only.
//!   cargo bench -p ariadnetor-algorithms --bench dmrg_local_eigensolver --features arpack
//!     → Lanczos + ARPACK arms.
//!
//! The MPO builder mirrors `tests/dmrg_wrapper.rs` (1D OBC Heisenberg
//! at `J = 1`) so the solver-comparison fixture matches what existing
//! correctness tests pin down.

use std::sync::Arc;

use arnet::TruncSvdParams;
use arnet::{ComputeBackend, DenseLayout, DenseStorage, DenseTensor, NativeBackend};
use arnet_algorithms::dmrg::{DmrgSweepParams, LocalEigensolverParams, dmrg_2site};
#[cfg(feature = "arpack")]
use arnet_algorithms::krylov::ArpackParams;
use arnet_algorithms::krylov::LanczosParams;
use arnet_mps::{Mpo, Mps};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

const D: usize = 2; // physical dim (spin-1/2)

// ---------------------------------------------------------------------------
// Heisenberg MPO + random MPS — inlined per the existing test convention
// (each test/bench file builds its own fixtures rather than sharing a
// helper module).
// ---------------------------------------------------------------------------

type Op = fn(usize, usize) -> f64;

fn op_id(k: usize, b: usize) -> f64 {
    if k == b { 1.0 } else { 0.0 }
}

fn op_sz(k: usize, b: usize) -> f64 {
    if k == b {
        if k == 0 { 1.0 } else { -1.0 }
    } else {
        0.0
    }
}

fn op_sp(k: usize, b: usize) -> f64 {
    if k == 1 && b == 0 { 1.0 } else { 0.0 }
}

fn op_sm(k: usize, b: usize) -> f64 {
    if k == 0 && b == 1 { 1.0 } else { 0.0 }
}

fn build_mpo_site_f64(
    w_l_dim: usize,
    w_r_dim: usize,
    cells: &[(usize, usize, Op, f64)],
) -> DenseTensor<f64> {
    let backend = NativeBackend::shared();
    let len = w_l_dim * D * D * w_r_dim;
    let mut data = vec![0.0_f64; len];
    for &(vl, vr, op, scale) in cells {
        for k in 0..D {
            for b in 0..D {
                let idx = vl + w_l_dim * (k + D * (b + D * vr));
                data[idx] += scale * op(k, b);
            }
        }
    }
    DenseTensor::from_raw_parts(
        data,
        vec![w_l_dim, D, D, w_r_dim],
        backend.preferred_order(),
        Arc::clone(&backend),
    )
}

fn heisenberg_mpo_f64(n: usize, j: f64) -> Mpo<DenseStorage<f64>, DenseLayout> {
    assert!(n >= 2, "heisenberg_mpo_f64 requires n >= 2");
    let mut sites = Vec::with_capacity(n);

    sites.push(build_mpo_site_f64(
        1,
        5,
        &[
            (0, 1, op_sm, 2.0 * j),
            (0, 2, op_sp, 2.0 * j),
            (0, 3, op_sz, j),
            (0, 4, op_id, 1.0),
        ],
    ));

    for _ in 1..n - 1 {
        sites.push(build_mpo_site_f64(
            5,
            5,
            &[
                (0, 0, op_id, 1.0),
                (1, 0, op_sp, 1.0),
                (2, 0, op_sm, 1.0),
                (3, 0, op_sz, 1.0),
                (4, 1, op_sm, 2.0 * j),
                (4, 2, op_sp, 2.0 * j),
                (4, 3, op_sz, j),
                (4, 4, op_id, 1.0),
            ],
        ));
    }

    sites.push(build_mpo_site_f64(
        5,
        1,
        &[
            (0, 0, op_id, 1.0),
            (1, 0, op_sp, 1.0),
            (2, 0, op_sm, 1.0),
            (3, 0, op_sz, 1.0),
        ],
    ));

    Mpo::from_sites(sites)
}

fn random_mps_unknown_f64(n: usize, chi: usize, seed: u64) -> Mps<DenseStorage<f64>, DenseLayout> {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * D * r;
            let data: Vec<f64> = (0..len).map(|_| rng.random_range(-0.5_f64..0.5)).collect();
            DenseTensor::from_raw_parts(
                data,
                vec![l, D, r],
                backend.preferred_order(),
                Arc::clone(&backend),
            )
        })
        .collect();
    Mps::from_sites(storages)
}

// ---------------------------------------------------------------------------
// Bench fixture grid: (n_sites, chi_max, max_sweeps).
//
// Sizes kept small so a full criterion sample fits in a few seconds per
// arm. The point of the bench is the relative Lanczos-vs-ARPACK cost,
// not absolute wall time.
// ---------------------------------------------------------------------------

struct Case {
    label: &'static str,
    n_sites: usize,
    chi_max: usize,
    max_sweeps: usize,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            label: "n8_chi16_s2",
            n_sites: 8,
            chi_max: 16,
            max_sweeps: 2,
        },
        Case {
            label: "n12_chi24_s2",
            n_sites: 12,
            chi_max: 24,
            max_sweeps: 2,
        },
        Case {
            label: "n16_chi32_s1",
            n_sites: 16,
            chi_max: 32,
            max_sweeps: 1,
        },
    ]
}

fn lanczos_eigensolver(seed: u64) -> LocalEigensolverParams {
    LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 200,
        tol: 1e-10,
        seed: Some(seed),
    })
}

#[cfg(feature = "arpack")]
fn arpack_eigensolver() -> LocalEigensolverParams {
    LocalEigensolverParams::Arpack(ArpackParams {
        tol: 1e-10,
        max_iter: 300,
        ncv: None,
    })
}

fn make_params(case: &Case, eigensolver: LocalEigensolverParams) -> DmrgSweepParams {
    DmrgSweepParams {
        max_sweeps: case.max_sweeps,
        min_sweeps: 1,
        energy_tol: 1e-10,
        eigensolver,
        trunc: TruncSvdParams {
            chi_max: Some(case.chi_max),
            target_trunc_err: None,
        },
    }
}

const RNG_SEED: u64 = 0xD3_BE_BC_AB_CD_EF_00_01;

fn bench_dmrg_local_eigensolver(c: &mut Criterion) {
    let mut group = c.benchmark_group("dmrg_local_eigensolver");

    // `dmrg_2site` takes `&Mpo` / `&Mps` and clones `psi0` internally
    // before mutating it — so each iteration's true incremental cost
    // is one Mps clone + one full sweep. `b.iter` re-invokes the
    // closure, so the inputs can be borrowed across iterations
    // without per-iteration cloning at the bench layer.
    for case in &cases() {
        let mpo = heisenberg_mpo_f64(case.n_sites, 1.0);
        let psi0 = random_mps_unknown_f64(case.n_sites, case.chi_max, RNG_SEED);

        // Lanczos arm.
        {
            let params = make_params(case, lanczos_eigensolver(RNG_SEED));
            group.bench_with_input(
                BenchmarkId::new("lanczos", case.label),
                &(&mpo, &psi0, &params),
                |b, (mpo, psi0, params)| {
                    b.iter(|| dmrg_2site(mpo, psi0, params).expect("dmrg lanczos"));
                },
            );
        }

        // ARPACK arm — feature-gated. When `arpack` is off, only the
        // Lanczos arm above runs and the bench output skips ARPACK
        // entries entirely.
        #[cfg(feature = "arpack")]
        {
            let params = make_params(case, arpack_eigensolver());
            group.bench_with_input(
                BenchmarkId::new("arpack", case.label),
                &(&mpo, &psi0, &params),
                |b, (mpo, psi0, params)| {
                    b.iter(|| dmrg_2site(mpo, psi0, params).expect("dmrg arpack"));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_dmrg_local_eigensolver);
criterion_main!(benches);
