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
//! The MPO builder is the shared `test_utils::dense_fixtures`
//! Heisenberg builder (1D OBC Heisenberg at `J = 1`), so the
//! solver-comparison fixture matches what the correctness tests pin down.

use arnet_algorithms::dmrg::{DmrgSweepParams, LocalEigensolverParams, dmrg_2site};
#[cfg(feature = "arpack")]
use arnet_algorithms::krylov::ArpackParams;
use arnet_algorithms::krylov::LanczosParams;
use arnet_linalg::TruncSvdParams;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use test_utils::dense_fixtures::{heisenberg_mpo_f64, random_mps_unknown_f64};

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
        let psi0 = random_mps_unknown_f64(case.n_sites, 2, case.chi_max, RNG_SEED);

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
