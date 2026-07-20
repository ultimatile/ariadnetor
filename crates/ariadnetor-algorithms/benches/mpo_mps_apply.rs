//! End-to-end MPO-MPS apply bench for `ApplyMethod::SuccessiveRandomized`.
//!
//! Build / run:
//!   cargo bench -p ariadnetor-algorithms --bench mpo_mps_apply
//!
//! Two arms per case over identical Heisenberg-MPO / random-MPS inputs:
//! - `adaptive`: cutoff-driven sketch growth. A random MPS times a
//!   Heisenberg MPO is essentially incompressible, so the sketch grows from
//!   `sketch_dim` to the per-site cap in `sketch_increment` steps — the
//!   regime whose per-round cost the incremental QR is meant to cut. Setup
//!   asserts the growth actually happens, so the timed loop provably
//!   exercises the multi-append path.
//! - `fixed`: one QB pass at a fixed output rank; a single QR per site, no
//!   estimator. Expected to be insensitive to the adaptive-loop internals.
//! - `sum`: the linear-combination entry over two Heisenberg terms with
//!   distinct couplings and states, adaptive mode. Exercises the per-term
//!   environment / cap plumbing end to end; no performance claim rides on
//!   it (the multi-term path is a feature, not an optimization).
//!
//! The bench lives here rather than in `ariadnetor-mps` because the shared
//! input builders (`algorithms_fixtures`) depend on that crate — hosting it
//! there would need a dev-dependency cycle.

use algorithms_fixtures::dense_fixtures::{heisenberg_mpo_f64, random_mps_unknown_f64};
use ariadnetor_mps::{
    ApplyMethod, Mpo, Mps, SuccessiveRandomizedParams, TensorChain,
    apply_sum_successive_randomized, apply_with_method,
};
use ariadnetor_tensor::{DenseLayout, DenseStorage, Host, OpsFor};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// `apply_with_method` unwrapped: the bench inputs are finite, so an `Err`
/// can only mean the apply contract itself broke.
fn apply_ok<B: OpsFor<DenseStorage<f64>>>(
    backend: &B,
    mpo: &Mpo<DenseStorage<f64>, DenseLayout>,
    psi: &Mps<DenseStorage<f64>, DenseLayout>,
    method: ApplyMethod,
) -> Mps<DenseStorage<f64>, DenseLayout> {
    apply_with_method(backend, mpo, psi, None, method).expect("apply must succeed on finite inputs")
}

// ---------------------------------------------------------------------------
// Bench fixture grid: (n_sites, chi). The Heisenberg MPO has bond dimension
// 5, so the adaptive arm's per-site cap — and with an incompressible input,
// its final sketch size — scales as 5 * chi. Sizes kept small enough for a
// criterion sample to finish in seconds; the point is the relative cost of
// the adaptive-loop internals, not absolute wall time.
// ---------------------------------------------------------------------------

struct Case {
    label: &'static str,
    n_sites: usize,
    chi: usize,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            label: "n8_chi8",
            n_sites: 8,
            chi: 8,
        },
        Case {
            label: "n12_chi16",
            n_sites: 12,
            chi: 16,
        },
    ]
}

const RNG_SEED: u64 = 0xD3_BE_BC_AB_CD_EF_00_02;
const SKETCH_DIM: usize = 4;

fn adaptive_method() -> ApplyMethod {
    ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: Some(1e-8),
        sketch_dim: SKETCH_DIM,
        sketch_increment: 4,
        seed: RNG_SEED,
        ..Default::default()
    })
}

fn fixed_method(chi: usize) -> ApplyMethod {
    ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        output_dim: Some(chi),
        seed: RNG_SEED,
        ..Default::default()
    })
}

fn bench_mpo_mps_apply(c: &mut Criterion) {
    let backend = Host::shared();
    let backend = backend.as_ref();
    let mut group = c.benchmark_group("mpo_mps_apply");
    // The adaptive arm sweeps every site to its cap, so one iteration is
    // orders slower than a microbench; a reduced sample keeps the run short
    // while still resolving the arm-level differences this bench is for.
    group.sample_size(30);

    for case in &cases() {
        let mpo = heisenberg_mpo_f64(case.n_sites, 1.0);
        let psi = random_mps_unknown_f64(case.n_sites, 2, case.chi, RNG_SEED);

        // Untimed probe: the timed loop only measures the multi-append
        // growth path if the adaptive run actually grows past the initial
        // sketch. An incompressible input guarantees it; assert rather than
        // assume.
        let probe = apply_ok(backend, &mpo, &psi, adaptive_method());
        assert!(
            probe.max_bond_dim() > SKETCH_DIM,
            "adaptive arm did not grow past the initial sketch; the bench \
             would not exercise the incremental path"
        );

        group.bench_with_input(
            BenchmarkId::new("adaptive", case.label),
            &(&mpo, &psi),
            |b, (mpo, psi)| {
                b.iter(|| apply_ok(backend, mpo, psi, adaptive_method()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fixed", case.label),
            &(&mpo, &psi),
            |b, (mpo, psi)| {
                b.iter(|| apply_ok(backend, mpo, psi, fixed_method(case.chi)));
            },
        );

        let mpo2 = heisenberg_mpo_f64(case.n_sites, 0.5);
        let psi2 = random_mps_unknown_f64(case.n_sites, 2, case.chi, RNG_SEED ^ 1);
        let adaptive_src = SuccessiveRandomizedParams {
            cutoff: Some(1e-8),
            sketch_dim: SKETCH_DIM,
            sketch_increment: 4,
            seed: RNG_SEED,
            ..Default::default()
        };
        group.bench_with_input(
            BenchmarkId::new("sum", case.label),
            &[(&mpo, &psi), (&mpo2, &psi2)],
            |b, terms| {
                b.iter(|| {
                    apply_sum_successive_randomized(
                        backend,
                        terms,
                        &[1.0, -0.5],
                        None,
                        adaptive_src,
                    )
                    .expect("apply must succeed on finite inputs")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_mpo_mps_apply);
criterion_main!(benches);
