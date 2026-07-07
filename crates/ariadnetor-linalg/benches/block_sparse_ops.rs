//! Criterion benchmarks for BlockSparse operations.
//!
//! Sweeps over sector count (q) and per-sector degeneracy (d) for:
//! - contract_block_sparse (rank-2 matmul, rank-3 contraction)
//! - svd_block_sparse / trunc_svd_block_sparse
//! - qr_block_sparse / lq_block_sparse
//!
//! Decomposition and rank-2 contract groups include a Dense baseline of
//! equivalent total dimension. Rank-3 contraction is BlockSparse-only.
//!
//! The `block_transpose` group is a separate micro-benchmark isolating the
//! physical per-block transpose (naive kernel vs HPTT via `backend.transpose`).

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;

use ariadnetor_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder, TransposeDescriptor};
use ariadnetor_linalg::{DenseHostOps, TruncSvdParams, lq, qr, svd, tensordot, trunc_svd};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::test_fixtures::{legs, square_legs};
use ariadnetor_tensor::{BlockSparseTensor, DenseTensor, Direction, U1Sector};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parameter set for benchmark sweeps.
struct Params {
    label: String,
    q: usize,
    d: usize,
}

/// Standard (q, d) sweep used across most benchmark groups.
fn standard_sweep() -> Vec<Params> {
    [(2, 16), (2, 64), (4, 16), (4, 64), (8, 16), (8, 64)]
        .into_iter()
        .map(|(q, d)| Params {
            label: format!("q{q}_d{d}"),
            q,
            d,
        })
        .collect()
}

/// Singh et al. (2010) Fig.13 regime: q=5, varying d.
fn singh_sweep() -> Vec<Params> {
    [16, 32, 64, 128]
        .into_iter()
        .map(|d| Params {
            label: format!("singh_d{d}"),
            q: 5,
            d,
        })
        .collect()
}

/// Build a rank-2 `BlockSparseTensor` with q U(1) sectors of degeneracy d each.
fn random_bsp_matrix(q: usize, d: usize) -> BlockSparseTensor<f64, U1Sector> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let sectors: Vec<(U1Sector, usize)> = (0..q as i32).map(|i| (U1Sector(i), d)).collect();
    BlockSparseTensor::random(square_legs(sectors), U1Sector(0), &mut rng)
}

/// Build a rank-3 `BlockSparseTensor`: (bond_left, physical, bond_right).
fn random_bsp_rank3(q: usize, d: usize) -> BlockSparseTensor<f64, U1Sector> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let bond_sectors: Vec<(U1Sector, usize)> = (0..q as i32).map(|i| (U1Sector(i), d)).collect();
    let phys_sectors = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    BlockSparseTensor::random(
        legs([
            (bond_sectors.clone(), Direction::Out),
            (phys_sectors, Direction::Out),
            (bond_sectors, Direction::In),
        ]),
        U1Sector(0),
        &mut rng,
    )
}

fn random_dense_matrix(total_dim: usize) -> DenseTensor<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    DenseTensor::random(vec![total_dim, total_dim], &mut rng)
}

// ---------------------------------------------------------------------------
// Contract benchmarks
// ---------------------------------------------------------------------------

fn bench_contract(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("contract_bsp_rank2");

    for p in &standard_sweep() {
        let a = random_bsp_matrix(p.q, p.d);
        let b = random_bsp_matrix(p.q, p.d);
        group.bench_with_input(
            BenchmarkId::new("bsp", &p.label),
            &(&a, &b),
            |bench, &(a, b)| {
                bench.iter_with_large_drop(|| tensordot(&backend, a, b, &[1], &[0]).unwrap());
            },
        );

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        let b_dense = random_dense_matrix(total);
        group.bench_with_input(
            BenchmarkId::new("dense", &p.label),
            &(&a_dense, &b_dense),
            |bench, (a, b)| {
                bench.iter_with_large_drop(|| a.contract(b, "ij,jk->ik").unwrap());
            },
        );
    }

    group.finish();
}

/// Contraction with non-standard axis pairing that forces internal permutation.
fn bench_contract_permuted(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("contract_bsp_permuted");

    for p in &standard_sweep() {
        let a = random_bsp_matrix(p.q, p.d);
        let b = random_bsp_matrix(p.q, p.d);
        group.bench_with_input(
            BenchmarkId::new("bsp", &p.label),
            &(&a, &b),
            |bench, &(a, b)| {
                bench.iter_with_large_drop(|| tensordot(&backend, a, b, &[0], &[1]).unwrap());
            },
        );

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        let b_dense = random_dense_matrix(total);
        group.bench_with_input(
            BenchmarkId::new("dense", &p.label),
            &(&a_dense, &b_dense),
            |bench, (a, b)| {
                bench.iter_with_large_drop(|| a.contract(b, "ij,ki->jk").unwrap());
            },
        );
    }

    group.finish();
}

fn bench_contract_rank3(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("contract_bsp_rank3");

    for p in &standard_sweep() {
        let a = random_bsp_rank3(p.q, p.d);
        let b = random_bsp_rank3(p.q, p.d);
        // Contract over rightmost of a (axis 2) with leftmost of b (axis 0)
        group.bench_with_input(
            BenchmarkId::new("bsp", &p.label),
            &(&a, &b),
            |bench, &(a, b)| {
                bench.iter_with_large_drop(|| tensordot(&backend, a, b, &[2], &[0]).unwrap());
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// SVD benchmarks
// ---------------------------------------------------------------------------

fn bench_svd(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("svd_bsp");

    for p in &standard_sweep() {
        let a = random_bsp_matrix(p.q, p.d);
        group.bench_with_input(BenchmarkId::new("bsp", &p.label), &a, |bench, a| {
            bench.iter_with_large_drop(|| svd(&backend, a, 1).unwrap());
        });

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        group.bench_with_input(BenchmarkId::new("dense", &p.label), &a_dense, |bench, a| {
            bench.iter_with_large_drop(|| a.svd(1).unwrap());
        });
    }

    group.finish();
}

fn bench_trunc_svd(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("trunc_svd_bsp");

    for p in &standard_sweep() {
        let a = random_bsp_matrix(p.q, p.d);
        let chi_max = (p.q * p.d) / 2;
        let params = TruncSvdParams {
            chi_max: Some(chi_max),
            target_trunc_err: None,
        };

        group.bench_with_input(BenchmarkId::new("bsp", &p.label), &a, |bench, a| {
            bench.iter_with_large_drop(|| trunc_svd(&backend, a, 1, &params).unwrap());
        });

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        group.bench_with_input(BenchmarkId::new("dense", &p.label), &a_dense, |bench, a| {
            bench.iter_with_large_drop(|| a.trunc_svd(1, &params).unwrap());
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// QR / LQ benchmarks
// ---------------------------------------------------------------------------

fn bench_qr(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("qr_bsp");

    for p in &standard_sweep() {
        let a = random_bsp_matrix(p.q, p.d);
        group.bench_with_input(BenchmarkId::new("bsp", &p.label), &a, |bench, a| {
            bench.iter_with_large_drop(|| qr(&backend, a, 1).unwrap());
        });

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        group.bench_with_input(BenchmarkId::new("dense", &p.label), &a_dense, |bench, a| {
            bench.iter_with_large_drop(|| a.qr(1).unwrap());
        });
    }

    group.finish();
}

fn bench_lq(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("lq_bsp");

    for p in &standard_sweep() {
        let a = random_bsp_matrix(p.q, p.d);
        group.bench_with_input(BenchmarkId::new("bsp", &p.label), &a, |bench, a| {
            bench.iter_with_large_drop(|| lq(&backend, a, 1).unwrap());
        });

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        group.bench_with_input(BenchmarkId::new("dense", &p.label), &a_dense, |bench, a| {
            bench.iter_with_large_drop(|| a.lq(1).unwrap());
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Singh et al. (2010) Fig.13 reference regime
// ---------------------------------------------------------------------------

fn bench_singh_reference(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("singh_fig13");

    for p in &singh_sweep() {
        let a = random_bsp_matrix(p.q, p.d);
        let b = random_bsp_matrix(p.q, p.d);
        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        let b_dense = random_dense_matrix(total);

        // Matrix multiply
        group.bench_with_input(
            BenchmarkId::new("matmul_bsp", &p.label),
            &(&a, &b),
            |bench, &(a, b)| {
                bench.iter_with_large_drop(|| tensordot(&backend, a, b, &[1], &[0]).unwrap());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("matmul_dense", &p.label),
            &(&a_dense, &b_dense),
            |bench, (a, b)| {
                bench.iter_with_large_drop(|| a.contract(b, "ij,jk->ik").unwrap());
            },
        );

        // SVD
        group.bench_with_input(BenchmarkId::new("svd_bsp", &p.label), &a, |bench, a| {
            bench.iter_with_large_drop(|| svd(&backend, a, 1).unwrap());
        });
        group.bench_with_input(
            BenchmarkId::new("svd_dense", &p.label),
            &a_dense,
            |bench, a| {
                bench.iter_with_large_drop(|| a.svd(1).unwrap());
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Per-block transpose micro-benchmark
// ---------------------------------------------------------------------------

/// One representative (shape, perm) case for the physical per-block transpose.
/// `d` is the per-sector degeneracy that generated the shape; the parallel
/// sweep filters on it so the "largest d" selection tracks [`D_SWEEP`] rather
/// than a hardcoded label substring.
struct TransposeCase {
    label: String,
    d: usize,
    shape: Vec<usize>,
    perm: Vec<usize>,
}

/// Per-sector degeneracies swept for the per-block transpose micro-benchmark.
const D_SWEEP: [usize; 4] = [16, 32, 64, 128];

/// Representative per-block transpose cases.
///
/// Block-sparse contraction reads a block through the GEMM `trans_a`/`trans_b`
/// flag when the fold is an ascending prefix/suffix, and only falls back to a
/// physical transpose otherwise; block-sparse permute transposes every
/// non-identity perm physically. Each perm here is neither an ascending prefix
/// nor suffix, so it exercises that physical fallback rather than the flag path.
/// `d` sweeps the per-sector degeneracy so the naive-vs-HPTT crossover can be
/// read per block shape instead of being diluted by surrounding contraction
/// work.
fn transpose_cases() -> Vec<TransposeCase> {
    let mut cases = Vec::new();
    for &d in &D_SWEEP {
        // rank-3 (d, phys, d): adjacent swap, cyclic, full reversal.
        for (tag, perm) in [
            ("rank3_swap", vec![0, 2, 1]),
            ("rank3_cyc", vec![2, 0, 1]),
            ("rank3_rev", vec![2, 1, 0]),
        ] {
            cases.push(TransposeCase {
                label: format!("{tag}_d{d}"),
                d,
                shape: vec![d, 2, d],
                perm,
            });
        }
        // rank-4 (d, phys, phys, d): fold-style perm.
        cases.push(TransposeCase {
            label: format!("rank4_fold_d{d}"),
            d,
            shape: vec![d, 2, 2, d],
            perm: vec![0, 2, 3, 1],
        });
    }
    cases
}

/// Time `backend.transpose` on a single case with the given policy. The output
/// buffer is allocated once and reused across iterations so the measurement is
/// the kernel cost, not per-block allocation.
fn bench_transpose_case(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    backend: &NativeBackend,
    order: MemoryOrder,
    prefix: &str,
    case: &TransposeCase,
    policy: ExecPolicy,
) {
    let total: usize = case.shape.iter().product();
    let input: Vec<f64> = (0..total).map(|i| i as f64).collect();
    let mut output = vec![0.0f64; total];

    group.bench_function(BenchmarkId::new(prefix, &case.label), |bench| {
        bench.iter(|| {
            backend
                .transpose(TransposeDescriptor {
                    input: &input,
                    output: &mut output,
                    shape: &case.shape,
                    perm: &case.perm,
                    order,
                    conj: false,
                    policy,
                })
                .unwrap();
            // Force the write to survive dead-store elimination on the naive path.
            std::hint::black_box(&output);
        });
    });
}

/// Micro-benchmark for the physical per-block transpose.
///
/// This times [`ComputeBackend::transpose`] — the native naive kernel on the
/// default build, HPTT under `--features hptt` — so comparing the two baselines
/// answers "does HPTT beat the naive kernel per block". Block-sparse
/// contract/permute route their per-block transpose through
/// [`ComputeBackend::transpose`], so this measures that production path directly:
/// the native-vs-HPTT ratio is the per-block payoff of building with HPTT.
fn bench_block_transpose(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let order = backend.preferred_order();
    let mut group = c.benchmark_group("block_transpose");

    let cases = transpose_cases();
    for case in &cases {
        bench_transpose_case(
            &mut group,
            &backend,
            order,
            "seq",
            case,
            ExecPolicy::Sequential,
        );
    }

    // Parallel at the largest `d` only, to see whether threading shifts the
    // crossover without quadrupling wall time across the whole sweep.
    let max_d = D_SWEEP.iter().copied().max().unwrap_or(0);
    for case in cases.iter().filter(|tc| tc.d == max_d) {
        bench_transpose_case(
            &mut group,
            &backend,
            order,
            "par",
            case,
            ExecPolicy::Parallel(0),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_contract,
    bench_contract_permuted,
    bench_contract_rank3,
    bench_svd,
    bench_trunc_svd,
    bench_qr,
    bench_lq,
    bench_singh_reference,
    bench_block_transpose,
);
criterion_main!(benches);
