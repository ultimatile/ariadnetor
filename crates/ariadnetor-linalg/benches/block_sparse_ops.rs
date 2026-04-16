//! Criterion benchmarks for BlockSparse operations.
//!
//! Sweeps over sector count (q) and per-sector degeneracy (d) for:
//! - contract_block_sparse (rank-2 matmul, rank-3 contraction)
//! - svd_block_sparse / trunc_svd_block_sparse
//! - qr_block_sparse / lq_block_sparse
//!
//! Each group includes a Dense baseline of equivalent total dimension.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;

use arnet_linalg::{
    TruncSvdParams, contract, contract_block_sparse, lq, lq_block_sparse, qr, qr_block_sparse, svd,
    svd_block_sparse, trunc_svd, trunc_svd_block_sparse,
};
use arnet_native::NativeBackend;
use arnet_tensor::{BlockSparse, Dense, Direction, QNIndex, U1Sector};

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

/// Build a rank-2 BlockSparse with q U(1) sectors of degeneracy d each.
///
/// Row index: sectors U1(0)..U1(q-1), each dim d, direction Out
/// Col index: same sectors, direction In
/// Flux: identity (U1(0)) → diagonal blocks allowed
/// Total logical shape: (q*d) × (q*d)
fn random_bsp_matrix(q: usize, d: usize) -> BlockSparse<f64, U1Sector> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let sectors: Vec<(U1Sector, usize)> = (0..q as i32).map(|i| (U1Sector(i), d)).collect();
    let row = QNIndex::new(sectors.clone(), Direction::Out);
    let col = QNIndex::new(sectors, Direction::In);
    BlockSparse::random(vec![row, col], U1Sector(0), &mut rng)
}

/// Build a rank-3 BlockSparse: (bond_left, physical, bond_right).
///
/// Bond indices: q sectors of degeneracy d each.
/// Physical index: 2 sectors of degeneracy 1 each (spin-1/2-like).
fn random_bsp_rank3(q: usize, d: usize) -> BlockSparse<f64, U1Sector> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let bond_sectors: Vec<(U1Sector, usize)> = (0..q as i32).map(|i| (U1Sector(i), d)).collect();
    let phys_sectors = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let left = QNIndex::new(bond_sectors.clone(), Direction::Out);
    let phys = QNIndex::new(phys_sectors, Direction::Out);
    let right = QNIndex::new(bond_sectors, Direction::In);
    BlockSparse::random(vec![left, phys, right], U1Sector(0), &mut rng)
}

fn random_dense_matrix(total_dim: usize) -> Dense<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    Dense::random(vec![total_dim, total_dim], &mut rng)
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
            |bench, (a, b)| {
                bench.iter_with_large_drop(|| {
                    contract_block_sparse(&backend, a, b, &[1], &[0]).unwrap()
                });
            },
        );

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        let b_dense = random_dense_matrix(total);
        group.bench_with_input(
            BenchmarkId::new("dense", &p.label),
            &(&a_dense, &b_dense),
            |bench, (a, b)| {
                bench.iter_with_large_drop(|| contract(&backend, a, b, "ij,jk->ik").unwrap());
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
            |bench, (a, b)| {
                bench.iter_with_large_drop(|| {
                    contract_block_sparse(&backend, a, b, &[2], &[0]).unwrap()
                });
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
            bench.iter_with_large_drop(|| svd_block_sparse(&backend, a, 1).unwrap());
        });

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        group.bench_with_input(BenchmarkId::new("dense", &p.label), &a_dense, |bench, a| {
            bench.iter_with_large_drop(|| svd(&backend, a, 1).unwrap());
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
            bench.iter_with_large_drop(|| trunc_svd_block_sparse(&backend, a, 1, &params).unwrap());
        });

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        group.bench_with_input(BenchmarkId::new("dense", &p.label), &a_dense, |bench, a| {
            bench.iter_with_large_drop(|| trunc_svd(&backend, a, 1, &params).unwrap());
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
            bench.iter_with_large_drop(|| qr_block_sparse(&backend, a, 1).unwrap());
        });

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        group.bench_with_input(BenchmarkId::new("dense", &p.label), &a_dense, |bench, a| {
            bench.iter_with_large_drop(|| qr(&backend, a, 1).unwrap());
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
            bench.iter_with_large_drop(|| lq_block_sparse(&backend, a, 1).unwrap());
        });

        let total = p.q * p.d;
        let a_dense = random_dense_matrix(total);
        group.bench_with_input(BenchmarkId::new("dense", &p.label), &a_dense, |bench, a| {
            bench.iter_with_large_drop(|| lq(&backend, a, 1).unwrap());
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
            |bench, (a, b)| {
                bench.iter_with_large_drop(|| {
                    contract_block_sparse(&backend, a, b, &[1], &[0]).unwrap()
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("matmul_dense", &p.label),
            &(&a_dense, &b_dense),
            |bench, (a, b)| {
                bench.iter_with_large_drop(|| contract(&backend, a, b, "ij,jk->ik").unwrap());
            },
        );

        // SVD
        group.bench_with_input(BenchmarkId::new("svd_bsp", &p.label), &a, |bench, a| {
            bench.iter_with_large_drop(|| svd_block_sparse(&backend, a, 1).unwrap());
        });
        group.bench_with_input(
            BenchmarkId::new("svd_dense", &p.label),
            &a_dense,
            |bench, a| {
                bench.iter_with_large_drop(|| svd(&backend, a, 1).unwrap());
            },
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
    bench_contract_rank3,
    bench_svd,
    bench_trunc_svd,
    bench_qr,
    bench_lq,
    bench_singh_reference,
);
criterion_main!(benches);
