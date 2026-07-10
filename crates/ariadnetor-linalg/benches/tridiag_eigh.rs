//! Specialized-vs-dense eigensolver bench on symmetric tridiagonal
//! input: `tridiag_eigh_with_backend` on the diagonal / subdiagonal
//! against `eigh_with_backend` on the assembled dense matrix.
//!
//! This is the perf-acceptance measurement for routing the Lanczos
//! tridiagonal eigenproblem through the specialized path: the dense arm
//! pays the O(m^3) tridiagonalization the specialized arm skips. Sizes
//! target the large-m regime where that term dominates; DMRG-scale
//! problems (m around 20-50) are H*v-bound and expected neutral.

// Shared with the `tridiag_eigen` contract test so the bench measures
// exactly the matrix class the tests verify.
#[path = "../tests/tridiag_fixtures/mod.rs"]
mod tridiag_fixtures;

use ariadnetor_linalg::{eigh_with_backend, tridiag_eigh_with_backend};
use ariadnetor_native::NativeBackend;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tridiag_fixtures::{assemble_dense, fixture};

const SIZES: [usize; 3] = [64, 256, 512];

fn bench_tridiag_vs_dense(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("tridiag_eigh");

    for &n in &SIZES {
        let (d, e) = fixture::<f64>(n);
        let dense = assemble_dense(&d, &e);

        group.bench_with_input(BenchmarkId::new("tridiag", n), &(&d, &e), |b, (d, e)| {
            b.iter_with_large_drop(|| tridiag_eigh_with_backend(&backend, d, e).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("dense_eigh", n), &dense, |b, t| {
            b.iter_with_large_drop(|| eigh_with_backend(&backend, t, 1).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_tridiag_vs_dense);
criterion_main!(benches);
