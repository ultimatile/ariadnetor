//! Micro-benchmark: cross-order reorder — naive `reorder_data` vs the
//! HPTT-routable `ComputeBackend::transpose`.
//!
//! The linalg row-major sandwich (decomposition / contract / expm / solve /
//! eigen / einsum) normalizes rank>2 operands across memory order with
//! `ariadnetor_tensor::reorder_data`, a backend-less per-element loop that
//! never reaches HPTT. A cross-order reorder of the same logical shape is a
//! physical axis-reversal permutation, so `backend.transpose` with a reverse
//! perm and the source order produces the identical bytes and uses HPTT under
//! `--features hptt`.
//!
//! This group compares the two at representative shapes so the naive-vs-kernel
//! crossover can be read per shape: the default build times the native naive
//! transpose kernel, `--features hptt` times HPTT, both against the same naive
//! `reorder_data` baseline, via criterion baselines. Both paths allocate their
//! output per iteration, matching the real call sites (`reorder_data` returns a
//! fresh tensor; the routed helper allocates a fresh buffer).

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder, TransposeDescriptor};
use ariadnetor_linalg::svd;
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseTensor, DenseTensorData, reorder_data};

/// Representative reorder shapes: tall-skinny 2D, square 2D, rank-3, rank-4,
/// each at a small and a large per-axis size (DMRG / MPS bond scales). `large`
/// marks the shapes carried into the parallel sweep.
struct ShapeCase {
    label: &'static str,
    shape: Vec<usize>,
    large: bool,
}

fn shapes() -> Vec<ShapeCase> {
    let mk = |label, shape: &[usize], large| ShapeCase {
        label,
        shape: shape.to_vec(),
        large,
    };
    vec![
        mk("tall2d_small", &[256, 8], false),
        mk("tall2d_large", &[4096, 32], true),
        mk("square2d_small", &[64, 64], false),
        mk("square2d_large", &[512, 512], true),
        mk("rank3_small", &[32, 8, 32], false),
        mk("rank3_large", &[128, 16, 128], true),
        mk("rank4_small", &[16, 8, 8, 16], false),
        mk("rank4_large", &[64, 8, 8, 64], true),
    ]
}

/// Time the naive `reorder_data` and the routed `backend.transpose` for one
/// scalar type across the shape sweep (both sequential), then the routed path
/// under `Parallel(0)` at the large shapes only.
fn bench_scalar<T: Scalar>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    backend: &NativeBackend,
    tag: &str,
) {
    // Source order = the backend's preferred (column-major) layout, matching
    // the operand a sandwich site holds; target = the row-major intermediate.
    let src = MemoryOrder::ColumnMajor;
    let dst = MemoryOrder::RowMajor;

    for case in shapes() {
        let total: usize = case.shape.iter().product();
        let tensor =
            DenseTensorData::from_raw_parts(vec![T::one(); total], case.shape.clone(), src);
        let perm: Vec<usize> = (0..case.shape.len()).rev().collect();

        // One routed run under a given policy; both the sequential and the
        // parallel bench measure this same kernel path, differing only in policy.
        let run_routed = |policy: ExecPolicy| {
            let mut out = vec![T::zero(); total];
            backend
                .transpose(TransposeDescriptor {
                    input: tensor.data(),
                    output: &mut out,
                    shape: &case.shape,
                    perm: &perm,
                    order: src,
                    conj: false,
                    policy,
                })
                .unwrap();
            std::hint::black_box(&out);
        };

        group.bench_function(BenchmarkId::new(format!("naive_{tag}"), case.label), |b| {
            b.iter(|| {
                let out = reorder_data(&tensor, dst);
                std::hint::black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new(format!("routed_{tag}"), case.label), |b| {
            b.iter(|| run_routed(ExecPolicy::Sequential));
        });

        if case.large {
            group.bench_function(
                BenchmarkId::new(format!("routed_par_{tag}"), case.label),
                |b| {
                    b.iter(|| run_routed(ExecPolicy::Parallel(0)));
                },
            );
        }
    }
}

fn bench_reorder(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("reorder_micro");

    bench_scalar::<f64>(&mut group, &backend, "f64");
    bench_scalar::<f32>(&mut group, &backend, "f32");
    bench_scalar::<ariadnetor_core::Complex<f64>>(&mut group, &backend, "c64");
    bench_scalar::<ariadnetor_core::Complex<f32>>(&mut group, &backend, "c32");

    group.finish();
}

/// End-to-end probe: a rank-3 SVD whose operand reshape routes its cross-order
/// reorder through the backend transpose. Unlike the isolated `reorder_micro`
/// group, this drives a whole decomposition, so the routed reorder is a real
/// (faer-SVD-diluted) fraction of the wall time — it confirms the routed path
/// is exercised end to end and moves with HPTT under `--features hptt`. `nrow=1`
/// splits axis 0 from the rest, so the rank>2 row-major sandwich runs.
fn bench_decomp_e2e(c: &mut Criterion) {
    let backend = NativeBackend::new();
    let mut group = c.benchmark_group("decomp_e2e");

    for (label, shape) in [
        ("rank3_64x8x64", vec![64usize, 8, 64]),
        ("rank3_128x4x128", vec![128usize, 4, 128]),
    ] {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tensor = DenseTensor::<f64>::random(shape, &mut rng);
        group.bench_function(BenchmarkId::new("svd", label), |b| {
            b.iter_with_large_drop(|| svd(&backend, &tensor, 1).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_reorder, bench_decomp_e2e);
criterion_main!(benches);
