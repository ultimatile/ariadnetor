use arnet_linalg::{diagonal_scale, trace};
use arnet_native::NativeBackend;
use arnet_tensor::DenseTensor;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rng;

struct TensorShape {
    label: &'static str,
    shape: Vec<usize>,
}

fn shapes_square() -> Vec<TensorShape> {
    vec![
        TensorShape {
            label: "64x64",
            shape: vec![64, 64],
        },
        TensorShape {
            label: "256x256",
            shape: vec![256, 256],
        },
        TensorShape {
            label: "1024x1024",
            shape: vec![1024, 1024],
        },
    ]
}

fn random_tensor(shape: Vec<usize>) -> DenseTensor<f64, NativeBackend> {
    DenseTensor::random(shape, &mut rng())
}

// ==========================================================================
// trace
// ==========================================================================

fn bench_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_trace");

    // Full trace of square matrix: tr(A)
    for s in &shapes_square() {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::new("full", s.label), &tensor, |b, t| {
            b.iter_with_large_drop(|| trace(t, &[(0, 1)]).unwrap());
        });
    }

    // Partial trace of rank-3 tensor: trace over axes (0, 2) -> vector
    let tensor = random_tensor(vec![64, 4, 64]);
    group.bench_with_input(BenchmarkId::new("partial", "64x4x64"), &tensor, |b, t| {
        b.iter_with_large_drop(|| trace(t, &[(0, 2)]).unwrap());
    });

    // Partial trace of rank-4 tensor: trace over axes (0, 3) -> rank-2
    let tensor = random_tensor(vec![32, 4, 4, 32]);
    group.bench_with_input(BenchmarkId::new("partial", "32x4x4x32"), &tensor, |b, t| {
        b.iter_with_large_drop(|| trace(t, &[(0, 3)]).unwrap());
    });

    group.finish();
}

// ==========================================================================
// diagonal_scale
// ==========================================================================

fn bench_diagonal_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_diagonal_scale");

    // Scale axis 0 (rows) of a square matrix
    for s in &shapes_square() {
        let tensor = random_tensor(s.shape.clone());
        let weights: Vec<f64> = (0..s.shape[0]).map(|i| (i + 1) as f64).collect();
        group.bench_with_input(
            BenchmarkId::new("axis0", s.label),
            &(&tensor, &weights),
            |b, (t, w)| {
                b.iter_with_large_drop(|| diagonal_scale(t, w, 0).unwrap());
            },
        );
    }

    // Scale axis 1 (columns) of a square matrix
    for s in &shapes_square() {
        let tensor = random_tensor(s.shape.clone());
        let weights: Vec<f64> = (0..s.shape[1]).map(|i| (i + 1) as f64).collect();
        group.bench_with_input(
            BenchmarkId::new("axis1", s.label),
            &(&tensor, &weights),
            |b, (t, w)| {
                b.iter_with_large_drop(|| diagonal_scale(t, w, 1).unwrap());
            },
        );
    }

    // Rank-3: scale axis 1
    let tensor = random_tensor(vec![64, 4, 64]);
    let weights: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    group.bench_with_input(
        BenchmarkId::new("axis1", "64x4x64"),
        &(&tensor, &weights),
        |b, (t, w)| {
            b.iter_with_large_drop(|| diagonal_scale(t, w, 1).unwrap());
        },
    );

    group.finish();
}

criterion_group!(benches, bench_trace, bench_diagonal_scale,);
criterion_main!(benches);
