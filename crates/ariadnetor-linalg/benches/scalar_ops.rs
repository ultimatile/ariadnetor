use arnet_linalg::{linear_combine, norm, normalize, scale, trace};
use arnet_tensor::Dense;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rng;

struct TensorShape {
    label: &'static str,
    shape: Vec<usize>,
}

fn shapes_all() -> Vec<TensorShape> {
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
        TensorShape {
            label: "1024x64",
            shape: vec![1024, 64],
        },
        TensorShape {
            label: "64x4x64",
            shape: vec![64, 4, 64],
        },
    ]
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

fn random_tensor(shape: Vec<usize>) -> Dense<f64> {
    Dense::random(shape, &mut rng())
}

// ==========================================================================
// scale (out-of-place)
// ==========================================================================

fn bench_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_scale");

    for s in &shapes_all() {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter_with_large_drop(|| scale(t, 2.5));
        });
    }

    group.finish();
}

// ==========================================================================
// norm
// ==========================================================================

fn bench_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_norm");

    for s in &shapes_all() {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter(|| norm(t));
        });
    }

    group.finish();
}

// ==========================================================================
// normalize
// ==========================================================================

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_normalize");

    let shapes: Vec<TensorShape> = vec![
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
        TensorShape {
            label: "64x4x64",
            shape: vec![64, 4, 64],
        },
    ];

    for s in &shapes {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter_with_large_drop(|| normalize(t));
        });
    }

    group.finish();
}

// ==========================================================================
// linear_combine
// ==========================================================================

fn bench_linear_combine(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_linear_combine");

    let shapes: Vec<TensorShape> = vec![
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
        TensorShape {
            label: "64x4x64",
            shape: vec![64, 4, 64],
        },
    ];

    for s in &shapes {
        let a = random_tensor(s.shape.clone());
        let b = random_tensor(s.shape.clone());
        let tensors = [&a, &b];
        let coefs = [0.7, 0.3];
        group.bench_with_input(
            BenchmarkId::from_parameter(s.label),
            &s.label,
            |bench, _| {
                bench.iter_with_large_drop(|| linear_combine(&tensors, &coefs).unwrap());
            },
        );
    }

    group.finish();
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

criterion_group!(
    benches,
    bench_scale,
    bench_norm,
    bench_normalize,
    bench_linear_combine,
    bench_trace,
);
criterion_main!(benches);
