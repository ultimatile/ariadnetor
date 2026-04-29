use arnet_tensor::{Dense, MemoryOrder};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rng;

// Shape definitions for reuse across benchmarks
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

fn shapes_rect() -> Vec<TensorShape> {
    vec![
        TensorShape {
            label: "1024x64",
            shape: vec![1024, 64],
        },
        TensorShape {
            label: "64x1024",
            shape: vec![64, 1024],
        },
    ]
}

fn shapes_rank3() -> Vec<TensorShape> {
    vec![TensorShape {
        label: "64x4x64",
        shape: vec![64, 4, 64],
    }]
}

#[allow(dead_code)]
fn shapes_rank4() -> Vec<TensorShape> {
    vec![TensorShape {
        label: "32x4x4x32",
        shape: vec![32, 4, 4, 32],
    }]
}

fn random_tensor(shape: Vec<usize>) -> Dense<f64> {
    Dense::random(shape, &mut rng())
}

/// Create a uniquely-owned copy (Arc refcount = 1) so mutating ops
/// don't trigger copy-on-write during the timed section.
fn unique_copy(t: &Dense<f64>) -> Dense<f64> {
    Dense::new(t.data().to_vec(), t.shape().to_vec())
}

// ==========================================================================
// map
// ==========================================================================

fn bench_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("map");

    let shapes: Vec<TensorShape> = shapes_square().into_iter().chain(shapes_rank3()).collect();

    for s in &shapes {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter_with_large_drop(|| t.map(|&x| x * 2.0 + 1.0));
        });
    }

    group.finish();
}

// ==========================================================================
// map_mut
// ==========================================================================

fn bench_map_mut(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_mut");

    let shapes: Vec<TensorShape> = shapes_square().into_iter().chain(shapes_rank3()).collect();

    for s in &shapes {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter_batched_ref(
                || unique_copy(t),
                |t| t.map_mut(|x| *x * 2.0 + 1.0),
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// ==========================================================================
// norm_squared (via norm_frobenius which calls norm_squared internally)
// ==========================================================================

fn bench_norm_frobenius(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm_frobenius");

    let shapes: Vec<TensorShape> = shapes_square()
        .into_iter()
        .chain(shapes_rect())
        .chain(shapes_rank3())
        .collect();

    for s in &shapes {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter(|| t.norm_frobenius());
        });
    }

    group.finish();
}

// ==========================================================================
// scale (in-place)
// ==========================================================================

fn bench_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale");

    let shapes: Vec<TensorShape> = shapes_square().into_iter().chain(shapes_rank3()).collect();

    for s in &shapes {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter_batched_ref(
                || unique_copy(t),
                |t| t.scale(2.5),
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// ==========================================================================
// linear_combine
// ==========================================================================

fn bench_linear_combine(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_combine");

    for s in &shapes_square() {
        let a = random_tensor(s.shape.clone());
        let b = random_tensor(s.shape.clone());
        let tensors = [&a, &b];
        let coefs = [0.7, 0.3];
        group.bench_with_input(
            BenchmarkId::from_parameter(s.label),
            &s.label,
            |bench, _| {
                bench.iter_with_large_drop(|| Dense::linear_combine(&tensors, &coefs).unwrap());
            },
        );
    }

    group.finish();
}

// ==========================================================================
// normalize
// ==========================================================================

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");

    let shapes: Vec<TensorShape> = shapes_square().into_iter().chain(shapes_rank3()).collect();

    for s in &shapes {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter_batched_ref(
                || unique_copy(t),
                |t| t.normalize(),
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// ==========================================================================
// slice
// ==========================================================================

fn bench_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice");

    for s in &[
        TensorShape {
            label: "256x256",
            shape: vec![256, 256],
        },
        TensorShape {
            label: "1024x1024",
            shape: vec![1024, 1024],
        },
    ] {
        let tensor = random_tensor(s.shape.clone());
        let half: Vec<(usize, usize)> = s.shape.iter().map(|&d| (0, d / 2)).collect();
        group.bench_with_input(
            BenchmarkId::new("half", s.label),
            &(&tensor, half),
            |b, (t, ranges)| {
                b.iter_with_large_drop(|| t.slice(ranges, MemoryOrder::RowMajor));
            },
        );
    }

    let tensor = random_tensor(vec![64, 4, 64]);
    group.bench_with_input(BenchmarkId::new("half", "64x4x64"), &tensor, |b, t| {
        b.iter_with_large_drop(|| t.slice(&[(0, 32), (0, 4), (0, 64)], MemoryOrder::RowMajor));
    });

    group.finish();
}

// ==========================================================================
// concatenate
// ==========================================================================

fn bench_concatenate(c: &mut Criterion) {
    let mut group = c.benchmark_group("concatenate");

    for s in &[
        TensorShape {
            label: "256x256",
            shape: vec![256, 256],
        },
        TensorShape {
            label: "1024x1024",
            shape: vec![1024, 1024],
        },
    ] {
        let a = random_tensor(s.shape.clone());
        let b = random_tensor(s.shape.clone());
        let tensors = [&a, &b];
        group.bench_with_input(BenchmarkId::new("axis0", s.label), &s.label, |bench, _| {
            bench.iter_with_large_drop(|| Dense::concatenate(&tensors, 0, MemoryOrder::RowMajor));
        });
    }

    let a = random_tensor(vec![64, 4, 64]);
    let b = random_tensor(vec![64, 4, 64]);
    let tensors = [&a, &b];
    group.bench_with_input(
        BenchmarkId::new("axis0", "64x4x64"),
        &"64x4x64",
        |bench, _| {
            bench.iter_with_large_drop(|| Dense::concatenate(&tensors, 0, MemoryOrder::RowMajor));
        },
    );

    group.finish();
}

// ==========================================================================
// expand
// ==========================================================================

fn bench_expand(c: &mut Criterion) {
    let mut group = c.benchmark_group("expand");

    for s in &[
        TensorShape {
            label: "256x256",
            shape: vec![256, 256],
        },
        TensorShape {
            label: "1024x1024",
            shape: vec![1024, 1024],
        },
    ] {
        let tensor = random_tensor(s.shape.clone());
        let padding: Vec<(usize, usize)> = vec![(16, 16); s.shape.len()];
        group.bench_with_input(
            BenchmarkId::new("pad16", s.label),
            &(&tensor, &padding),
            |b, (t, p)| {
                b.iter_with_large_drop(|| t.expand(p, MemoryOrder::RowMajor));
            },
        );
    }

    let tensor = random_tensor(vec![64, 4, 64]);
    group.bench_with_input(BenchmarkId::new("pad16", "64x4x64"), &tensor, |b, t| {
        b.iter_with_large_drop(|| t.expand(&[(16, 16), (0, 0), (16, 16)], MemoryOrder::RowMajor));
    });

    group.finish();
}

// ==========================================================================
// replace_slice
// ==========================================================================

fn bench_replace_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("replace_slice");

    let dst = random_tensor(vec![256, 256]);
    let sub = random_tensor(vec![128, 128]);
    group.bench_function("128x128_into_256x256", |b| {
        b.iter_batched_ref(
            || unique_copy(&dst),
            |d| d.replace_slice(&sub, &[64, 64], MemoryOrder::RowMajor),
            criterion::BatchSize::LargeInput,
        );
    });

    let dst = random_tensor(vec![1024, 1024]);
    let sub = random_tensor(vec![512, 512]);
    group.bench_function("512x512_into_1024x1024", |b| {
        b.iter_batched_ref(
            || unique_copy(&dst),
            |d| d.replace_slice(&sub, &[256, 256], MemoryOrder::RowMajor),
            criterion::BatchSize::LargeInput,
        );
    });

    let dst = random_tensor(vec![64, 4, 64]);
    let sub = random_tensor(vec![32, 4, 64]);
    group.bench_function("32x4x64_into_64x4x64", |b| {
        b.iter_batched_ref(
            || unique_copy(&dst),
            |d| d.replace_slice(&sub, &[16, 0, 0], MemoryOrder::RowMajor),
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_map,
    bench_map_mut,
    bench_norm_frobenius,
    bench_scale,
    bench_linear_combine,
    bench_normalize,
    bench_slice,
    bench_concatenate,
    bench_expand,
    bench_replace_slice,
);
criterion_main!(benches);
