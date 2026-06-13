//! Benchmarks for the inherent unary scalar ops on the joined
//! `DenseTensor` surface (`scaled`, `norm`, `normalized`).

use arnet_tensor::DenseTensor;
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

fn shapes_normalized() -> Vec<TensorShape> {
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
            label: "64x4x64",
            shape: vec![64, 4, 64],
        },
    ]
}

fn random_tensor(shape: Vec<usize>) -> DenseTensor<f64> {
    DenseTensor::random(shape, &mut rng())
}

fn bench_scaled(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_scaled");
    for s in &shapes_all() {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter_with_large_drop(|| t.scaled(2.5));
        });
    }
    group.finish();
}

fn bench_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_norm");
    for s in &shapes_all() {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter(|| t.norm());
        });
    }
    group.finish();
}

fn bench_normalized(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_normalized");
    for s in &shapes_normalized() {
        let tensor = random_tensor(s.shape.clone());
        group.bench_with_input(BenchmarkId::from_parameter(s.label), &tensor, |b, t| {
            b.iter_with_large_drop(|| t.normalized());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_scaled, bench_norm, bench_normalized);
criterion_main!(benches);
