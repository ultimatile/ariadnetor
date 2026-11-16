//! Bench branch only: conjugating vs plain `Complex<f64>` transpose through
//! `NativeBackend::transpose`, column-major, on the shapes of
//! tensor4all/strided-rs#467 / #478 plus their `2^k ± 1` neighbours.
//!
//! The plain column always takes `strided_perm::copy_into`. The conj column
//! takes whichever kernel the build selects: naive (default), HPTT
//! (`--features hptt`), or `strided_perm::copy_conj_into`
//! (`--features strided-conj`).
//!
//! Each row: `SAMPLES` samples; inside a sample plain and conj alternate
//! order and each runs `iters` times. Times are the minimum per-call time
//! over samples; `conj/plain` is the median of per-sample ratios.

use std::time::Instant;

use ariadnetor_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder, TransposeDescriptor};
use ariadnetor_native::NativeBackend;
use num_complex::Complex;

const SAMPLES: usize = 15;

fn kernel_name() -> &'static str {
    if cfg!(feature = "strided-conj") {
        "strided_perm::copy_conj_into"
    } else if cfg!(feature = "hptt") {
        "HPTT conjA"
    } else {
        "naive"
    }
}

/// The fixed part of one bench row: every timed call differs only in `conj`
/// and the iteration count.
struct Case<'a> {
    backend: &'a NativeBackend,
    input: &'a [Complex<f64>],
    shape: &'a [usize],
    perm: &'a [usize],
    policy: ExecPolicy,
}

fn run(case: &Case<'_>, output: &mut [Complex<f64>], conj: bool, iters: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        case.backend
            .transpose(TransposeDescriptor {
                input: case.input,
                output,
                shape: case.shape,
                perm: case.perm,
                order: MemoryOrder::ColumnMajor,
                conj,
                policy: case.policy,
            })
            .unwrap();
        std::hint::black_box(&*output);
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn row(backend: &NativeBackend, shape: &[usize], perm: &[usize], policy: ExecPolicy) {
    let n: usize = shape.iter().product();
    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(i as f64, -(i as f64) * 0.5))
        .collect();
    let mut output = vec![Complex::new(0.0, 0.0); n];
    let case = Case {
        backend,
        input: &input,
        shape,
        perm,
        policy,
    };

    // Correctness spot-check against the plain route.
    run(&case, &mut output, false, 1);
    let plain_out = output.clone();
    run(&case, &mut output, true, 1);
    assert!(
        plain_out.iter().zip(&output).all(|(p, c)| p.conj() == *c),
        "conj output differs from conj(plain) for {shape:?} {perm:?}"
    );

    // Size iterations so one batch is ~20 ms.
    let probe = run(&case, &mut output, false, 3);
    let iters = ((0.02 / probe).ceil() as usize).clamp(1, 100_000);

    let mut plain_min = f64::INFINITY;
    let mut conj_min = f64::INFINITY;
    let mut ratios = Vec::with_capacity(SAMPLES);
    for s in 0..SAMPLES {
        let (p, c) = if s % 2 == 0 {
            let p = run(&case, &mut output, false, iters);
            let c = run(&case, &mut output, true, iters);
            (p, c)
        } else {
            let c = run(&case, &mut output, true, iters);
            let p = run(&case, &mut output, false, iters);
            (p, c)
        };
        plain_min = plain_min.min(p);
        conj_min = conj_min.min(c);
        ratios.push(c / p);
    }
    let pol = match policy {
        ExecPolicy::Sequential => "seq",
        ExecPolicy::Parallel(_) => "par",
    };
    println!(
        "| {:<22} | {:<12} | {} | {:>10.3} | {:>10.3} | {:>6.2} |",
        format!("{shape:?}"),
        format!("{perm:?}"),
        pol,
        plain_min * 1e6,
        conj_min * 1e6,
        median(ratios)
    );
}

fn main() {
    let backend = NativeBackend::new();
    let cases: Vec<(Vec<usize>, Vec<usize>)> = [255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049]
        .iter()
        .map(|&n| (vec![n, n], vec![1, 0]))
        .chain(
            [63, 64, 65, 127, 128, 129]
                .iter()
                .map(|&n| (vec![n, n, n], vec![2, 0, 1])),
        )
        .chain(
            [31, 32, 33]
                .iter()
                .map(|&n| (vec![n, n, n, n], vec![3, 1, 0, 2])),
        )
        .chain(std::iter::once((vec![100, 2, 100], vec![2, 0, 1])))
        .collect();

    println!("conj kernel: {}", kernel_name());
    println!("| shape | perm | policy | plain µs | conj µs | conj/plain |");
    println!("|---|---|---|---|---|---|");
    for policy in [ExecPolicy::Sequential, ExecPolicy::Parallel(0)] {
        for (shape, perm) in &cases {
            row(&backend, shape, perm, policy);
        }
    }
}
