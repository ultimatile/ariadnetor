//! Threshold sweep: dense SVD/QR/LQ/eigh/eig time vs matrix size,
//! comparing `faer::Par::Seq` and `faer::Par::Rayon(NCPU)`.
//!
//! Goal: find the matrix size at which parallel decomposition starts
//! to beat sequential. The crossover is the threshold to use for
//! per-call `Par` dispatch in `arnet-native`.

use std::time::{Duration, Instant};

use rand::SeedableRng;

use arnet_linalg::{eig, eigh, lq, qr, svd};
use arnet_native::NativeBackend;
use arnet_tensor::Dense;

fn random_dense(n: usize) -> Dense<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    Dense::random(vec![n, n], &mut rng)
}

fn random_symmetric(n: usize) -> Dense<f64> {
    // A + A^T yields a real symmetric matrix suitable for eigh.
    let a = random_dense(n);
    let src = a.data();
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = src[i * n + j] + src[j * n + i];
        }
    }
    Dense::new(out, vec![n, n])
}

fn measure<F: FnMut()>(target: Duration, mut f: F) -> (Duration, u32) {
    f();
    let probe_start = Instant::now();
    f();
    let per = probe_start.elapsed();
    let iters =
        ((target.as_nanos() as u64 / per.as_nanos().max(1) as u64).max(1) as u32).min(10_000);

    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    (start.elapsed() / iters, iters)
}

fn run_sweep<MF, OF>(label: &str, sizes: &[usize], make: MF, op: OF)
where
    MF: Fn(usize) -> Dense<f64>,
    OF: Fn(&Dense<f64>),
{
    eprintln!("\n=== {label} ===");
    eprintln!(
        "{:>6} {:>10} {:>12} {:>12} {:>10}",
        "n", "iters", "Seq", "Rayon", "ratio(P/S)"
    );
    eprintln!("{}", "-".repeat(56));

    for &n in sizes {
        let mat = make(n);
        let target = if n >= 512 {
            Duration::from_millis(500)
        } else {
            Duration::from_millis(150)
        };

        faer::set_global_parallelism(faer::Par::Seq);
        let (t_seq, iters) = measure(target, || op(&mat));

        faer::set_global_parallelism(faer::Par::rayon(0));
        let (t_par, _) = measure(target, || op(&mat));

        let ratio = t_par.as_secs_f64() / t_seq.as_secs_f64();
        eprintln!(
            "{:>6} {:>10} {:>12.3?} {:>12.3?} {:>10.3}x",
            n, iters, t_seq, t_par, ratio
        );
    }
}

fn main() {
    let backend = NativeBackend::new();
    let sizes = [16usize, 32, 64, 128, 256, 512, 1024];

    run_sweep("SVD (thin)", &sizes, random_dense, |m| {
        let _ = svd(&backend, m, 1).unwrap();
    });
    run_sweep("QR", &sizes, random_dense, |m| {
        let _ = qr(&backend, m, 1).unwrap();
    });
    run_sweep("LQ", &sizes, random_dense, |m| {
        let _ = lq(&backend, m, 1).unwrap();
    });
    run_sweep("eigh (symmetric)", &sizes, random_symmetric, |m| {
        let _ = eigh(&backend, m, 1).unwrap();
    });
    run_sweep("eig (general)", &sizes, random_dense, |m| {
        let _ = eig(&backend, m, 1).unwrap();
    });

    eprintln!(
        "\nratio < 1: parallel wins (use Par::Rayon)\nratio > 1: sequential wins (use Par::Seq)"
    );
}
