//! Threshold sweep: dense SVD/QR/LQ/eigh/eig time vs matrix size,
//! comparing `ExecPolicy::Sequential` and `ExecPolicy::Parallel(0)`.
//!
//! Goal: find the matrix size at which parallel decomposition starts
//! to beat sequential. The crossover is the threshold to use for
//! per-call `ExecPolicy` dispatch in `arnet-native`.
//!
//! Each op is measured through the `*_with_policy` expert-layer entry
//! point with an explicit `ExecPolicy`, so the sweep exercises the
//! two branches of the dispatch decision directly. Global parallelism
//! state (`faer::set_global_parallelism`) is not consulted by the
//! per-call path and is intentionally not touched here.

use std::time::{Duration, Instant};

use rand::SeedableRng;

use arnet_core::backend::ExecPolicy;
use arnet_linalg::{
    eig_with_policy_dense as eig_with_policy, eigh_with_policy_dense as eigh_with_policy,
    lq_with_policy_dense as lq_with_policy, qr_with_policy_dense as qr_with_policy,
    svd_with_policy_dense as svd_with_policy,
};
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, MemoryOrder};

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
    Dense::new(out, vec![n, n], MemoryOrder::ColumnMajor)
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
    OF: Fn(&Dense<f64>, ExecPolicy),
{
    eprintln!("\n=== {label} ===");
    eprintln!(
        "{:>6} {:>8} {:>8} {:>14} {:>14} {:>10}",
        "n", "iters_s", "iters_p", "Sequential", "Parallel(0)", "ratio(P/S)"
    );
    eprintln!("{}", "-".repeat(67));

    for &n in sizes {
        let mat = make(n);
        let target = if n >= 512 {
            Duration::from_millis(500)
        } else {
            Duration::from_millis(150)
        };

        let (t_seq, iters_seq) = measure(target, || op(&mat, ExecPolicy::Sequential));
        let (t_par, iters_par) = measure(target, || op(&mat, ExecPolicy::Parallel(0)));

        let ratio = t_par.as_secs_f64() / t_seq.as_secs_f64();
        eprintln!(
            "{:>6} {:>8} {:>8} {:>14.3?} {:>14.3?} {:>10.3}x",
            n, iters_seq, iters_par, t_seq, t_par, ratio
        );
    }
}

fn main() {
    let backend = NativeBackend::new();
    let sizes = [16usize, 32, 64, 128, 256, 512, 1024];

    run_sweep("SVD (thin)", &sizes, random_dense, |m, policy| {
        let _ = svd_with_policy(&backend, m, 1, policy).unwrap();
    });
    run_sweep("QR", &sizes, random_dense, |m, policy| {
        let _ = qr_with_policy(&backend, m, 1, policy).unwrap();
    });
    run_sweep("LQ", &sizes, random_dense, |m, policy| {
        let _ = lq_with_policy(&backend, m, 1, policy).unwrap();
    });
    run_sweep("eigh (symmetric)", &sizes, random_symmetric, |m, policy| {
        let _ = eigh_with_policy(&backend, m, 1, policy).unwrap();
    });
    run_sweep("eig (general)", &sizes, random_dense, |m, policy| {
        let _ = eig_with_policy(&backend, m, 1, policy).unwrap();
    });

    eprintln!(
        "\nratio < 1: parallel wins (threshold → Parallel)\nratio > 1: sequential wins (threshold above n)"
    );
}
