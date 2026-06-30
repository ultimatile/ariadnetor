//! Threshold sweep: dense GEMM time vs square matrix size,
//! comparing `ExecPolicy::Sequential` and `ExecPolicy::Parallel(0)`.
//!
//! Goal: find the matrix size at which parallel GEMM starts to beat
//! sequential. The crossover is the threshold to use for per-call
//! `ExecPolicy` dispatch in `ariadnetor-native` (currently `usize::MAX` on
//! laptop — unmeasured per ADR-0008).
//!
//! `ariadnetor_linalg` does not expose a dedicated `gemm_with_policy`. The
//! linalg-layer entry that probes `par_for_gemm(m, n, k)` is
//! `expert::contract(.., "ij,jk->ik", ..)`: it calls `backend.gemm`
//! with a caller-provided `ExecPolicy`. Global parallelism state is not
//! consulted by the per-call path and is intentionally not touched here.
//!
//! Each size `n` measures the square `(n, n) × (n, n) → (n, n)` GEMM.

use std::time::{Duration, Instant};

use rand::SeedableRng;

use ariadnetor_core::backend::ExecPolicy;
use ariadnetor_linalg::expert::contract;
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::DenseTensor;

fn random_dense(n: usize, seed: u64) -> DenseTensor<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    DenseTensor::random(vec![n, n], &mut rng)
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

fn run_sweep<State, MF, OF>(label: &str, sizes: &[usize], make: MF, op: OF)
where
    MF: Fn(usize) -> State,
    OF: Fn(&State, ExecPolicy),
{
    eprintln!("\n=== {label} ===");
    eprintln!(
        "{:>6} {:>8} {:>8} {:>14} {:>14} {:>10}",
        "n", "iters_s", "iters_p", "Sequential", "Parallel(0)", "ratio(P/S)"
    );
    eprintln!("{}", "-".repeat(67));

    for &n in sizes {
        let state = make(n);
        let target = if n >= 512 {
            Duration::from_millis(500)
        } else {
            Duration::from_millis(150)
        };

        let (t_seq, iters_seq) = measure(target, || op(&state, ExecPolicy::Sequential));
        let (t_par, iters_par) = measure(target, || op(&state, ExecPolicy::Parallel(0)));

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

    run_sweep(
        "GEMM (square m=n=k)",
        &sizes,
        |n| (random_dense(n, 42), random_dense(n, 43)),
        |(a, b), policy| {
            let _ = contract(&backend, a, b, "ij,jk->ik", policy).unwrap();
        },
    );

    eprintln!(
        "\nratio < 1: parallel wins (threshold → Parallel)\nratio > 1: sequential wins (threshold above n)"
    );
}
