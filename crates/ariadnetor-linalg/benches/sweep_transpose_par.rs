//! Threshold sweep: dense 2D transpose time vs square matrix size,
//! comparing `ExecPolicy::Sequential` and `ExecPolicy::Parallel(0)`.
//!
//! Goal: find the matrix size at which parallel transpose starts to
//! beat sequential. The crossover is the threshold to use for per-call
//! `ExecPolicy` dispatch in `arnet-native` (currently `usize::MAX` on
//! laptop — unmeasured per ADR-0008).
//!
//! Each measurement uses `expert::transpose(.., perm=[1, 0], ..)`
//! with a caller-provided `ExecPolicy`. Global parallelism state is not
//! consulted by the per-call path and is intentionally not touched here.
//!
//! Each size `n` measures permuting an `n × n` tensor with `perm = [1, 0]`.
//! The upper size extends to 2048 to reach the parallel-dominant regime.
//!
//! `ThresholdTable::transpose` keys on total element count (not side
//! length) — see `NativeBackend::par_for_transpose`, which feeds
//! `shape.iter().product()` into `policy_by_n`. The output includes a
//! `key(elems)` column giving this total so that calibration sets the
//! threshold in the correct unit.

use std::time::{Duration, Instant};

use rand::SeedableRng;

use arnet_core::backend::ExecPolicy;
use arnet_linalg::expert::transpose;
use arnet_native::NativeBackend;
use arnet_tensor::DenseTensor;

fn random_square(n: usize, seed: u64) -> DenseTensor<f64> {
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

fn run_sweep<State, MF, KF, OF>(label: &str, sizes: &[usize], make: MF, key_of: KF, op: OF)
where
    MF: Fn(usize) -> State,
    KF: Fn(&State) -> usize,
    OF: Fn(&State, ExecPolicy),
{
    eprintln!("\n=== {label} ===");
    eprintln!(
        "{:>6} {:>12} {:>8} {:>8} {:>14} {:>14} {:>10}",
        "n", "key(elems)", "iters_s", "iters_p", "Sequential", "Parallel(0)", "ratio(P/S)"
    );
    eprintln!("{}", "-".repeat(79));

    for &n in sizes {
        let state = make(n);
        let key = key_of(&state);
        let target = if n >= 512 {
            Duration::from_millis(500)
        } else {
            Duration::from_millis(150)
        };

        let (t_seq, iters_seq) = measure(target, || op(&state, ExecPolicy::Sequential));
        let (t_par, iters_par) = measure(target, || op(&state, ExecPolicy::Parallel(0)));

        let ratio = t_par.as_secs_f64() / t_seq.as_secs_f64();
        eprintln!(
            "{:>6} {:>12} {:>8} {:>8} {:>14.3?} {:>14.3?} {:>10.3}x",
            n, key, iters_seq, iters_par, t_seq, t_par, ratio
        );
    }
}

fn main() {
    let backend = NativeBackend::new();
    let sizes = [16usize, 32, 64, 128, 256, 512, 1024, 2048];

    run_sweep(
        "Transpose (2D n×n, perm=[1,0])",
        &sizes,
        |n| random_square(n, 42),
        |t| t.len(),
        |t, policy| {
            let _ = transpose(&backend, t, &[1, 0], policy).unwrap();
        },
    );

    eprintln!(
        "\n`key(elems)` is the value ThresholdTable::transpose compares against.\n\
         ratio < 1: parallel wins (set threshold ≤ key(elems))\n\
         ratio > 1: sequential wins (threshold above key(elems))"
    );
}
