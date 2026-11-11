//! Threshold sweep: dense transpose time vs tensor size, comparing
//! `ExecPolicy::Sequential` and `ExecPolicy::Parallel(0)`.
//!
//! Goal: find the size at which parallel transpose starts to beat
//! sequential. The crossover is the threshold to use for per-call
//! `ExecPolicy` dispatch in `ariadnetor-native`.
//!
//! Each measurement uses `expert::permute` with a caller-provided
//! `ExecPolicy`. Global parallelism state is not consulted by the per-call
//! path and is intentionally not touched here.
//!
//! The sweeps cover `n × n` with `perm = [1, 0]` (up to 2048, the
//! parallel-dominant regime), `n × n × n` with `perm = [2, 0, 1]`, and
//! `n⁴` with `perm = [3, 1, 0, 2]`, for f64 and for `Complex<f64>`
//! (non-conjugating, so both take the strided-perm route).
//!
//! `ThresholdTable::transpose` keys on total element count (not side
//! length) — see `NativeBackend::par_for_transpose`, which feeds
//! `shape.iter().product()` into `policy_by_n`. The output includes a
//! `key(elems)` column giving this total so that calibration sets the
//! threshold in the correct unit.

use std::time::{Duration, Instant};

use rand::SeedableRng;

use ariadnetor_core::Complex;
use ariadnetor_core::backend::ExecPolicy;
use ariadnetor_linalg::expert::permute;
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::DenseTensor;

fn random_f64(shape: Vec<usize>, seed: u64) -> DenseTensor<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    DenseTensor::random(shape, &mut rng)
}

/// A complex tensor of the given shape. A transpose only moves elements, so
/// its time does not depend on their values.
fn filled_c64(shape: Vec<usize>) -> DenseTensor<Complex<f64>> {
    DenseTensor::filled(shape, Complex::new(0.5, -0.25))
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
        let target = if key >= 262_144 {
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
    let sizes_2d = [16usize, 32, 64, 128, 256, 512, 1024, 2048];
    let sizes_3d = [8usize, 16, 32, 64, 128, 256];
    let sizes_4d = [4usize, 8, 16, 32, 64];
    let perm_2d = [1usize, 0];
    let perm_3d = [2usize, 0, 1];
    let perm_4d = [3usize, 1, 0, 2];

    run_sweep(
        "Transpose f64 (2D n×n, perm=[1,0])",
        &sizes_2d,
        |n| random_f64(vec![n, n], 42),
        |t| t.len(),
        |t, policy| {
            let _ = permute(&backend, t, &perm_2d, policy).unwrap();
        },
    );
    run_sweep(
        "Transpose f64 (3D n×n×n, perm=[2,0,1])",
        &sizes_3d,
        |n| random_f64(vec![n, n, n], 42),
        |t| t.len(),
        |t, policy| {
            let _ = permute(&backend, t, &perm_3d, policy).unwrap();
        },
    );
    run_sweep(
        "Transpose f64 (4D n⁴, perm=[3,1,0,2])",
        &sizes_4d,
        |n| random_f64(vec![n, n, n, n], 42),
        |t| t.len(),
        |t, policy| {
            let _ = permute(&backend, t, &perm_4d, policy).unwrap();
        },
    );
    run_sweep(
        "Transpose c64 (2D n×n, perm=[1,0])",
        &sizes_2d,
        |n| filled_c64(vec![n, n]),
        |t| t.len(),
        |t, policy| {
            let _ = permute(&backend, t, &perm_2d, policy).unwrap();
        },
    );
    run_sweep(
        "Transpose c64 (3D n×n×n, perm=[2,0,1])",
        &sizes_3d,
        |n| filled_c64(vec![n, n, n]),
        |t| t.len(),
        |t, policy| {
            let _ = permute(&backend, t, &perm_3d, policy).unwrap();
        },
    );

    eprintln!(
        "\n`key(elems)` is the value ThresholdTable::transpose compares against.\n\
         ratio < 1: parallel wins (set threshold ≤ key(elems))\n\
         ratio > 1: sequential wins (threshold above key(elems))"
    );
}
