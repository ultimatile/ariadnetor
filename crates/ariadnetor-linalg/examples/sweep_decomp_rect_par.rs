//! Rectangular threshold sweep: dense SVD/QR/LQ time vs `(m, n)` shape,
//! comparing `ExecPolicy::Sequential` and `ExecPolicy::Parallel(0)`.
//!
//! Goal: collect calibration data for choosing between two candidate
//! work proxies:
//!
//! - `min(m, n)` — the current proxy used by `par_for_{svd, qr, lq}`
//! - `cbrt(m * n * min(m, n))` — scales with aspect ratio
//!
//! On square inputs the two candidates agree (both equal `n`), so
//! `sweep_decomp_par.rs` cannot tell them apart. On rectangular inputs
//! they diverge: e.g. `(m, n) = (512, 64)` gives `min = 64` but
//! `cbrt_work ≈ 128`. If the Sequential-vs-Parallel crossover at fixed
//! `min(m, n)` is stable across aspect ratios, the current proxy is
//! adequate; if it tracks `cbrt_work`, sub-issue D should switch.
//!
//! Only tall grids (`m >= n`) are measured here — this is sufficient to
//! distinguish the proxies. `eigh` and `eig` are skipped because they
//! require square input.

use std::time::{Duration, Instant};

use rand::SeedableRng;

use arnet_core::backend::ExecPolicy;
use arnet_linalg::{lq_with_policy, qr_with_policy, svd_with_policy};
use arnet_native::NativeBackend;
use arnet_tensor::DenseTensorData;

fn random_rect(m: usize, n: usize, seed: u64) -> DenseTensorData<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    DenseTensorData::random(vec![m, n], &mut rng)
}

fn cbrt_work(m: usize, n: usize) -> usize {
    let k = m.min(n);
    ((m * n * k) as f64).cbrt() as usize
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

fn run_sweep_rect<OF>(label: &str, grid: &[(usize, usize)], op: OF)
where
    OF: Fn(&DenseTensorData<f64>, ExecPolicy),
{
    eprintln!("\n=== {label} ===");
    eprintln!(
        "{:>5} {:>5} {:>10} {:>10} {:>8} {:>8} {:>14} {:>14} {:>10}",
        "m",
        "n",
        "min(m,n)",
        "cbrt_work",
        "iters_s",
        "iters_p",
        "Sequential",
        "Parallel(0)",
        "ratio(P/S)"
    );
    eprintln!("{}", "-".repeat(99));

    for &(m, n) in grid {
        let mat = random_rect(m, n, 42);
        let k = m.min(n);
        let work = cbrt_work(m, n);

        // Scale target duration by work proxy, not by side length — a
        // 2048x256 case carries ~8x the work of 256x256 and deserves a
        // proportionally longer measurement window.
        let target = if work >= 256 {
            Duration::from_millis(500)
        } else {
            Duration::from_millis(150)
        };

        let (t_seq, iters_seq) = measure(target, || op(&mat, ExecPolicy::Sequential));
        let (t_par, iters_par) = measure(target, || op(&mat, ExecPolicy::Parallel(0)));

        let ratio = t_par.as_secs_f64() / t_seq.as_secs_f64();
        eprintln!(
            "{:>5} {:>5} {:>10} {:>10} {:>8} {:>8} {:>14.3?} {:>14.3?} {:>10.3}x",
            m, n, k, work, iters_seq, iters_par, t_seq, t_par, ratio
        );
    }
}

fn main() {
    let backend = NativeBackend::new();

    // Tall grid: (m, n) with m >= n. Each n-block spans aspect ratios
    // {1, 2, 4, 8} so rows within a block share min(m,n) but vary in
    // cbrt_work — the comparison axis for sub-issue D.
    let grid: &[(usize, usize)] = &[
        (64, 64),
        (128, 64),
        (256, 64),
        (512, 64),
        (128, 128),
        (256, 128),
        (512, 128),
        (1024, 128),
        (256, 256),
        (512, 256),
        (1024, 256),
        (2048, 256),
    ];

    run_sweep_rect("SVD (thin)", grid, |mat, policy| {
        let _ = svd_with_policy(&backend, mat, 1, policy).unwrap();
    });
    run_sweep_rect("QR", grid, |mat, policy| {
        let _ = qr_with_policy(&backend, mat, 1, policy).unwrap();
    });
    run_sweep_rect("LQ", grid, |mat, policy| {
        let _ = lq_with_policy(&backend, mat, 1, policy).unwrap();
    });

    eprintln!(
        "\nWithin each min(m,n) block, compare ratio(P/S) across rows.\n\
         If ratio tracks cbrt_work (rows within a block disagree), sub-issue D\n\
         should switch par_for_* to the cbrt(m*n*min(m,n)) proxy.\n\
         If ratio tracks min(m,n) (rows within a block agree), the current\n\
         proxy is adequate and only the threshold value needs calibration."
    );
}
