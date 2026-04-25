//! Boundary tests for `NativeBackend::par_for_*`.
//!
//! Each test constructs a `NativeBackend` with a `ThresholdTable` whose
//! fields are pinned to small non-sentinel values so the Sequential /
//! Parallel flip is observable through the trait surface. A separate
//! block verifies that sentinel (`usize::MAX`) thresholds always resolve
//! to Sequential regardless of input size.

use arnet_core::backend::{ComputeBackend, ExecPolicy};
use arnet_native::{NativeBackend, PerformanceManager, ThresholdTable};

/// Construct a backend whose every threshold is pinned to a known value,
/// so each `par_for_*` boundary is individually testable.
fn pinned_backend(t: ThresholdTable) -> NativeBackend {
    NativeBackend::with_perf(PerformanceManager::new(t))
}

fn all_pinned() -> ThresholdTable {
    ThresholdTable {
        svd: 10,
        qr: 10,
        lq: 10,
        eigh: 10,
        eig: 10,
        gemm: 10,
        solve: 10,
        transpose: 24,
    }
}

fn all_sentinel() -> ThresholdTable {
    ThresholdTable {
        svd: usize::MAX,
        qr: usize::MAX,
        lq: usize::MAX,
        eigh: usize::MAX,
        eig: usize::MAX,
        gemm: usize::MAX,
        solve: usize::MAX,
        transpose: usize::MAX,
    }
}

// ---- SVD / QR / LQ: key is cbrt(m*n*min(m,n)) ------------------------------

#[test]
fn par_for_svd_below_threshold_is_sequential() {
    let b = pinned_backend(all_pinned());
    // cbrt(8*8*8) = 8 < 10 → Sequential
    assert_eq!(b.par_for_svd(8, 8), ExecPolicy::Sequential);
    // Rectangular with small total work: cbrt(20*4*4) = cbrt(320) ≈ 6.84 → 6
    assert_eq!(b.par_for_svd(20, 4), ExecPolicy::Sequential);
    assert_eq!(b.par_for_svd(4, 20), ExecPolicy::Sequential);
}

#[test]
fn par_for_svd_at_threshold_is_parallel() {
    let b = pinned_backend(all_pinned());
    // cbrt(10*10*10) = 10 ≥ 10 → Parallel
    assert_eq!(b.par_for_svd(10, 10), ExecPolicy::Parallel(0));
    // Rectangular above threshold: cbrt(20*10*10) = cbrt(2000) ≈ 12.6 → 12
    assert_eq!(b.par_for_svd(10, 20), ExecPolicy::Parallel(0));
    assert_eq!(b.par_for_svd(20, 10), ExecPolicy::Parallel(0));
}

#[test]
fn par_for_qr_uses_cbrt_proxy() {
    let b = pinned_backend(all_pinned());
    // Tall-and-thin with small min but cbrt also small: below threshold.
    assert_eq!(b.par_for_qr(20, 4), ExecPolicy::Sequential);
    // Square at threshold.
    assert_eq!(b.par_for_qr(10, 10), ExecPolicy::Parallel(0));
    // Tall matrix with large total work flips Parallel even though min < 10:
    // cbrt(1000 * 9 * 9) = cbrt(81000) ≈ 43.2 → 43 ≥ 10 → Parallel.
    // Contrasts with the old min(m,n) proxy, which would stay Sequential here.
    assert_eq!(b.par_for_qr(1000, 9), ExecPolicy::Parallel(0));
}

#[test]
fn par_for_lq_uses_cbrt_proxy() {
    let b = pinned_backend(all_pinned());
    assert_eq!(b.par_for_lq(20, 4), ExecPolicy::Sequential);
    assert_eq!(b.par_for_lq(10, 10), ExecPolicy::Parallel(0));
    // Wide matrix analogue (min < 10 but cbrt ≥ 10).
    assert_eq!(b.par_for_lq(9, 1000), ExecPolicy::Parallel(0));
}

// ---- Eigh / Eig: key is n --------------------------------------------------

#[test]
fn par_for_eigh_at_threshold_flips() {
    let b = pinned_backend(all_pinned());
    assert_eq!(b.par_for_eigh(9), ExecPolicy::Sequential);
    assert_eq!(b.par_for_eigh(10), ExecPolicy::Parallel(0));
}

#[test]
fn par_for_eig_at_threshold_flips() {
    let b = pinned_backend(all_pinned());
    assert_eq!(b.par_for_eig(9), ExecPolicy::Sequential);
    assert_eq!(b.par_for_eig(10), ExecPolicy::Parallel(0));
}

// ---- GEMM: key is cbrt(m*n*k) ----------------------------------------------

#[test]
fn par_for_gemm_below_threshold_is_sequential() {
    let b = pinned_backend(all_pinned());
    // cbrt(9*9*9) = cbrt(729) = 9 < 10
    assert_eq!(b.par_for_gemm(9, 9, 9), ExecPolicy::Sequential);
}

#[test]
fn par_for_gemm_at_threshold_is_parallel() {
    let b = pinned_backend(all_pinned());
    // cbrt(10*10*10) = cbrt(1000) = 10 ≥ 10
    assert_eq!(b.par_for_gemm(10, 10, 10), ExecPolicy::Parallel(0));
}

#[test]
fn par_for_gemm_non_cubic_uses_geometric_mean() {
    let b = pinned_backend(all_pinned());
    // cbrt(4*5*50) = cbrt(1000) = 10 ≥ 10 — unbalanced dims still parallel
    // when the geometric mean crosses the threshold.
    assert_eq!(b.par_for_gemm(4, 5, 50), ExecPolicy::Parallel(0));
    // cbrt(4*5*40) = cbrt(800) ≈ 9.28 < 10 — stays sequential.
    assert_eq!(b.par_for_gemm(4, 5, 40), ExecPolicy::Sequential);
}

// ---- Solve: key is n -------------------------------------------------------

#[test]
fn par_for_solve_keys_on_n_not_nrhs() {
    let b = pinned_backend(all_pinned());
    // n=9 below threshold, regardless of nrhs.
    assert_eq!(b.par_for_solve(9, 10_000), ExecPolicy::Sequential);
    // n=10 at threshold.
    assert_eq!(b.par_for_solve(10, 1), ExecPolicy::Parallel(0));
}

// ---- Transpose: key is total element count ---------------------------------

#[test]
fn par_for_transpose_uses_total_elements() {
    let b = pinned_backend(all_pinned());
    // Threshold is 24. Shape [2, 3, 3] = 18 < 24.
    assert_eq!(b.par_for_transpose(&[2, 3, 3]), ExecPolicy::Sequential);
    // Shape [2, 3, 4] = 24 at threshold.
    assert_eq!(b.par_for_transpose(&[2, 3, 4]), ExecPolicy::Parallel(0));
    // Empty shape → product = 1 (empty product), below threshold.
    assert_eq!(b.par_for_transpose(&[]), ExecPolicy::Sequential);
}

// ---- Sentinel: usize::MAX threshold always stays Sequential ----------------

#[test]
fn sentinel_thresholds_never_dispatch_parallel() {
    let b = pinned_backend(all_sentinel());
    // Inputs large enough that any finite threshold would flip Parallel.
    assert_eq!(b.par_for_svd(10_000, 10_000), ExecPolicy::Sequential);
    assert_eq!(b.par_for_qr(10_000, 10_000), ExecPolicy::Sequential);
    assert_eq!(b.par_for_lq(10_000, 10_000), ExecPolicy::Sequential);
    assert_eq!(b.par_for_eigh(10_000), ExecPolicy::Sequential);
    assert_eq!(b.par_for_eig(10_000), ExecPolicy::Sequential);
    assert_eq!(
        b.par_for_gemm(10_000, 10_000, 10_000),
        ExecPolicy::Sequential
    );
    assert_eq!(b.par_for_solve(10_000, 10_000), ExecPolicy::Sequential);
    assert_eq!(
        b.par_for_transpose(&[10_000, 10_000]),
        ExecPolicy::Sequential
    );
}

// ---- Profile-level sentinel propagation ------------------------------------
//
// On the laptop profile, `transpose` retains the `usize::MAX` sentinel
// because sweep_transpose_par showed no regime where parallel wins at
// practical sizes. Other ops are calibrated and dispatch Parallel above
// their thresholds; the workstation profile still has several sentinels.

#[test]
fn laptop_profile_transpose_stays_sequential() {
    let b = NativeBackend::with_perf(PerformanceManager::new(ThresholdTable::laptop()));
    // transpose sentinel must never flip Parallel even at huge element counts.
    assert_eq!(
        b.par_for_transpose(&[10_000, 10_000]),
        ExecPolicy::Sequential
    );
}

#[test]
fn laptop_profile_calibrated_ops_dispatch_parallel_above_threshold() {
    let b = NativeBackend::with_perf(PerformanceManager::new(ThresholdTable::laptop()));
    // laptop thresholds: svd/qr=384, lq=512, eigh/eig=256, gemm=192, solve=768.
    // All keyed appropriately; large inputs should flip Parallel.
    assert_eq!(
        b.par_for_gemm(10_000, 10_000, 10_000),
        ExecPolicy::Parallel(0)
    );
    assert_eq!(b.par_for_solve(10_000, 10_000), ExecPolicy::Parallel(0));
    assert_eq!(b.par_for_svd(10_000, 10_000), ExecPolicy::Parallel(0));
    assert_eq!(b.par_for_qr(10_000, 10_000), ExecPolicy::Parallel(0));
    assert_eq!(b.par_for_lq(10_000, 10_000), ExecPolicy::Parallel(0));
    assert_eq!(b.par_for_eigh(10_000), ExecPolicy::Parallel(0));
    assert_eq!(b.par_for_eig(10_000), ExecPolicy::Parallel(0));
}

#[test]
fn workstation_profile_decomp_and_solve_stay_sequential() {
    let b = NativeBackend::with_perf(PerformanceManager::new(ThresholdTable::workstation()));
    // svd/qr/lq/eigh/eig/solve all retain usize::MAX on workstation:
    // sweep showed no crossover below n=1024 — parallel sync cost on
    // high-core NUMA dominates gains on these ops at practical sizes.
    assert_eq!(b.par_for_svd(10_000, 10_000), ExecPolicy::Sequential);
    assert_eq!(b.par_for_qr(10_000, 10_000), ExecPolicy::Sequential);
    assert_eq!(b.par_for_lq(10_000, 10_000), ExecPolicy::Sequential);
    assert_eq!(b.par_for_eigh(10_000), ExecPolicy::Sequential);
    assert_eq!(b.par_for_eig(10_000), ExecPolicy::Sequential);
    assert_eq!(b.par_for_solve(10_000, 10_000), ExecPolicy::Sequential);
}

#[test]
fn workstation_profile_gemm_and_transpose_dispatch_parallel_above_threshold() {
    let b = NativeBackend::with_perf(PerformanceManager::new(ThresholdTable::workstation()));
    // workstation thresholds: gemm=768, transpose=4_194_304 (=2048*2048).
    // Below threshold both stay Sequential; at/above they flip Parallel.
    assert_eq!(b.par_for_gemm(64, 64, 64), ExecPolicy::Sequential);
    assert_eq!(
        b.par_for_gemm(10_000, 10_000, 10_000),
        ExecPolicy::Parallel(0)
    );
    assert_eq!(b.par_for_transpose(&[1024, 1024]), ExecPolicy::Sequential);
    assert_eq!(b.par_for_transpose(&[2048, 2048]), ExecPolicy::Parallel(0));
}
