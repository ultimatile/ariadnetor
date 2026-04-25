//! Hardware-aware parallelism threshold tables for `NativeBackend`.
//!
//! A `ThresholdTable` stores the minimum problem-size key at which each
//! linear-algebra op is worth running in parallel on this machine. The
//! sentinel `usize::MAX` means "no finite parallel threshold" — either
//! the op is unmeasured on this profile, or calibration showed no
//! regime where parallel beats sequential (e.g. `ThresholdTable::laptop().transpose`).
//! `PerformanceManager` pairs a table with the comparison logic that
//! `NativeBackend::par_for_*` methods call.

use arnet_core::backend::ExecPolicy;

/// Per-op parallelism thresholds.
///
/// Each field is the smallest problem-size key at which the op should
/// dispatch as `ExecPolicy::Parallel(0)`. Keys are op-specific and
/// produced by the corresponding `NativeBackend::par_for_*` method:
/// `svd`/`qr`/`lq` and `gemm` use `cbrt(m*n*min(m,n))` and `cbrt(m*n*k)`
/// respectively, `eigh`/`eig`/`solve` use `n`, `transpose` uses total
/// element count.
///
/// `usize::MAX` marks "no finite parallel threshold": either unmeasured
/// on this profile, or a calibrated decision that parallel never wins
/// (e.g. `ThresholdTable::laptop().transpose`). `policy_by_n` treats it as "always
/// Sequential" in both cases.
#[derive(Clone, Debug)]
pub struct ThresholdTable {
    pub svd: usize,
    pub qr: usize,
    pub lq: usize,
    pub eigh: usize,
    pub eig: usize,
    pub gemm: usize,
    pub solve: usize,
    pub transpose: usize,
}

impl ThresholdTable {
    /// Thresholds calibrated for laptop-class CPUs (Apple M2 8-core).
    ///
    /// Values come from `examples/sweep_{decomp,decomp_rect,gemm,solve,
    /// transpose}_par.rs` run in a single session. `transpose` retains
    /// the `usize::MAX` sentinel — the sweep showed no regime where
    /// parallel wins on laptop at practical sizes; Rayon dispatch
    /// overhead dominates gains on this memory-bound op.
    pub fn laptop() -> Self {
        Self {
            svd: 384,
            qr: 384,
            lq: 512,
            eigh: 256,
            eig: 256,
            gemm: 192,
            solve: 768,
            transpose: usize::MAX,
        }
    }

    /// Thresholds calibrated for workstation-class CPUs (Xeon NUMA 112-core).
    ///
    /// On large-core NUMA machines parallel sync cost stays high even at
    /// moderate problem sizes, so the crossover shifts upward. Ops still
    /// marked `usize::MAX` never beat sequential at any measured size and
    /// await further calibration.
    pub fn workstation() -> Self {
        Self {
            svd: 1024,
            qr: usize::MAX,
            lq: usize::MAX,
            eigh: 1024,
            eig: usize::MAX,
            gemm: usize::MAX,
            solve: usize::MAX,
            transpose: usize::MAX,
        }
    }

    /// Pick a profile based on `std::thread::available_parallelism()`.
    ///
    /// `> 16` logical cores → `workstation`, otherwise `laptop`. If the
    /// query fails, fall back to the more conservative `laptop` profile.
    pub fn detect() -> Self {
        let n = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1);
        if n > 16 {
            Self::workstation()
        } else {
            Self::laptop()
        }
    }
}

/// Pairs a `ThresholdTable` with the comparison logic used by
/// `NativeBackend::par_for_*` to translate a problem-size key into an
/// `ExecPolicy`.
#[derive(Clone, Debug)]
pub struct PerformanceManager {
    thresholds: ThresholdTable,
}

impl PerformanceManager {
    pub fn new(thresholds: ThresholdTable) -> Self {
        Self { thresholds }
    }

    pub fn thresholds(&self) -> &ThresholdTable {
        &self.thresholds
    }

    /// Map a problem-size key to an `ExecPolicy`.
    ///
    /// Returns `Parallel(0)` iff the threshold is non-sentinel
    /// (`!= usize::MAX`) and the key meets or exceeds it; otherwise
    /// `Sequential`. The explicit `usize::MAX` check covers both
    /// "unmeasured" thresholds and calibrated-no-win sentinels (see
    /// the crate-level note on `usize::MAX` semantics) and prevents
    /// either from ever tripping Parallel, even if `n` were also
    /// `usize::MAX`.
    pub(crate) fn policy_by_n(threshold: usize, n: usize) -> ExecPolicy {
        if threshold != usize::MAX && n >= threshold {
            ExecPolicy::Parallel(0)
        } else {
            ExecPolicy::Sequential
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laptop_constants_pinned() {
        let t = ThresholdTable::laptop();
        assert_eq!(t.svd, 384);
        assert_eq!(t.qr, 384);
        assert_eq!(t.lq, 512);
        assert_eq!(t.eigh, 256);
        assert_eq!(t.eig, 256);
        assert_eq!(t.gemm, 192);
        assert_eq!(t.solve, 768);
        assert_eq!(t.transpose, usize::MAX);
    }

    #[test]
    fn workstation_constants_pinned() {
        let t = ThresholdTable::workstation();
        assert_eq!(t.svd, 1024);
        assert_eq!(t.eigh, 1024);
        assert_eq!(t.qr, usize::MAX);
        assert_eq!(t.lq, usize::MAX);
        assert_eq!(t.eig, usize::MAX);
        assert_eq!(t.gemm, usize::MAX);
        assert_eq!(t.solve, usize::MAX);
        assert_eq!(t.transpose, usize::MAX);
    }

    #[test]
    fn policy_by_n_below_threshold_is_sequential() {
        assert_eq!(
            PerformanceManager::policy_by_n(256, 255),
            ExecPolicy::Sequential
        );
    }

    #[test]
    fn policy_by_n_at_threshold_is_parallel() {
        assert_eq!(
            PerformanceManager::policy_by_n(256, 256),
            ExecPolicy::Parallel(0)
        );
    }

    #[test]
    fn policy_by_n_above_threshold_is_parallel() {
        assert_eq!(
            PerformanceManager::policy_by_n(256, 1024),
            ExecPolicy::Parallel(0)
        );
    }

    #[test]
    fn policy_by_n_sentinel_is_always_sequential() {
        assert_eq!(
            PerformanceManager::policy_by_n(usize::MAX, 0),
            ExecPolicy::Sequential
        );
        assert_eq!(
            PerformanceManager::policy_by_n(usize::MAX, usize::MAX),
            ExecPolicy::Sequential
        );
    }
}
