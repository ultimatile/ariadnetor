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
/// (e.g. `ThresholdTable::laptop().transpose`). `policy_by_n` treats it
/// as `ExecPolicy::Sequential` in both cases.
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
    /// Values come from `crates/ariadnetor-linalg/benches/sweep_{decomp,
    /// decomp_rect,gemm,solve,transpose}_par.rs` run in a single session.
    ///
    /// `transpose` is calibrated per backend at compile time. Under the
    /// `hptt` feature the sweep showed no regime where Rayon-style
    /// parallel can beat HPTT's tiled sequential on laptop, so the
    /// sentinel `usize::MAX` is retained. Under `--no-default-features`
    /// the naive fallback's simpler sequential loses to the parallel
    /// kernel above ~65k total elements.
    pub fn laptop() -> Self {
        Self {
            svd: 384,
            qr: 384,
            lq: 512,
            eigh: 256,
            eig: 256,
            gemm: 192,
            solve: 768,
            transpose: if cfg!(feature = "hptt") {
                usize::MAX
            } else {
                65_536
            },
        }
    }

    /// Thresholds calibrated for workstation-class CPUs (Xeon NUMA, 112 cores).
    ///
    /// Calibrated with the same five sweeps listed for `laptop()`. Most
    /// ops carry the `usize::MAX` sentinel: at workstation scale parallel
    /// sync cost is high enough that `svd`/`qr`/`lq`/`eigh`/`eig`/`solve`
    /// never beat sequential at any `n ≤ 1024` tested. Only large GEMMs
    /// (`cbrt(m*n*k) ≥ 768`) and transposes benefit from parallel
    /// dispatch.
    ///
    /// `transpose` is calibrated per backend at compile time. Under
    /// `hptt` the tiled kernel only crosses over at total element count
    /// ≥ 4_194_304. Under `--no-default-features` the naive fallback
    /// crosses over much earlier — its parallel kernel beats its own
    /// sequential above ~262_144 total elements. Calibration was
    /// performed on 2D `[n, n]` inputs; the dispatch key is total
    /// elements for any rank.
    pub fn workstation() -> Self {
        Self {
            svd: usize::MAX,
            qr: usize::MAX,
            lq: usize::MAX,
            eigh: usize::MAX,
            eig: usize::MAX,
            gemm: 768,
            solve: usize::MAX,
            transpose: if cfg!(feature = "hptt") {
                4_194_304
            } else {
                262_144
            },
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
        #[cfg(feature = "hptt")]
        assert_eq!(t.transpose, usize::MAX);
        #[cfg(not(feature = "hptt"))]
        assert_eq!(t.transpose, 65_536);
    }

    #[test]
    fn workstation_constants_pinned() {
        let t = ThresholdTable::workstation();
        assert_eq!(t.svd, usize::MAX);
        assert_eq!(t.qr, usize::MAX);
        assert_eq!(t.lq, usize::MAX);
        assert_eq!(t.eigh, usize::MAX);
        assert_eq!(t.eig, usize::MAX);
        assert_eq!(t.gemm, 768);
        assert_eq!(t.solve, usize::MAX);
        #[cfg(feature = "hptt")]
        assert_eq!(t.transpose, 4_194_304);
        #[cfg(not(feature = "hptt"))]
        assert_eq!(t.transpose, 262_144);
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
