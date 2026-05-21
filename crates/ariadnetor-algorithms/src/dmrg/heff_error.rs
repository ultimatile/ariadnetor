//! Shared error type for the 2-site DMRG step entry points.
//!
//! Lives in its own module rather than alongside `dmrg_2site_step`
//! because the same error type is also produced by
//! `dmrg_2site_step_block_sparse` (BlockSparse path) — splitting it
//! out keeps the per-storage entry points decoupled and keeps the
//! Dense `heff.rs` under the per-file size cap as the operator + the
//! ARPACK arm grow.

use arnet::LinalgError;

#[cfg(feature = "arpack")]
use crate::krylov::ArpackError;

/// Errors raised by the 2-site DMRG step entry points
/// ([`super::heff::dmrg_2site_step`] for the Dense path, and
/// [`super::heff_block_sparse::dmrg_2site_step_block_sparse`] for the
/// BlockSparse / U(1) path). Most variants are produced by both; the
/// [`DmrgHeffError::QnMismatch`] variant is BlockSparse-specific
/// and only surfaces from the BlockSparse entry point's QN /
/// Direction / sector / per-site-flux pre-validation.
#[derive(Debug)]
#[non_exhaustive]
pub enum DmrgHeffError {
    /// `site + 1` was not a valid two-site index for the chain.
    InvalidSite { site: usize, n_sites: usize },
    /// The env slot required for the two-site step (`left(site)` for
    /// the left side, `right(site + 2)` for the right side) was
    /// `None`. Indicates the caller has not built / advanced the
    /// envs into a state where this `site` can be optimized.
    StaleEnv { side: &'static str, index: usize },
    /// MPS and MPO chain lengths disagree, or disagree with the
    /// envs the function was given.
    LengthMismatch { mps: usize, mpo: usize, envs: usize },
    /// The selected eigensolver's params (Lanczos or, behind the
    /// `arpack` feature, ARPACK) violated their preconditions.
    /// Surfaced here so callers see a fallible `Result` instead of
    /// the underlying solver panic / upstream error.
    InvalidEigensolverParams { detail: &'static str },
    /// A bond / physical dimension on one of the inputs to the
    /// 2-site step did not match the expectation derived from the
    /// surrounding tensors. Surfaced *before* the matvec runs so
    /// the operator's `.expect` calls can stay infallible. `field`
    /// names the constraint that failed (e.g.,
    /// `"left.bot_ket vs mps[i].left_bond"`).
    ShapeMismatch {
        site: usize,
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A QNIndex / Direction / sector / per-site-flux compatibility
    /// check on a BlockSparse 2-site step's inputs failed. Surfaced
    /// up front by `dmrg_2site_step_block_sparse` so the matvec
    /// body's `.expect` calls cannot fire on user input. `field`
    /// names the leg pair (or single leg for MPO well-formedness
    /// checks), and `detail` carries a human-readable summary of
    /// the offending `(sector list, direction, flux-if-applicable)`
    /// data on each side.
    QnMismatch {
        site: usize,
        field: &'static str,
        detail: String,
    },
    /// An underlying `arnet_linalg` call (currently the truncated
    /// SVD) failed. The matvec body itself is shape-validated up
    /// front and never reaches this branch.
    Contract(LinalgError),
    /// The ARPACK-backed local eigensolver returned an upstream
    /// error (parameter validation, ARPACK info codes, max-iter
    /// without convergence, …). Forwarded without information loss
    /// from [`crate::krylov::ArpackError`]. Only present when the
    /// `arpack` feature is enabled.
    #[cfg(feature = "arpack")]
    Arpack(ArpackError),
}

impl From<LinalgError> for DmrgHeffError {
    fn from(e: LinalgError) -> Self {
        DmrgHeffError::Contract(e)
    }
}

#[cfg(feature = "arpack")]
impl From<ArpackError> for DmrgHeffError {
    fn from(e: ArpackError) -> Self {
        DmrgHeffError::Arpack(e)
    }
}

impl std::fmt::Display for DmrgHeffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DmrgHeffError::InvalidSite { site, n_sites } => write!(
                f,
                "two-site index {site} (with {site}+1) out of range for chain of length {n_sites}"
            ),
            DmrgHeffError::StaleEnv { side, index } => write!(
                f,
                "{side} env at index {index} is stale (None); build / advance envs into the right \
                 state before stepping"
            ),
            DmrgHeffError::LengthMismatch { mps, mpo, envs } => write!(
                f,
                "chain length mismatch: mps = {mps}, mpo = {mpo}, envs = {envs}"
            ),
            DmrgHeffError::ShapeMismatch {
                site,
                field,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch at site {site}, {field}: expected {expected}, got {actual}"
            ),
            DmrgHeffError::InvalidEigensolverParams { detail } => {
                write!(f, "invalid local-eigensolver params: {detail}")
            }
            DmrgHeffError::QnMismatch {
                site,
                field,
                detail,
            } => write!(f, "QN mismatch at site {site}, {field}: {detail}"),
            DmrgHeffError::Contract(_) => {
                write!(f, "linalg failure during two-site DMRG step")
            }
            #[cfg(feature = "arpack")]
            DmrgHeffError::Arpack(err) => {
                write!(f, "ARPACK failure during two-site DMRG step: {err}")
            }
        }
    }
}

impl std::error::Error for DmrgHeffError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DmrgHeffError::Contract(err) => Some(err),
            #[cfg(feature = "arpack")]
            DmrgHeffError::Arpack(err) => Some(err),
            _ => None,
        }
    }
}
