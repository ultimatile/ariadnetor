//! Shared error type for the 2-site DMRG step entry points.
//!
//! Lives in its own module rather than alongside `dmrg_2site_step`
//! because the same error type is also produced by
//! `dmrg_2site_step_block_sparse` (BlockSparse path) — splitting it
//! out keeps the per-storage entry points decoupled and keeps the
//! Dense `heff.rs` under the per-file size cap as the operator + the
//! ARPACK arm grow.

use arnet_linalg::LinalgError;

#[cfg(feature = "arpack")]
use crate::krylov::ArpackError;

/// Errors raised by the 2-site DMRG step entry points
/// (`dmrg_2site_step` for the Dense path, and
/// `dmrg_2site_step_block_sparse` for the
/// BlockSparse / U(1) path; both crate-internal). Most variants are
/// produced by both; the
/// [`DmrgHeffError::QnMismatch`] variant is BlockSparse-specific
/// and only surfaces from the BlockSparse entry point's QN /
/// Direction / sector / per-site-flux pre-validation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DmrgHeffError {
    /// `site + 1` was not a valid two-site index for the chain.
    #[error("two-site index {site} (with {site}+1) out of range for chain of length {n_sites}")]
    InvalidSite {
        /// The requested left site of the two-site block.
        site: usize,
        /// Chain length the index was checked against.
        n_sites: usize,
    },
    /// The env slot required for the two-site step (`left(site)` for
    /// the left side, `right(site + 2)` for the right side) was
    /// `None`. Indicates the caller has not built / advanced the
    /// envs into a state where this `site` can be optimized.
    #[error(
        "{side} env at index {index} is stale (None); build / advance envs into the right \
         state before stepping"
    )]
    StaleEnv {
        /// Which side's env is stale (`"left"` / `"right"`).
        side: &'static str,
        /// Index of the stale (`None`) env slot.
        index: usize,
    },
    /// MPS and MPO chain lengths disagree, or disagree with the
    /// envs the function was given.
    #[error("chain length mismatch: mps = {mps}, mpo = {mpo}, envs = {envs}")]
    LengthMismatch {
        /// Site count reported by the MPS.
        mps: usize,
        /// Site count reported by the MPO.
        mpo: usize,
        /// Site count reported by the environments.
        envs: usize,
    },
    /// The selected eigensolver's params (Lanczos or, behind the
    /// `arpack` feature, ARPACK) violated their preconditions.
    /// Surfaced here so callers see a fallible `Result` instead of
    /// the underlying solver panic / upstream error.
    #[error("invalid local-eigensolver params: {detail}")]
    InvalidEigensolverParams {
        /// Human-readable description of the violated precondition.
        detail: &'static str,
    },
    /// A bond / physical dimension on one of the inputs to the
    /// 2-site step did not match the expectation derived from the
    /// surrounding tensors. Surfaced *before* the matvec runs so
    /// the operator's `.expect` calls can stay infallible. `field`
    /// names the constraint that failed (e.g.,
    /// `"left.bot_ket vs mps[i].left_bond"`).
    #[error("shape mismatch at site {site}, {field}: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Left site of the two-site block being stepped.
        site: usize,
        /// Names the dimension constraint that failed.
        field: &'static str,
        /// Extent derived from the surrounding tensors.
        expected: usize,
        /// Extent actually found on the input.
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
    #[error("QN mismatch at site {site}, {field}: {detail}")]
    QnMismatch {
        /// Left site of the two-site block being stepped.
        site: usize,
        /// Names the offending leg pair (or single leg).
        field: &'static str,
        /// Human-readable summary of the incompatible QN data on each side.
        detail: String,
    },
    /// The layout `MemoryOrder` of one of the BlockSparse 2-site
    /// step's four contracted operands diverged from the host
    /// substrate's `preferred_order()`. Surfaced by
    /// `EffectiveHamiltonian2SiteBlockSparse::new`
    /// before any contract runs so the `apply` body's `.expect`
    /// calls cannot fire on a mixed-order operand set. `operand`
    /// names which of the four contracted operands (`"left_env"`,
    /// `"w_i"`, `"w_ip1"`, `"right_env"`) carried a non-matching
    /// layout order, and `detail` carries a human-readable summary
    /// of the offending vs expected layout order. The MPS sites
    /// passed to `new` are template-derivation-only and not asserted
    /// here; the psi template they derive is built in host order, so
    /// the matvec stays self-consistent. `detail` holds a rendered
    /// `MemoryOrder` to keep that layout type off the public error
    /// surface (it is workspace-internal).
    #[error("BlockSparse heff operand `{operand}` has layout order mismatch: {detail}")]
    OrderMismatch {
        /// Which of the four contracted operands (`"left_env"`, `"w_i"`,
        /// `"w_ip1"`, `"right_env"`) carried a non-matching layout order.
        operand: &'static str,
        /// Human-readable summary of the offending vs expected layout order.
        detail: String,
    },
    /// An underlying `arnet` linalg call (currently the truncated
    /// SVD) failed. The matvec body itself is shape-validated up
    /// front and never reaches this branch.
    #[error("linalg failure during two-site DMRG step")]
    Contract(#[from] LinalgError),
    /// The native Lanczos local eigensolver produced a non-finite
    /// result (NaN/Inf eigenpair). Forwarded without information loss
    /// from [`crate::krylov::LanczosError`]. Always present, since
    /// Lanczos is the default eigensolver and is not feature-gated.
    #[error("Lanczos failure during two-site DMRG step")]
    Lanczos(#[from] crate::krylov::LanczosError),
    /// The ARPACK-backed local eigensolver returned an upstream
    /// error (parameter validation, ARPACK info codes, max-iter
    /// without convergence, …). Forwarded without information loss
    /// from [`crate::krylov::ArpackError`]. Only present when the
    /// `arpack` feature is enabled.
    #[cfg(feature = "arpack")]
    #[error("ARPACK failure during two-site DMRG step")]
    Arpack(#[from] ArpackError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krylov::LanczosError;

    // The `?` on the Lanczos arm of the DMRG step relies on
    // `From<LanczosError>` routing to the `Lanczos` variant with the payload
    // intact. Pin it so the conversion cannot silently decay to a different
    // variant or drop a diagnostic field.
    #[test]
    fn from_lanczos_error_preserves_payload_in_lanczos_variant() {
        let err: DmrgHeffError = LanczosError::NonFinite {
            iters: 7,
            eigenvalue: f64::NAN,
            residual: f64::INFINITY,
        }
        .into();
        match err {
            DmrgHeffError::Lanczos(LanczosError::NonFinite {
                iters,
                eigenvalue,
                residual,
            }) => {
                assert_eq!(iters, 7);
                assert!(eigenvalue.is_nan());
                assert_eq!(residual, f64::INFINITY);
            }
            other => panic!("expected DmrgHeffError::Lanczos(NonFinite), got {other:?}"),
        }
    }
}
