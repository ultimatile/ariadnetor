//! Local-eigensolver selection for the 2-site DMRG step.
//!
//! [`LocalEigensolverParams`] is the runtime selector used by
//! [`super::sweep::DmrgSweepParams`] (and through it both
//! [`super::heff::dmrg_2site_step`] and
//! [`super::heff_block_sparse::dmrg_2site_step_block_sparse`]) to
//! pick which Krylov solver drives the local eigenpair extraction.
//!
//! Lanczos is always available. An ARPACK-backed variant lives behind
//! the `arpack` feature gate; its presence in the enum is itself
//! `cfg`-gated so callers cannot select it at compile time without
//! enabling the feature.
//!
//! Helper functions [`validate_eigensolver_params`] and
//! [`eigensolver_tol`] centralize the per-variant param sanity checks
//! and tolerance extraction so [`super::sweep::sweep_2site`] and the
//! heff entry points don't drift.

use arnet::Scalar;

#[cfg(feature = "arpack")]
use crate::krylov::ArpackParams;
use crate::krylov::LanczosParams;

/// Runtime-selectable local eigensolver for the 2-site DMRG step.
///
/// `Lanczos` is the default; `Arpack` requires the `arpack` Cargo
/// feature.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LocalEigensolverParams {
    /// In-tree Lanczos with full reorthogonalization
    /// ([`crate::krylov::lanczos_smallest`]).
    Lanczos(LanczosParams),
    /// ARPACK-NG-backed solver
    /// ([`crate::krylov::arpack_smallest`]). Only constructible when
    /// the `arpack` feature is enabled.
    #[cfg(feature = "arpack")]
    Arpack(ArpackParams),
}

impl Default for LocalEigensolverParams {
    fn default() -> Self {
        Self::Lanczos(LanczosParams::default())
    }
}

impl From<LanczosParams> for LocalEigensolverParams {
    fn from(p: LanczosParams) -> Self {
        Self::Lanczos(p)
    }
}

#[cfg(feature = "arpack")]
impl From<ArpackParams> for LocalEigensolverParams {
    fn from(p: ArpackParams) -> Self {
        Self::Arpack(p)
    }
}

/// Validate the per-variant solver params (max_iter, tol). The
/// `T::Real` representability check is left to the caller because it
/// depends on the storage's element type.
///
/// Returns the `&'static str` detail of the first failing constraint
/// so the caller can wrap it into either
/// [`super::sweep::DmrgSweepError::InvalidParams`] or
/// [`super::heff_error::DmrgHeffError::InvalidEigensolverParams`]
/// without duplicating the per-variant logic.
pub(crate) fn validate_eigensolver_params(
    params: &LocalEigensolverParams,
) -> Result<(), &'static str> {
    match params {
        LocalEigensolverParams::Lanczos(p) => {
            if p.max_iter == 0 {
                return Err("lanczos.max_iter must be >= 1");
            }
            if !p.tol.is_finite() {
                return Err("lanczos.tol must be finite");
            }
            if p.tol < 0.0 {
                return Err("lanczos.tol must be non-negative");
            }
            Ok(())
        }
        #[cfg(feature = "arpack")]
        LocalEigensolverParams::Arpack(p) => {
            if p.max_iter == 0 {
                return Err("arpack.max_iter must be >= 1");
            }
            if !p.tol.is_finite() {
                return Err("arpack.tol must be finite");
            }
            // ARPACK rejects tol == 0 (would request its
            // machine-epsilon default and silently break the
            // converged flag).
            if p.tol <= 0.0 {
                return Err("arpack.tol must be strictly positive");
            }
            Ok(())
        }
    }
}

/// Extract the variant's `tol` field as `f64` for cross-variant
/// downstream casts (e.g. `try_real_from_f64`).
pub(crate) fn eigensolver_tol(params: &LocalEigensolverParams) -> f64 {
    match params {
        LocalEigensolverParams::Lanczos(p) => p.tol,
        #[cfg(feature = "arpack")]
        LocalEigensolverParams::Arpack(p) => p.tol,
    }
}

/// Trait alias bundling `Scalar` with the per-feature solver bounds
/// the heff entry points need.
///
/// Without the `arpack` feature this is just `Scalar`. With the
/// feature on, it additionally requires `crate::krylov::ArpackScalar`,
/// which `arpack_smallest` demands. Both `Scalar` and `ArpackScalar`
/// are sealed to the same set of scalar types
/// (`f32`/`f64`/`Complex<f32>`/`Complex<f64>`), so toggling the
/// feature does not restrict any existing caller — every supported
/// scalar already satisfies both.
#[cfg(not(feature = "arpack"))]
pub trait DmrgScalar: Scalar {}
#[cfg(not(feature = "arpack"))]
impl<T: Scalar> DmrgScalar for T {}

#[cfg(feature = "arpack")]
pub trait DmrgScalar: Scalar + crate::krylov::ArpackScalar {}
#[cfg(feature = "arpack")]
impl<T: Scalar + crate::krylov::ArpackScalar> DmrgScalar for T {}
