//! High-level 2-site DMRG entry point.
//!
//! [`dmrg_2site`] hides [`DmrgEnvs`] construction and canonical-form
//! management from non-expert callers, layered on top of the
//! storage-generic low-level driver [`super::sweep_2site`]. The
//! low-level driver intentionally rejects MPS not in `Right` or
//! `Mixed { center: 0 }` form so that a caller-supplied env is never
//! silently invalidated by an internal canonicalize; this wrapper
//! defensively clones the input MPS, canonicalizes the clone to
//! `Mixed { center: 0 }`, builds a fresh [`DmrgEnvs`] against it, and
//! then invokes the driver. The caller's `psi0` is left untouched.
//!
//! Naming convention: `dmrg_2site` is the pure 2-site entry point.
//! The bare name `dmrg` is intentionally reserved for a future
//! mixed-strategy entry point (a single call that varies the number
//! of sites optimized per sweep, mirroring ITensorMPS.jl's `dmrg`
//! with `nsite` keyword). Pure 1-site DMRG with subspace expansion,
//! when it lands, will sit alongside as `dmrg_1site`.
//!
//! # Generic surface
//!
//! `dmrg_2site` is generic over `R: super::DmrgOps + Clone` and
//! `B: ComputeBackend + Clone`, so the same entry point covers both
//! the Dense (`R = Dense<T>`) and BlockSparse / U(1)
//! (`R = BlockSparse<T, S>`) storage paths. The `Clone` bounds are
//! required because `Mps<R, B>` is `Clone` only when both type
//! parameters are (the `#[derive(Clone)]` on `Mps` introduces an
//! implicit `B: Clone` bound even though the backend is held behind
//! `Arc`). Both storage types and `NativeBackend` satisfy `Clone`, so
//! the bound is met for every concrete storage / backend in the
//! workspace.
//!
//! # Errors
//!
//! See [`DmrgError`] for the full set. Two failure modes are caught
//! by the wrapper itself before the lower layers can panic or repeat
//! the check:
//!
//! - [`DmrgError::EmptyMps`] — `arnet_mps::canonicalize` asserts
//!   `center < n` and would panic on an empty chain.
//! - [`DmrgError::LengthMismatch`] — surfaced eagerly so callers see
//!   one failure mode for the same bug regardless of whether the
//!   build or the sweep would have caught it.
//!
//! Underlying [`DmrgEnvError`] (e.g. BlockSparse edge-bond validation)
//! and [`DmrgSweepError`] (param validation, Lanczos / SVD failure,
//! `TooFewSites`, etc.) are forwarded as
//! [`DmrgError::Env`] / [`DmrgError::Sweep`] respectively; the
//! `MpsNotRightCanonical` and downstream `LengthMismatch` variants of
//! [`DmrgSweepError`] are unreachable through the wrapper but kept
//! visible as defense-in-depth.

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_mps::{Mpo, Mps, TensorChain, canonicalize};

use super::dispatch::DmrgOps;
use super::env::{DmrgEnvError, DmrgEnvs};
use super::sweep::{DmrgResult, DmrgSweepError, DmrgSweepParams, sweep_2site};

/// Errors raised by [`dmrg_2site`].
#[derive(Debug)]
#[non_exhaustive]
pub enum DmrgError {
    /// Input MPS had zero sites. Surfaced before the wrapper would
    /// invoke `canonicalize`, which asserts `center < n` and would
    /// otherwise panic.
    EmptyMps,
    /// MPO and MPS chain lengths disagreed.
    LengthMismatch { mps: usize, mpo: usize },
    /// `DmrgEnvs::build` failed (e.g. BlockSparse edge-bond
    /// validation).
    Env(DmrgEnvError),
    /// The underlying low-level sweep driver returned an error.
    Sweep(DmrgSweepError),
}

impl std::fmt::Display for DmrgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DmrgError::EmptyMps => write!(f, "input MPS has zero sites"),
            DmrgError::LengthMismatch { mps, mpo } => {
                write!(f, "chain length mismatch: mps = {mps}, mpo = {mpo}")
            }
            DmrgError::Env(_) => write!(f, "DMRG environment build failed"),
            DmrgError::Sweep(_) => write!(f, "DMRG sweep driver failed"),
        }
    }
}

impl std::error::Error for DmrgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DmrgError::Env(e) => Some(e),
            DmrgError::Sweep(e) => Some(e),
            _ => None,
        }
    }
}

impl From<DmrgEnvError> for DmrgError {
    fn from(e: DmrgEnvError) -> Self {
        DmrgError::Env(e)
    }
}

impl From<DmrgSweepError> for DmrgError {
    fn from(e: DmrgSweepError) -> Self {
        DmrgError::Sweep(e)
    }
}

/// Run a 2-site DMRG calculation with caller-friendly defaults for
/// canonical-form management and environment construction.
///
/// The caller's `psi0` is **defensively cloned**; the input MPS is
/// not mutated regardless of outcome. The clone is canonicalized to
/// `Mixed { center: 0 }`, paired with a freshly built [`DmrgEnvs`],
/// and handed to [`super::sweep_2site`]. The optimized MPS is
/// returned alongside the diagnostic [`DmrgResult`].
///
/// The returned MPS ends in `CanonicalForm::Mixed { center: 0 }`
/// (the orthogonality center sits at site 0 because the driver runs
/// R→L last); see [`super::sweep_2site`] for details.
///
/// # Errors
///
/// Returns [`DmrgError::EmptyMps`] when `psi0.len() == 0`,
/// [`DmrgError::LengthMismatch`] when MPO and MPS lengths disagree,
/// [`DmrgError::Env`] on environment-build failure, and
/// [`DmrgError::Sweep`] on driver failure.
#[allow(clippy::type_complexity)]
pub fn dmrg_2site<R, B>(
    mpo: &Mpo<R, B>,
    psi0: &Mps<R, B>,
    params: &DmrgSweepParams,
) -> Result<(DmrgResult<<R::Elem as Scalar>::Real>, Mps<R, B>), DmrgError>
where
    R: DmrgOps + Clone,
    <R::Elem as Scalar>::Real: Scalar<Real = <R::Elem as Scalar>::Real>,
    B: ComputeBackend + Clone,
{
    if psi0.len() == 0 {
        return Err(DmrgError::EmptyMps);
    }
    if mpo.len() != psi0.len() {
        return Err(DmrgError::LengthMismatch {
            mps: psi0.len(),
            mpo: mpo.len(),
        });
    }

    let mut psi = psi0.clone();
    canonicalize(&mut psi, 0);
    let mut envs = DmrgEnvs::build(&psi, mpo)?;
    let result = sweep_2site(&mut envs, &mut psi, mpo, params)?;
    Ok((result, psi))
}
