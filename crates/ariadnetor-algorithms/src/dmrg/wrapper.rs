//! High-level 2-site DMRG entry point.
//!
//! [`dmrg_2site`] hides [`DmrgEnvs`] construction and canonical-form
//! management from non-expert callers, layered on top of the
//! layout-generic low-level driver [`super::sweep_2site`]. The
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
//! `dmrg_2site` is generic over the `Mps<St, L>` chain
//! (`Mps<St, L>: super::DmrgOps<T>` + `Clone`), so the same entry point
//! covers both the Dense and BlockSparse / U(1) paths. The `Clone` bounds
//! (on the storage and layout) are required because the wrapper
//! defensively clones the input `Mps` before canonicalizing it. Both
//! concrete chains satisfy `Clone`, so the bound is met for every concrete
//! chain in the workspace. DMRG is host-pinned in the
//! CPU-only Stage B scope, so the boundary supplies the [`Host`]
//! substrate rather than an arbitrary backend.
//!
//! # Errors
//!
//! See [`DmrgError`] for the full set. Two failure modes are caught
//! by the wrapper itself before the lower layers can panic or repeat
//! the check:
//!
//! - [`DmrgError::EmptyMps`] — `ariadnetor_mps::canonicalize` asserts
//!   `center < n` and would panic on an empty chain.
//! - [`DmrgError::LengthMismatch`] — surfaced eagerly so callers see
//!   one failure mode for the same bug regardless of whether the
//!   build or the sweep would have caught it.
//!
//! Underlying [`DmrgEnvError`] (e.g. BlockSparse edge-bond validation)
//! and [`DmrgSweepError`] (param validation, local-eigensolver / SVD
//! failure, `TooFewSites`, etc.) are forwarded as
//! [`DmrgError::Env`] / [`DmrgError::Sweep`] respectively; the
//! `MpsNotRightCanonical` and downstream `LengthMismatch` variants of
//! [`DmrgSweepError`] are unreachable through the wrapper but kept
//! visible as defense-in-depth.

use ariadnetor_core::Scalar;
use ariadnetor_mps::{Mpo, Mps, MpsOps, TensorChain};
use ariadnetor_tensor::{Host, OpsFor, Storage, StorageFor, TensorLayout};

use super::dispatch::DmrgOps;
use super::env::{DmrgEnvError, DmrgEnvOps, DmrgEnvs};
use super::sweep::{DmrgResult, DmrgSweepError, DmrgSweepParams, sweep_2site};

/// Errors raised by [`dmrg_2site`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DmrgError {
    /// Input MPS had zero sites. Surfaced before the wrapper would
    /// invoke `canonicalize`, which asserts `center < n` and would
    /// otherwise panic.
    #[error("input MPS has zero sites")]
    EmptyMps,
    /// MPO and MPS chain lengths disagreed.
    #[error("chain length mismatch: mps = {mps}, mpo = {mpo}")]
    LengthMismatch {
        /// Site count reported by the MPS.
        mps: usize,
        /// Site count reported by the MPO.
        mpo: usize,
    },
    /// `DmrgEnvs::build` failed (e.g. BlockSparse edge-bond
    /// validation).
    #[error("DMRG environment build failed")]
    Env(#[from] DmrgEnvError),
    /// The underlying low-level sweep driver returned an error.
    #[error("DMRG sweep driver failed")]
    Sweep(#[from] DmrgSweepError),
}

/// Output of [`dmrg_2site`]: the diagnostic [`DmrgResult`] paired with the
/// optimized MPS, or a [`DmrgError`].
type Dmrg2SiteOutput<R, St, L> = Result<(DmrgResult<R>, Mps<St, L>), DmrgError>;

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
pub fn dmrg_2site<T, St, L>(
    mpo: &Mpo<St, L>,
    psi0: &Mps<St, L>,
    params: &DmrgSweepParams,
) -> Dmrg2SiteOutput<T::Real, St, L>
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
    St: Storage + StorageFor<L> + Clone,
    L: TensorLayout + Clone,
    Mps<St, L>: DmrgOps<T> + MpsOps<T, Storage = St, Layout = L>,
    DmrgEnvs<St, L>: DmrgEnvOps<T, Storage = St, Layout = L>,
    // Host-pinned: the host backend supplies every kernel, so it must declare
    // capability for this chain's storage (satisfied by Dense / BlockSparse).
    Host: OpsFor<St>,
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
    psi.canonicalize(Host::shared().as_ref(), 0);
    let mut envs = DmrgEnvs::<St, L>::build::<T>(&psi, mpo)?;
    let result = sweep_2site::<T, St, L>(&mut envs, &mut psi, mpo, params)?;
    Ok((result, psi))
}
