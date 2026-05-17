//! Storage-generic dispatch for the 2-site DMRG sweep driver.
//!
//! [`DmrgOps`] is the per-storage trait that lets [`super::sweep::sweep_2site`]
//! be written once over both `Dense<T>` and `BlockSparse<T, S>`. It mirrors
//! the [`MpsOps`] pattern in `arnet-mps`: each implementation is a thin
//! delegation to the existing storage-specific free functions, with no
//! logic duplication.
//!
//! The trait surface captures the only four storage-specific divergences
//! between the two former driver bodies:
//!
//! 1. Inner step call — `dmrg_2site_step` vs `dmrg_2site_step_block_sparse`
//!    (covered by [`DmrgOps::step`]).
//! 2. S-absorb call — `diagonal_scale` vs `diagonal_scale_block_sparse`
//!    (covered by [`DmrgOps::commit_step`]).
//! 3. Bond-dimension extraction from `s` — `shape()[0]` vs sector-summed
//!    `values.iter().map(|(_, v)| v.len()).sum()` (covered by
//!    [`DmrgOps::commit_step`]).
//! 4. The type of `result.s` itself — covered by the [`DmrgOps::StepResult`]
//!    associated type.
//!
//! Everything else in the sweep body — validation, canonical-form gate,
//! env advance, post-sweep diagnostics, convergence test — uses only
//! [`MpsOps`] dispatch and the storage-generic [`super::env::DmrgEnvs`] API,
//! so it lifts to the unified `sweep_2site<R: DmrgOps, B>` directly.
//!
//! # Sector preservation (BlockSparse path)
//!
//! The BlockSparse step returns `U` with identity flux and `Vt` with
//! `psi_flux = mps[i].flux ⊕ mps[i+1].flux`. The S-absorb is a per-block
//! real scale that does not touch flux, so the chain's total flux is
//! preserved across every step regardless of sweep direction.

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::{
    LinalgError, TruncSvdParams, diagonal_scale_block_sparse,
    diagonal_scale_dense as diagonal_scale,
};
use arnet_mps::{Mpo, Mps, MpsOps};
use arnet_tensor::{BlockSparse, Dense, Sector, TensorRepr};

use super::env::{DmrgEnvOps, DmrgEnvs};
use super::heff::{TwoSiteStepResult, dmrg_2site_step};
use super::heff_block_sparse::{TwoSiteStepResultBlockSparse, dmrg_2site_step_block_sparse};
use super::heff_error::DmrgHeffError;
use super::solver::{DmrgScalar, LocalEigensolverParams};
use super::sweep::SweepDirection;

/// Per-step output projected to scalar diagnostics + post-S-absorb site
/// storages + new bond dimension.
///
/// Returned by [`DmrgOps::commit_step`]. The two `R`-typed fields
/// (`site_i`, `site_ip1`) are the site tensors the caller writes back
/// into the MPS at indices `(site, site + 1)`.
pub struct AbsorbedStep<R: TensorRepr> {
    /// Post-absorb storage to write into MPS site `i`.
    pub site_i: R,
    /// Post-absorb storage to write into MPS site `i + 1`.
    pub site_ip1: R,
    /// Bond dimension of the new shared bond between sites `i` and
    /// `i + 1`. For BlockSparse / U(1), this is the total over all
    /// retained sectors.
    pub bond_dim: usize,
    /// Smallest eigenvalue of `H_eff` at this step (pre-truncation).
    pub eigenvalue: <R::Elem as Scalar>::Real,
    /// Local-eigensolver true residual `‖H v − λ v‖₂`.
    pub residual: <R::Elem as Scalar>::Real,
    /// Frobenius norm of the discarded singular values.
    pub trunc_err: <R::Elem as Scalar>::Real,
    /// Number of iterations the local eigensolver executed.
    pub iters: usize,
    /// `true` iff the local eigensolver succeeded — Lanczos by its
    /// absolute true-residual test against `LanczosParams::tol`,
    /// ARPACK by its relative-tol stopping criterion (i.e. `Ok`
    /// return from `arpack_smallest`). The two arms intentionally
    /// disagree on what they call "converged": Lanczos uses the
    /// absolute residual; ARPACK uses `residual <= tol * |lambda|`.
    /// See [`super::heff::TwoSiteStepResult::converged`] for the
    /// upstream contract this field forwards from.
    pub converged: bool,
}

/// Per-storage dispatch trait for the 2-site DMRG sweep driver.
///
/// Extends [`MpsOps`] so the sweep body can call `braket` / `norm` for
/// post-sweep diagnostics, and [`DmrgEnvOps`] so `DmrgEnvs<Self, B>` is
/// well-formed. Adds the two ops that vary per storage: the local solve
/// ([`step`](DmrgOps::step)) and the S-absorb-and-split
/// ([`commit_step`](DmrgOps::commit_step)).
pub trait DmrgOps: MpsOps + DmrgEnvOps + Sized {
    /// Storage-typed step result.
    ///
    /// `Dense<T>` → [`TwoSiteStepResult<T>`],
    /// `BlockSparse<T, S>` → [`TwoSiteStepResultBlockSparse<T, S>`].
    type StepResult;

    /// Build local `H_eff` at `(site, site + 1)`, drive the chosen
    /// local eigensolver (Lanczos or, behind the `arpack` feature,
    /// ARPACK) to its smallest eigenpair, and return the
    /// truncated-SVD split (U, S, Vt) of the optimized two-site block
    /// plus diagnostic scalars. Does not mutate `mps` or `envs`.
    fn step<B: ComputeBackend>(
        envs: &DmrgEnvs<Self, B>,
        mps: &Mps<Self, B>,
        mpo: &Mpo<Self, B>,
        site: usize,
        eigensolver: &LocalEigensolverParams,
        trunc: &TruncSvdParams,
    ) -> Result<Self::StepResult, DmrgHeffError>;

    /// Consume the step result, absorb `S` into `U` (R→L) or `Vt` (L→R)
    /// according to `direction`, and project diagnostic scalars +
    /// `bond_dim` into [`AbsorbedStep`]. The caller writes
    /// `out.site_i` / `out.site_ip1` to `mps` at `(site, site + 1)`.
    fn commit_step<B: ComputeBackend>(
        backend: &B,
        result: Self::StepResult,
        direction: SweepDirection,
    ) -> Result<AbsorbedStep<Self>, LinalgError>;
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T: DmrgScalar> DmrgOps for Dense<T>
where
    T::Real: Scalar<Real = T::Real>,
{
    type StepResult = TwoSiteStepResult<T>;

    fn step<B: ComputeBackend>(
        envs: &DmrgEnvs<Self, B>,
        mps: &Mps<Self, B>,
        mpo: &Mpo<Self, B>,
        site: usize,
        eigensolver: &LocalEigensolverParams,
        trunc: &TruncSvdParams,
    ) -> Result<Self::StepResult, DmrgHeffError> {
        dmrg_2site_step(envs, mps, mpo, site, eigensolver, trunc)
    }

    fn commit_step<B: ComputeBackend>(
        backend: &B,
        result: Self::StepResult,
        direction: SweepDirection,
    ) -> Result<AbsorbedStep<Self>, LinalgError> {
        let bond_dim = result.s.shape()[0];
        let (site_i, site_ip1) = match direction {
            SweepDirection::LeftToRight => {
                // site i ← U (left-isometric)
                // site i+1 ← S·Vt (axis 0 = new bond, carries S right)
                let s_vt = diagonal_scale(backend, &result.vt, result.s.data(), 0)?;
                (result.u, s_vt)
            }
            SweepDirection::RightToLeft => {
                // site i   ← U·S  (axis 2 = new bond, carries S left)
                // site i+1 ← Vt   (right-isometric)
                let u_s = diagonal_scale(backend, &result.u, result.s.data(), 2)?;
                (u_s, result.vt)
            }
        };
        Ok(AbsorbedStep {
            site_i,
            site_ip1,
            bond_dim,
            eigenvalue: result.eigenvalue,
            residual: result.residual,
            trunc_err: result.trunc_err,
            iters: result.iters,
            converged: result.converged,
        })
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T: DmrgScalar, S: Sector> DmrgOps for BlockSparse<T, S>
where
    T::Real: Scalar<Real = T::Real>,
{
    type StepResult = TwoSiteStepResultBlockSparse<T, S>;

    fn step<B: ComputeBackend>(
        envs: &DmrgEnvs<Self, B>,
        mps: &Mps<Self, B>,
        mpo: &Mpo<Self, B>,
        site: usize,
        eigensolver: &LocalEigensolverParams,
        trunc: &TruncSvdParams,
    ) -> Result<Self::StepResult, DmrgHeffError> {
        dmrg_2site_step_block_sparse(envs, mps, mpo, site, eigensolver, trunc)
    }

    fn commit_step<B: ComputeBackend>(
        backend: &B,
        result: Self::StepResult,
        direction: SweepDirection,
    ) -> Result<AbsorbedStep<Self>, LinalgError> {
        // Total post-truncation singular values across sectors — the
        // conventional U(1) MPS bond dimension.
        let bond_dim: usize = result.s.values.iter().map(|(_, v)| v.len()).sum();
        let (site_i, site_ip1) = match direction {
            SweepDirection::LeftToRight => {
                // site i ← U (left-isometric per fused sector)
                // site i+1 ← S·Vt (axis 0 = bond(Out), carries S right)
                let s_vt = diagonal_scale_block_sparse(backend, &result.vt, &result.s, 0)?;
                (result.u, s_vt)
            }
            SweepDirection::RightToLeft => {
                // site i   ← U·S  (axis 2 = bond(In), carries S left)
                // site i+1 ← Vt   (right-isometric per fused sector)
                let u_s = diagonal_scale_block_sparse(backend, &result.u, &result.s, 2)?;
                (u_s, result.vt)
            }
        };
        Ok(AbsorbedStep {
            site_i,
            site_ip1,
            bond_dim,
            eigenvalue: result.eigenvalue,
            residual: result.residual,
            trunc_err: result.trunc_err,
            iters: result.iters,
            converged: result.converged,
        })
    }
}
