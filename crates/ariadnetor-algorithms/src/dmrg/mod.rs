//! DMRG (Density Matrix Renormalization Group) algorithm primitives.
//!
//! Exposes the environment-tensor data structure ([`DmrgEnvs`],
//! generic over a [`DmrgEnvOps`]-implementing layout type so the
//! same struct serves Dense and BlockSparse chains), the local
//! 2-site update step in two parallel forms — [`dmrg_2site_step`]
//! for the Dense path and [`dmrg_2site_step_block_sparse`] for the
//! BlockSparse / U(1) path (effective Hamiltonian + local-eigensolver
//! drive — Lanczos by default, ARPACK behind the `arpack` feature —
//! via the Dense Krylov family through a flat-buffer adapter on
//! the BlockSparse path, plus a truncated-SVD split) — and a single
//! layout-generic 2-site sweep driver [`sweep_2site`] dispatched
//! over [`DmrgOps`] so one call site covers both the Dense and
//! BlockSparse / U(1) paths. The driver shares `DmrgSweepParams`,
//! `DmrgSweepError`, `DmrgResult`, `DmrgSweepRecord`,
//! `DmrgStepRecord`, and `SweepDirection`.
//!
//! For non-expert callers, [`dmrg_2site`] wraps the low-level driver
//! with defensive `psi0.clone()`, automatic canonicalization to
//! `Mixed { center: 0 }`, and a fresh [`DmrgEnvs::build`], reporting
//! its own [`DmrgError`] that subsumes empty / length-mismatch input
//! and forwards underlying env / sweep failures.

mod dispatch;
mod env;
mod env_block_sparse;
mod heff;
mod heff_block_sparse;
mod heff_error;
mod sealed;
mod solver;
mod sweep;
mod wrapper;

pub use dispatch::{AbsorbedStep, DmrgOps, FullStepError};
pub use env::{DmrgEnvError, DmrgEnvOps, DmrgEnvs};
pub use heff::{EffectiveHamiltonian2Site, TwoSiteStepResult, dmrg_2site_step};
pub use heff_block_sparse::{
    EffectiveHamiltonian2SiteBlockSparse, TwoSiteStepResultBlockSparse,
    dmrg_2site_step_block_sparse,
};
pub use heff_error::DmrgHeffError;
pub use solver::LocalEigensolverParams;
pub use sweep::{
    DmrgResult, DmrgStepRecord, DmrgSweepError, DmrgSweepParams, DmrgSweepRecord, SweepDirection,
    sweep_2site,
};
pub use wrapper::{DmrgError, dmrg_2site};
