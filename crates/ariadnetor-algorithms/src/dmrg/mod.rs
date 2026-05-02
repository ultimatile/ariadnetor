//! DMRG (Density Matrix Renormalization Group) algorithm primitives.
//!
//! Exposes the environment-tensor data structure ([`DmrgEnvs`],
//! generic over a [`DmrgEnvOps`]-implementing storage type so the
//! same struct serves Dense and BlockSparse chains), the local
//! 2-site update step in two parallel forms — [`dmrg_2site_step`]
//! for the Dense path and [`dmrg_2site_step_block_sparse`] for the
//! BlockSparse / U(1) path (effective Hamiltonian + Lanczos drive
//! via the existing Dense Krylov solver through a flat-buffer
//! adapter + truncated-SVD split) — and a single storage-generic
//! 2-site sweep driver [`sweep_2site`] dispatched over [`DmrgOps`]
//! so one call site covers both the Dense and BlockSparse / U(1)
//! paths. The driver shares `DmrgSweepParams`, `DmrgSweepError`,
//! `DmrgResult`, `DmrgSweepRecord`, `DmrgStepRecord`, and
//! `SweepDirection`.

mod dispatch;
mod env;
mod env_block_sparse;
mod heff;
mod heff_block_sparse;
mod sweep;

pub use dispatch::{AbsorbedStep, DmrgOps};
pub use env::{DmrgEnvError, DmrgEnvOps, DmrgEnvs};
pub use heff::{DmrgHeffError, EffectiveHamiltonian2Site, TwoSiteStepResult, dmrg_2site_step};
pub use heff_block_sparse::{
    EffectiveHamiltonian2SiteBlockSparse, TwoSiteStepResultBlockSparse,
    dmrg_2site_step_block_sparse,
};
pub use sweep::{
    DmrgResult, DmrgStepRecord, DmrgSweepError, DmrgSweepParams, DmrgSweepRecord, SweepDirection,
    sweep_2site,
};
