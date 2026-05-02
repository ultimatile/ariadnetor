//! DMRG (Density Matrix Renormalization Group) algorithm primitives.
//!
//! Exposes the environment-tensor data structure ([`DmrgEnvs`],
//! generic over a [`DmrgEnvOps`]-implementing storage type so the
//! same struct serves Dense and BlockSparse chains), the local
//! 2-site update step in two parallel forms — [`dmrg_2site_step`]
//! for the Dense path and [`dmrg_2site_step_block_sparse`] for the
//! BlockSparse / U(1) path (effective Hamiltonian + Lanczos drive
//! via the existing Dense Krylov solver through a flat-buffer
//! adapter + truncated-SVD split) — and the 2-site sweep driver
//! in two parallel forms — [`dmrg_2site_sweep`] for the Dense path
//! and [`dmrg_2site_sweep_block_sparse`] for the BlockSparse / U(1)
//! path. The two sweep entry points share `DmrgSweepParams`,
//! `DmrgSweepError`, `DmrgResult`, `DmrgSweepRecord`,
//! `DmrgStepRecord`, and `SweepDirection`.

mod env;
mod env_block_sparse;
mod heff;
mod heff_block_sparse;
mod sweep;
mod sweep_block_sparse;

pub use env::{DmrgEnvError, DmrgEnvOps, DmrgEnvs};
pub use heff::{DmrgHeffError, EffectiveHamiltonian2Site, TwoSiteStepResult, dmrg_2site_step};
pub use heff_block_sparse::{
    EffectiveHamiltonian2SiteBlockSparse, TwoSiteStepResultBlockSparse,
    dmrg_2site_step_block_sparse,
};
pub use sweep::{
    DmrgResult, DmrgStepRecord, DmrgSweepError, DmrgSweepParams, DmrgSweepRecord, SweepDirection,
    dmrg_2site_sweep,
};
pub use sweep_block_sparse::dmrg_2site_sweep_block_sparse;
