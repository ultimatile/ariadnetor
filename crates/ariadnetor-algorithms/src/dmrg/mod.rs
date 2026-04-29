//! DMRG (Density Matrix Renormalization Group) algorithm primitives.
//!
//! Currently exposes the environment-tensor data structure plus the
//! local 2-site update step (effective Hamiltonian, Lanczos drive,
//! truncated-SVD split). The sweep driver lands in a subsequent
//! phase.

mod env;
mod heff;
mod sweep;

pub use env::{DmrgEnvError, DmrgEnvs};
pub use heff::{DmrgHeffError, EffectiveHamiltonian2Site, TwoSiteStepResult, dmrg_2site_step};
pub use sweep::{
    DmrgResult, DmrgStepRecord, DmrgSweepError, DmrgSweepParams, DmrgSweepRecord, SweepDirection,
    dmrg_2site_sweep,
};
