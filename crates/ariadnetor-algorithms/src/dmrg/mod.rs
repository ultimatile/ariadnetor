//! DMRG (Density Matrix Renormalization Group) algorithm primitives.
//!
//! Exposes the environment-tensor data structure ([`DmrgEnvs`]), the
//! local 2-site update step ([`dmrg_2site_step`] — effective
//! Hamiltonian + Lanczos drive + truncated-SVD split), and the
//! 2-site sweep driver ([`dmrg_2site_sweep`]) that orchestrates
//! repeated L→R / R→L sweeps with diagnostics and convergence
//! tracking.

mod env;
mod heff;
mod sweep;

pub use env::{DmrgEnvError, DmrgEnvs};
pub use heff::{DmrgHeffError, EffectiveHamiltonian2Site, TwoSiteStepResult, dmrg_2site_step};
pub use sweep::{
    DmrgResult, DmrgStepRecord, DmrgSweepError, DmrgSweepParams, DmrgSweepRecord, SweepDirection,
    dmrg_2site_sweep,
};
