//! DMRG (Density Matrix Renormalization Group) algorithm primitives.
//!
//! Builds on the [`ariadnetor_mps::BraketEnvs`] environment primitive
//! (the three-layer ⟨bra|W|ket⟩ tensor chain, consumed here with
//! bra = ket) and exposes a single
//! layout-generic 2-site sweep driver [`sweep_2site`] dispatched
//! over [`DmrgOps`] so one call site covers both the Dense and
//! BlockSparse / U(1) paths. Internally the driver runs a local
//! 2-site update step in two parallel forms (Dense and BlockSparse /
//! U(1)) — effective Hamiltonian + local-eigensolver drive (Lanczos
//! by default, ARPACK behind the `arpack` feature; the BlockSparse
//! path reuses the Dense Krylov family through a flat-buffer adapter)
//! plus a truncated-SVD split — but these per-step primitives are
//! crate-internal, not part of the exposed surface. The driver shares
//! `DmrgSweepParams`, `DmrgSweepError`, `DmrgResult`,
//! `DmrgSweepRecord`, `DmrgStepRecord`, and `SweepDirection`.
//!
//! For non-expert callers, [`dmrg_2site`] wraps the low-level driver
//! with defensive `psi0.clone()`, automatic canonicalization to
//! `Mixed { center: 0 }`, and a fresh [`ariadnetor_mps::BraketEnvs::build`], reporting
//! its own [`DmrgError`] that subsumes empty / length-mismatch input
//! and forwards underlying env / sweep failures.

mod dispatch;
mod heff;
mod heff_block_sparse;
mod heff_error;
mod solver;
mod sweep;
mod wrapper;

pub use dispatch::{AbsorbedStep, DmrgOps, FullStepError};
pub use heff::TwoSiteStepResult;
pub use heff_block_sparse::TwoSiteStepResultBlockSparse;
pub use heff_error::DmrgHeffError;
pub use solver::LocalEigensolverParams;
pub use sweep::{
    DmrgResult, DmrgStepRecord, DmrgSweepError, DmrgSweepParams, DmrgSweepRecord, SweepDirection,
    sweep_2site,
};
pub use wrapper::{DmrgError, dmrg_2site};
