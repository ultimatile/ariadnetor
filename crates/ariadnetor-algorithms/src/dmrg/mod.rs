//! DMRG (Density Matrix Renormalization Group) algorithm primitives.
//!
//! Currently exposes the environment-tensor data structure and its
//! incremental update operations. Effective-Hamiltonian construction
//! and the sweep driver land in subsequent phases.

mod env;

pub use env::{DmrgEnvError, DmrgEnvs};
