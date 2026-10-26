//! Tensor-network algorithms. Currently provides DMRG with the Krylov
//! solvers it relies on, and TEBD time evolution.
//!
//! Builds on [`ariadnetor_tensor`] for tensor operations and [`ariadnetor_mps`] for
//! MPS / MPO data structures.

#![deny(missing_docs)]

pub mod dmrg;
pub mod krylov;
pub mod tebd;
mod trunc_params;
