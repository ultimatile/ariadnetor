//! Tensor-network algorithms. Currently provides DMRG and the Krylov
//! solvers it relies on.
//!
//! Builds on [`ariadnetor_tensor`] for tensor operations and [`ariadnetor_mps`] for
//! MPS / MPO data structures.

#![deny(missing_docs)]

pub mod dmrg;
pub mod krylov;
