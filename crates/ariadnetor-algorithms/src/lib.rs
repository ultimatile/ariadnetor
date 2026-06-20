//! Tensor-network algorithms. Currently provides DMRG and the Krylov
//! solvers it relies on.
//!
//! Builds on [`arnet_tensor`] for tensor operations and [`arnet_mps`] for
//! MPS / MPO data structures.

#![deny(missing_docs)]

pub mod dmrg;
pub mod krylov;

pub(crate) mod numeric;
