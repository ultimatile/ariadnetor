//! Tensor-network algorithms built on top of Ariadnetor.
//!
//! Sits above the middle layer (`arnet-mps`) and depends on the
//! low-level crates (`arnet-core`, `arnet-tensor`, `arnet-linalg`,
//! `arnet-native`). This crate is the home for algorithms such as
//! DMRG, TEBD, TDVP — and the supporting solvers they need.

pub mod krylov;
