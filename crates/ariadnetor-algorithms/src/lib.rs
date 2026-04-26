//! Tensor-network algorithms built on top of Ariadnetor.
//!
//! This crate is the home for algorithms such as DMRG, TEBD, and
//! TDVP — along with the supporting solvers they need. The current
//! contents are limited to the Krylov eigensolver
//! ([`krylov::lanczos_smallest`]); MPS-aware algorithms that depend
//! on `arnet-mps` will be added here in subsequent work.

pub mod krylov;
