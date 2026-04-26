//! Krylov-subspace eigensolvers.
//!
//! Currently exposes a single entry point, [`lanczos::lanczos_smallest`],
//! which finds the smallest eigenvalue and corresponding eigenvector of
//! a Hermitian linear operator using the Lanczos iteration with full
//! reorthogonalization. The operator is supplied through the
//! [`LinearOp`] trait, which is the only contract callers need to
//! implement.
//!
//! Additional solvers (Arnoldi, LOBPCG, deflation/restart variants) are
//! deferred. The current scope is what DMRG's local effective-Hamiltonian
//! problem requires.

mod lanczos;

pub use lanczos::{LanczosParams, LanczosResult, LinearOp, lanczos_smallest};
