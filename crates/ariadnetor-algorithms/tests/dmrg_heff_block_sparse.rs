//! BlockSparse 2-site DMRG local-update integration tests.
//!
//! Coverage:
//! - matvec correctness vs Dense oracle (densify-and-compare),
//!   on both an n=2 fixture (envs are pure boundary) and an n=3
//!   fixture at site=0 where `envs.right(2)` is extended through
//!   `MPS[2] + W[2]`, exercising the `extend_right_step` path
//! - eigenvalue oracle via direct `eigh` on the BlockSparse-flat
//!   matvec matrix with Hermiticity precondition
//! - U / Vt canonical form per fused sector
//! - non-identity 2-site flux propagation
//! - n=2 edge case
//! - error paths (`QnMismatch` and others)
//! - complex `BlockSparseTensorData<Complex<f64>, U1Sector>` coverage
//!
//! Test fixtures use the XY hopping interaction
//! `H = J (S+_a S-_{a+1} + S-_a S+_{a+1})`. See the `fixtures`
//! module of the `test-utils` crate for the concrete chain
//! constructions; matvec / step / canonical / flux tests live in
//! `matvec.rs` and error / complex tests live in `errors.rs`.

use test_utils::{fixtures, helpers};

#[path = "dmrg_heff_block_sparse/matvec.rs"]
mod matvec;

#[path = "dmrg_heff_block_sparse/errors.rs"]
mod errors;
