//! Matrix Product State (MPS) and Matrix Product Operator (MPO) module.
//!
//! Provides data structures and operations for tensor chains
//! used in tensor network algorithms (DMRG, TDVP, TEBD, etc.).
//!
//! # Index convention
//!
//! - **MPS**: `(χ_L, d, χ_R)` — left-bond, physical, right-bond
//! - **MPO**: `(χ_L, d_ket, d_bra, χ_R)` — left-bond, ket, bra, right-bond

mod chain;
mod types;

pub use chain::TensorChain;
pub use types::{CanonicalForm, Mpo, Mps};
