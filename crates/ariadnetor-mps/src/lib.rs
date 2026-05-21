//! Matrix Product State (MPS) and Matrix Product Operator (MPO) crate.
//!
//! Provides data structures and operations for tensor chains
//! used in tensor network algorithms (DMRG, TDVP, TEBD, etc.).
//!
//! Sits above the umbrella `arnet` crate (which is itself the tensor
//! library after the #262 redo), so algorithm crates such as
//! `arnet-algorithms` consume `arnet-mps` directly rather than
//! re-routing through the umbrella.
//!
//! # Index convention
//!
//! - **MPS**: `(χ_L, d, χ_R)` — left-bond, physical, right-bond
//! - **MPO**: `(χ_L, d_ket, d_bra, χ_R)` — left-bond, ket, bra, right-bond

mod apply;
mod canonicalize;
mod chain;
mod dispatch;
mod inner;
mod internal_helpers;
mod site_ops;
mod truncate;
mod types;

// Dispatch trait — enables generic algorithms over Dense / BlockSparse
pub use dispatch::MpsOps;

// Unified free functions (dispatch via MpsOps trait)
pub use dispatch::{apply, apply_with_method, braket, canonicalize, inner, norm, truncate};

pub use chain::TensorChain;
pub use site_ops::{Qubit, SiteOps, SpinHalf};
pub use types::{ApplyMethod, CanonicalForm, Mpo, Mps, SvdAbsorb, TruncResult, TruncateParams};

// Re-export TruncSvdParams for convenience
pub use arnet::TruncSvdParams;
