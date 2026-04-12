//! Matrix Product State (MPS) and Matrix Product Operator (MPO) module.
//!
//! Provides data structures and operations for tensor chains
//! used in tensor network algorithms (DMRG, TDVP, TEBD, etc.).
//!
//! # Index convention
//!
//! - **MPS**: `(χ_L, d, χ_R)` — left-bond, physical, right-bond
//! - **MPO**: `(χ_L, d_ket, d_bra, χ_R)` — left-bond, ket, bra, right-bond

mod apply;
mod canonicalize;
mod chain;
mod inner;
mod site_ops;
mod truncate;
mod types;

pub use apply::apply;
pub use canonicalize::{canonicalize, canonicalize_block_sparse};
pub use chain::TensorChain;
pub use inner::{braket, inner, inner_block_sparse, norm, norm_block_sparse};
pub use site_ops::{Qubit, SiteOps, SpinHalf};
pub use truncate::{truncate, truncate_block_sparse};
pub use types::{CanonicalForm, Mpo, Mps, SvdAbsorb, TruncResult, TruncateParams};

// Re-export TruncSvdParams for convenience
pub use arnet_linalg::TruncSvdParams;
