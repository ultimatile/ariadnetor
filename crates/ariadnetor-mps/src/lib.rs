//! Matrix Product State (MPS) and Matrix Product Operator (MPO) crate.
//!
//! Builds on [`ariadnetor_tensor`] for tensor storage and [`ariadnetor_linalg`]
//! for linear algebra. Consumed by `ariadnetor_algorithms`.
//!
//! # Index convention
//!
//! - **MPS**: `(χ_L, d, χ_R)` — left-bond, physical, right-bond
//! - **MPO**: `(χ_L, d_ket, d_bra, χ_R)` — left-bond, ket, bra, right-bond

#![deny(missing_docs)]

mod absorb;
mod apply;
mod canonicalize;
mod chain;
mod dispatch;
mod inner;
mod site_ops;
mod truncate;
mod types;

// Sealed chain-keyed dispatch trait — enables generic algorithms over the
// Dense / BlockSparse `Mps` chains. Sealed (not downstream-implementable).
pub use dispatch::MpsOps;

// Multi-arg free functions (dispatch via the MpsOps trait). The single-chain
// operations `canonicalize` / `truncate` / `norm` are inherent methods on
// `Mps` rather than free functions.
pub use dispatch::{apply, apply_with_method, braket, inner};

pub use chain::TensorChain;
pub use site_ops::{Qubit, SiteOps, SpinHalf};
pub use types::{ApplyMethod, CanonicalForm, Mpo, Mps, SvdAbsorb, TruncResult, TruncateParams};

// Re-export TruncSvdParams for convenience
pub use ariadnetor_linalg::TruncSvdParams;
