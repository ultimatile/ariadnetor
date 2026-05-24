//! Matrix Product State (MPS) and Matrix Product Operator (MPO) crate.
//!
//! Builds on [`arnet`] for tensor storage and linalg. Consumed by
//! `arnet_algorithms`.
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

// `LayoutOrderCheck` is referenced in the `where` bounds of
// `Mps`/`Mpo` constructors. Re-exporting it at the crate root makes
// the bound reachable by downstream callers (so trait-resolution
// diagnostics name a public path), while the `#[doc(hidden)]`
// attribute on the trait definition keeps it out of rendered docs —
// users never need to think about the sealed-dispatch mechanism.
pub use types::LayoutOrderCheck;

// Re-export TruncSvdParams for convenience
pub use arnet::TruncSvdParams;
