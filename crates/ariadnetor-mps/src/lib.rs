//! Matrix Product State (MPS) and Matrix Product Operator (MPO) crate.
//!
//! Provides data structures and operations for tensor chains
//! used in tensor network algorithms (DMRG, TDVP, TEBD, etc.).
//!
//! Sits between the low-level tensor / linalg crates and algorithm crates
//! so that the latter can depend on the middle layer without consuming the
//! umbrella `arnet` crate.
//!
//! # Index convention
//!
//! - **MPS**: `(χ_L, d, χ_R)` — left-bond, physical, right-bond
//! - **MPO**: `(χ_L, d_ket, d_bra, χ_R)` — left-bond, ket, bra, right-bond
//!
//! # API forms
//!
//! Bare names ([`Mps`], [`Mpo`], [`MpsOps`], [`TensorChain`],
//! [`canonicalize`], [`inner`], …) hold chain sites as
//! [`TensorData<St, L>`](arnet_tensor::TensorData) parameterized over
//! a [`Storage`](arnet_tensor::Storage) / [`TensorLayout`](arnet_tensor::TensorLayout)
//! pair.
//!
//! `*Repr` / `*_repr` names ([`MpsRepr`], [`MpoRepr`],
//! [`MpsOpsRepr`], [`TensorChainRepr`], [`canonicalize_repr`],
//! [`inner_repr`], …) hold sites as
//! [`Dense<T>`](arnet_tensor::Dense) / [`BlockSparse<T, S>`](arnet_tensor::BlockSparse)
//! and keep the same algorithm semantics; bodies on the bare names
//! delegate to these via per-site `Arc` moves.

mod apply;
mod apply_data;
mod canonicalize;
mod chain;
mod dispatch;
mod inner;
mod site_ops;
mod truncate;
mod truncate_data;
mod types;

// Dispatch traits — enable generic algorithms over Dense / BlockSparse.
pub use dispatch::{MpsOps, MpsOpsRepr};

// Unified free functions (dispatch via MpsOps* trait).
pub use dispatch::{apply, apply_with_method, braket, canonicalize, inner, norm, truncate};
pub use dispatch::{
    apply_repr, apply_with_method_repr, braket_repr, canonicalize_repr, inner_repr, norm_repr,
    truncate_repr,
};

pub use chain::{TensorChain, TensorChainRepr};
pub use site_ops::{Qubit, SiteOps, SpinHalf};
pub use types::{
    ApplyMethod, CanonicalForm, Mpo, MpoRepr, Mps, MpsRepr, SvdAbsorb, TruncResult, TruncateParams,
};

// Re-export TruncSvdParams for convenience.
pub use arnet_linalg::TruncSvdParams;
