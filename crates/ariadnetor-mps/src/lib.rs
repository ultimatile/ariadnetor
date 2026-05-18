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
//! # API
//!
//! [`Mps`] / [`Mpo`] / [`MpsOps`] / [`TensorChain`] /
//! [`canonicalize`] / [`inner`] / … hold chain sites as
//! [`TensorData<St, L>`](arnet_tensor::TensorData) parameterized over
//! a [`Storage`](arnet_tensor::Storage) / [`TensorLayout`](arnet_tensor::TensorLayout)
//! pair.

mod apply;
mod canonicalize;
mod chain;
mod dispatch;
mod inner;
mod site_ops;
mod truncate;
mod types;

// Dispatch trait — enables generic algorithms over DenseStorage / BlockSparseStorage.
pub use dispatch::MpsOps;

// Unified free functions (dispatch via MpsOps trait).
pub use dispatch::{apply, apply_with_method, braket, canonicalize, inner, norm, truncate};

pub use chain::TensorChain;
pub use site_ops::{Qubit, SiteOps, SpinHalf};
pub use types::{ApplyMethod, CanonicalForm, Mpo, Mps, SvdAbsorb, TruncResult, TruncateParams};

// Re-export TruncSvdParams for convenience.
pub use arnet_linalg::TruncSvdParams;
