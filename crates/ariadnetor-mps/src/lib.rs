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
mod env;
mod env_block_sparse;
mod inner;
mod serialize;
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

// Three-layer ⟨bra|W|ket⟩ environment primitive (Dense / BlockSparse),
// consumed by DMRG (bra = ket) and, later, variational fitting (distinct
// bra / ket). Its dispatch trait is sealed (not downstream-implementable).
pub use env::{BraketEnvError, BraketEnvOps, BraketEnvs};

pub use chain::TensorChain;
// MPS serialization primitive: lossless, deterministic save/load for restart.
pub use serialize::{
    MpsCodec, MpsIoError, MpsManifest, OrderTag, SiteMeta, load_mps, load_mps_from_path, save_mps,
    save_mps_to_path,
};
pub use site_ops::{Qubit, SiteOps, SpinHalf};
pub use types::{
    ApplyMethod, CanonicalForm, Mpo, Mps, SvdAbsorb, TruncResult, TruncateParams, VariationalInit,
};

// Re-export TruncSvdParams for convenience
pub use ariadnetor_linalg::TruncSvdParams;
