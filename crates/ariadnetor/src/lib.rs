//! ariadnetor: tensor network library in Rust.
//!
//! `ariadnetor` is the umbrella tensor library; it re-exports types and
//! functions from the layers listed below into its own namespace.
//! Each layer depends only on the layers listed earlier:
//!
//! - [`ariadnetor_core`] — backend-agnostic abstractions (`Scalar`,
//!   `ComputeBackend`, `EinsumExpr`). The `MemoryOrder` layout type
//!   is intentionally *not* re-exported: the umbrella's public API
//!   hides memory layout from end users.
//! - [`ariadnetor_native`] — `NativeBackend`: faer (+ optional hptt-rs transpose).
//! - [`ariadnetor_tensor`] — user-facing `Tensor`, `DenseTensor`,
//!   `BlockSparseTensor`, `Sector`, `QNIndex`.
//! - [`ariadnetor_linalg`] — backend-agnostic linear algebra over
//!   `&Tensor` (contract, svd, qr, eigh, expm, …).
//!
//! `ariadnetor_mps` and `ariadnetor_algorithms` are separate consumer crates that
//! depend on the leaf crates directly rather than on this umbrella; they
//! are not re-exported here.
//!
//! # Example
//!
//! ```
//! use ariadnetor::DenseTensor;
//!
//! let a = DenseTensor::<f64>::zeros(vec![2, 3]);
//! let b = DenseTensor::<f64>::zeros(vec![3, 2]);
//!
//! assert_eq!(a.shape(), &[2, 3]);
//! assert_eq!(b.shape(), &[3, 2]);
//! ```

#![deny(missing_docs)]

mod ops;

// Main types
pub use ariadnetor_tensor::{BlockSparseTensor, DenseTensor, Tensor};

// Block-sparse construction and introspection vocabulary: the index /
// sector / direction types an end user writes to build a symmetric tensor,
// plus the block coordinate / metadata types that block introspection
// returns. The storage / layout building blocks (`DenseStorage`,
// `DenseLayout`, the block-sparse counterparts, and the `Storage` /
// `StorageFor` / `TensorLayout` traits) and the joined `TensorData<St, L>`
// type are intentionally not re-exported: the umbrella hides memory layout
// and storage plumbing from end users. Crates that parameterize their own
// generic containers, define a new storage flavor, or perform cross-crate
// kernel access depend on `ariadnetor-tensor` directly rather than the umbrella.
pub use ariadnetor_tensor::{
    BlockCoord, BlockMeta, Direction, QNIndex, Sector, U1Sector, Z2Sector,
};

// Backend-capability scaffolding: the `OpsFor<St>` marker and the `Host`
// substrate alias.
pub use ariadnetor_tensor::{Host, OpsFor};

// Re-export from ariadnetor-core. `ExecPolicy` is the per-call parallelism
// knob the `expert` layer (re-exported below) takes by argument; without it on
// the umbrella an umbrella-only consumer could name `expert::permute` but not
// construct its policy argument.
pub use ariadnetor_core::{Complex, ComputeBackend, EinsumExpr, ExecPolicy, Scalar};

// High-level free functions over host-resident dense tensors (no backend).
// `add_all` is intentionally not re-exported: it is `linear_combine` with
// all-unit coefficients, so the umbrella exposes only the general form.
pub use ariadnetor_tensor::linear_combine;
// Explicit-backend dense free functions (backend supplied at the call site).
// The single-backend ergonomic surface is the `DenseHostOps` /
// `BlockSparseHostOps` extension traits re-exported below, not free functions.
pub use ops::{
    diag_with_backend, eig_with_backend, eigh_with_backend, eigvals_with_backend,
    eigvalsh_with_backend, einsum_with_backend, expm_antihermitian_with_backend,
    expm_hermitian_with_backend, expm_with_backend, inverse_with_backend, permute_with_backend,
    solve_with_backend, trace_with_backend,
};

// Tensor-keyed dispatch: the unified `svd` / `trunc_svd` / `qr` / `lq`
// (decomposition), `contract` / `tensordot`, and `diagonal_scale` free fns
// serve both Dense and BlockSparse via [`LinalgDecompose`] / [`LinalgContract`]
// / [`LinalgScale`], so one call site covers both flavors. The policy-explicit
// forms live under `expert`.
pub use ops::{
    LinalgContract, LinalgDecompose, LinalgScale, contract, diagonal_scale, lq, qr, svd, tensordot,
    trunc_svd,
};

// The block-sparse low-level free functions are intentionally not
// re-exported: they are consumer-internal API that `ariadnetor-mps` /
// `ariadnetor-algorithms` reach through a direct `ariadnetor-linalg` dependency, not
// this umbrella. Their result types, by contrast, ARE re-exported below:
// the Host-defaulting `BlockSparseHostOps` methods return them, so an
// umbrella user must be able to name them — mirroring the dense result
// aliases re-exported further down.
pub use ariadnetor_linalg::{
    BlockScalars, BlockSparseEigResult, BlockSparseEighResult, BlockSparseQrResult,
    BlockSparseSvdResult, BlockSparseTruncSvdResult,
};

// Ergonomic Host-defaulting method surface over the explicit-backend paths.
pub use ariadnetor_linalg::{BlockSparseHostOps, DenseHostOps};

// Expert layer: the per-call `ExecPolicy` escape hatch over the auto-policy
// default. Re-exported as the `ariadnetor::expert` namespace so an umbrella-only
// consumer can reach `expert::permute`, `expert::contract`,
// `expert::svd`, … — the decomposition policy variants dispatch over the
// tensor type, so `expert::svd` serves both Dense and BlockSparse.
pub use ariadnetor_linalg::expert;

// `flat_index` is intentionally not re-exported: it takes a `MemoryOrder`
// argument, so exposing it on the umbrella would reintroduce the
// memory-order leak that the rest of this surface closes. End users do not
// need memory-order-aware index math; in-tree code that does (tests) depends
// on `ariadnetor-tensor` directly.

// Linalg-level error type and SVD parameters.
pub use ariadnetor_linalg::{LinalgError, TruncSvdParams};
pub use ariadnetor_tensor::TensorError;
pub use ops::{EigResult, EighResult, LqResult, QrResult, SvdResult, TruncSvdResult};

// Re-export the native backend
pub use ariadnetor_native::NativeBackend;
