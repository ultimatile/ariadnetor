//! Ariadnetor: tensor network framework in Rust.
//!
//! `arnet` is the umbrella tensor library; it re-exports types and
//! functions from the layers listed below into its own namespace.
//! Each layer depends only on the layers listed earlier:
//!
//! - [`arnet_core`] — backend-agnostic abstractions (`Scalar`,
//!   `ComputeBackend`, `EinsumExpr`). The `MemoryOrder` layout type
//!   is intentionally *not* re-exported: the umbrella's public API
//!   hides memory layout from end users.
//! - [`arnet_native`] — `NativeBackend`: faer + hptt-rs.
//! - [`arnet_tensor`] — user-facing `Tensor`, `DenseTensor`,
//!   `BlockSparseTensor`, `Sector`, `QNIndex`.
//! - [`arnet_linalg`] — backend-agnostic linear algebra over
//!   `&Tensor` (contract, svd, qr, eigh, expm, …).
//!
//! `arnet_mps` and `arnet_algorithms` are separate consumer crates that
//! depend on the leaf crates directly rather than on this umbrella; they
//! are not re-exported here.
//!
//! # Example
//!
//! ```
//! use arnet::DenseTensor;
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
pub use arnet_tensor::{BlockSparseTensor, DenseTensor, Tensor};

// Block-sparse construction and introspection vocabulary: the index /
// sector / direction types an end user writes to build a symmetric tensor,
// plus the block coordinate / metadata types that block introspection
// returns. The storage / layout building blocks (`DenseStorage`,
// `DenseLayout`, the block-sparse counterparts, and the `Storage` /
// `StorageFor` / `TensorLayout` traits) and the joined `TensorData<St, L>`
// type are intentionally not re-exported: the umbrella hides memory layout
// and storage plumbing from end users. Crates that parameterize their own
// generic containers, define a new storage flavor, or perform cross-crate
// kernel access depend on `arnet-tensor` directly rather than the umbrella.
pub use arnet_tensor::{BlockCoord, BlockMeta, Direction, QNIndex, Sector, U1Sector, Z2Sector};

// Backend-capability scaffolding: the `OpsFor<St>` marker and the `Host`
// substrate alias.
pub use arnet_tensor::{Host, OpsFor};

// Re-export from ariadnetor-core. `ExecPolicy` is the per-call parallelism
// knob the `expert` layer (re-exported below) takes by argument; without it on
// the umbrella an umbrella-only consumer could name `expert::permute` but not
// construct its policy argument.
pub use arnet_core::{Complex, ComputeBackend, EinsumExpr, ExecPolicy, Scalar};

// High-level free functions over host-resident dense tensors (no backend).
// `add_all` is intentionally not re-exported: it is `linear_combine` with
// all-unit coefficients, so the umbrella exposes only the general form.
pub use arnet_tensor::linear_combine;
// Explicit-backend dense free functions (backend supplied at the call site).
// The single-backend ergonomic surface is the `DenseHostOps` /
// `BlockSparseHostOps` extension traits re-exported below, not free functions.
pub use ops::{
    diag_with_backend, eig_with_backend, eigh_with_backend, eigvals_with_backend,
    eigvalsh_with_backend, einsum_with_backend, expm_antihermitian_with_backend,
    expm_hermitian_with_backend, expm_with_backend, inverse_with_backend, permute_with_backend,
    solve_with_backend, trace_with_backend,
};

// Layout-keyed dispatch: the unified `svd` / `trunc_svd` / `qr` / `lq`
// (decomposition), `contract` / `tensordot`, and `diagonal_scale` free fns
// serve both Dense and BlockSparse via [`LinalgDecompose`] / [`LinalgContract`]
// / [`LinalgScale`], so one call site covers both flavors. The policy-explicit
// forms live under `expert`.
pub use ops::{
    LinalgContract, LinalgDecompose, LinalgScale, contract, diagonal_scale, lq, qr, svd, tensordot,
    trunc_svd,
};

// The block-sparse low-level free functions are intentionally not
// re-exported: they are consumer-internal API that `arnet-mps` /
// `arnet-algorithms` reach through a direct `arnet-linalg` dependency, not
// this umbrella. Their result types, by contrast, ARE re-exported below:
// the Host-defaulting `BlockSparseHostOps` methods return them, so an
// umbrella user must be able to name them — mirroring the dense result
// aliases re-exported further down.
pub use arnet_linalg::{
    BlockScalars, BlockSparseEigResult, BlockSparseEighResult, BlockSparseQrResult,
    BlockSparseSvdResult, BlockSparseTruncSvdResult,
};

// Ergonomic Host-defaulting method surface over the explicit-backend paths.
pub use arnet_linalg::{BlockSparseHostOps, DenseHostOps};

// Expert layer: the per-call `ExecPolicy` escape hatch over the auto-policy
// default. Re-exported as the `arnet::expert` namespace so an umbrella-only
// consumer can reach `expert::permute`, `expert::contract`,
// `expert::svd`, … — the decomposition policy variants dispatch over layout,
// so `expert::svd` serves both Dense and BlockSparse.
pub use arnet_linalg::expert;

// `flat_index` is intentionally not re-exported: it takes a `MemoryOrder`
// argument, so exposing it on the umbrella would reintroduce the
// memory-order leak that the rest of this surface closes. End users do not
// need memory-order-aware index math; in-tree code that does (tests) depends
// on `arnet-tensor` directly.

// Linalg-level error type and SVD parameters.
pub use arnet_linalg::{LinalgError, TruncSvdParams};
pub use arnet_tensor::TensorError;
pub use ops::{EigResult, EighResult, LqResult, QrResult, SvdResult, TruncSvdResult};

// Re-export the native backend
pub use arnet_native::NativeBackend;
