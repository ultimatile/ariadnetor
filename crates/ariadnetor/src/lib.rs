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

mod ops;

// Main types
pub use arnet_tensor::{BlockSparseTensor, DenseTensor, Tensor};

// Storage / Layout building blocks. Required by downstream crates that
// parameterize their own generic containers (e.g. `Mps<St, L>`) over
// the joined `Tensor<St, L>` type. The `TensorData<St, L>`
// joined-data aliases are intentionally not re-exported here — umbrella
// consumers should only see the joined `Tensor` surface. `TensorData`
// stays `pub` in `arnet-tensor` for crates that perform cross-crate
// kernel access; such crates depend on `arnet-tensor` directly rather
// than the umbrella.
pub use arnet_tensor::{
    BlockCoord, BlockMeta, BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage,
    Direction, QNIndex, Sector, Storage, StorageFor, TensorLayout, U1Sector, Z2Sector,
};

// Backend-capability scaffolding: the `OpsFor<St>` marker and the `Host`
// substrate alias.
pub use arnet_tensor::{Host, OpsFor};

// Re-export from ariadnetor-core
pub use arnet_core::{Complex, ComputeBackend, ContractionError, EinsumExpr, LabelId, Scalar};

// High-level free functions over host-resident dense tensors (no backend).
pub use arnet_tensor::{add_all, linear_combine};
// Explicit-backend dense free functions (backend supplied at the call site).
// The single-backend ergonomic surface is the `DenseHostOps` /
// `BlockSparseHostOps` extension traits re-exported below, not free functions.
pub use ops::{
    contract_with_backend, diag_with_backend, diagonal_scale_with_backend, eig_with_backend,
    eigh_with_backend, eigvals_with_backend, eigvalsh_with_backend, einsum_with_backend,
    expm_antihermitian_with_backend, expm_hermitian_with_backend, expm_with_backend,
    inverse_with_backend, lq_with_backend, qr_with_backend, solve_with_backend, svd_with_backend,
    trace_with_backend, transpose_with_backend, trunc_svd_with_backend,
};

// The block-sparse low-level free functions and their result types are
// intentionally not re-exported: they are consumer-internal API that
// `arnet-mps` / `arnet-algorithms` reach through a direct `arnet-linalg`
// dependency, not this umbrella.

// Ergonomic Host-defaulting method surface over the explicit-backend paths.
pub use arnet_linalg::{BlockSparseHostOps, DenseHostOps};

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
