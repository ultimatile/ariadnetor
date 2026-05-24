//! Ariadnetor: tensor network framework in Rust.
//!
//! `arnet` is the umbrella tensor library; it re-exports types and
//! functions from the layers listed below into its own namespace.
//! Each layer depends only on the ones above it in this list:
//!
//! - [`arnet_core`] — backend-agnostic abstractions (`Scalar`,
//!   `ComputeBackend`, `EinsumExpr`, `MemoryOrder`).
//! - [`arnet_native`] — `NativeBackend`: faer + hptt-rs.
//! - [`arnet_tensor`] — user-facing `Tensor`, `DenseTensor`,
//!   `BlockSparseTensor`, `Sector`, `QNIndex`.
//! - [`arnet_linalg`] — backend-agnostic linear algebra over
//!   `&Tensor` (contract, svd, qr, eigh, expm, …).
//!
//! `arnet_mps` and `arnet_algorithms` are *separate* consumer crates
//! that build on this umbrella; they are not re-exported here.
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
// parameterize their own generic containers (e.g. `Mps<St, L, B>`) over
// the joined `Tensor<St, L, B>` type. The legacy `Dense` / `BlockSparse`
// representations and the `TensorData<St, L>` joined-data aliases are
// intentionally not re-exported here — umbrella consumers should only
// see the joined `Tensor` surface. `TensorData` stays `pub` in
// `arnet-tensor` for crates that perform cross-crate kernel access;
// such crates depend on `arnet-tensor` directly rather than the
// umbrella.
pub use arnet_tensor::{
    BlockCoord, BlockMeta, BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage,
    Direction, QNIndex, Sector, Storage, StorageFor, TensorLayout, U1Sector, Z2Sector,
};

// Re-export from ariadnetor-core
pub use arnet_core::{
    Complex, ComputeBackend, ContractionError, EinsumExpr, LabelId, MemoryOrder, Scalar,
};

// High-level free functions (backend extracted from Tensor)
pub use ops::{
    contract, diag, eig, eigh, eigvals, eigvalsh, einsum, expm, expm_antihermitian, expm_hermitian,
    inverse, linear_combine, lq, norm, normalize, qr, scale, solve, svd, trace, transpose,
    trunc_svd,
};

// Block-sparse low-level free functions and result types. Needed by
// downstream crates (`arnet-mps`, `arnet-algorithms`) whose internals
// perform per-site block-sparse contractions and decompositions.
pub use arnet_linalg::{
    BlockSingularValues, BlockSparseContractResult, BlockSparseQrResult, BlockSparseSvdResult,
    BlockSparseTruncSvdResult, contract_block_sparse, diagonal_scale, diagonal_scale_block_sparse,
    fuse_legs_block_sparse, lq_block_sparse, permute_block_sparse, qr_block_sparse,
    svd_block_sparse, trunc_svd_block_sparse,
};

// Coordinate-to-flat index helper used by downstream callers for
// memory-order-aware index math. The actual reorder routine lives as
// the `DenseTensor::reordered` inherent method on the joined surface
// (see `arnet_tensor::dense_ops`); the `DenseTensorData`-typed
// `reorder_dense_data` is intentionally not re-exported here.
pub use arnet_tensor::flat_index;

// Linalg-level error type and SVD parameters.
pub use arnet_linalg::{LinalgError, TruncSvdParams};
pub use ops::{EigResult, EighResult, LqResult, QrResult, SvdResult, TruncSvdResult};

// Re-export the native backend
pub use arnet_native::NativeBackend;
