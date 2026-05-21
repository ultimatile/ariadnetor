//! Ariadnetor: tensor network framework in Rust
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
// the joined `Tensor<St, L, B>` type. Legacy `Dense` / `BlockSparse`
// representations are intentionally not re-exported here — consumers
// should only see the joined type.
pub use arnet_tensor::{
    BlockCoord, BlockMeta, BlockSparseLayout, BlockSparseStorage, BlockSparseTensorData,
    DenseLayout, DenseStorage, DenseTensorData, Direction, QNIndex, Sector, Storage, StorageFor,
    TensorLayout, U1Sector, Z2Sector,
};

// Re-export from ariadnetor-core
pub use arnet_core::{ComputeBackend, ContractionError, EinsumExpr, LabelId, MemoryOrder, Scalar};

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

// Reorder helper used by downstream crates for memory-order conversions
// at axis-merge / axis-split boundaries.
pub use arnet_tensor::{flat_index, reorder};

// Linalg-level error type and SVD parameters.
pub use arnet_linalg::{LinalgError, TruncSvdParams};
pub use ops::{EigResult, EighResult, LqResult, QrResult, SvdResult, TruncSvdResult};

// Re-export the native backend
pub use arnet_native::NativeBackend;
