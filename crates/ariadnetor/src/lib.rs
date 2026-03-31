//! Ariadnetor: tensor network framework in Rust
//!
//! # Example
//!
//! ```
//! use arnet::{Dense, Tensor};
//!
//! let a = Tensor::<Dense<f64>>::zeros(vec![2, 3]);
//! let b = Tensor::<Dense<f64>>::zeros(vec![3, 2]);
//!
//! assert_eq!(a.shape(), &[2, 3]);
//! assert_eq!(b.shape(), &[3, 2]);
//! ```

pub mod diag_tensor;
#[cfg(feature = "mps")]
pub mod mps;
pub mod ops;
pub mod tensor;

// Main types
pub use diag_tensor::DiagTensor;
pub use tensor::{Dense, Tensor};

// Re-export from ariadnetor-core
pub use arnet_core::{ComputeBackend, ContractionError, EinsumExpr, LabelId, Scalar};

// High-level free functions (backend extracted from Tensor)
pub use ops::{
    contract, diag, diag_extract, eig, eigh, eigvals, eigvalsh, einsum, expm, expm_antihermitian,
    expm_hermitian, inverse, linear_combine, lq, norm, normalize, qr, scale, solve, svd, trace,
    transpose, trunc_svd,
};

// Re-export result types (ops-level with DiagTensor for SVD)
pub use arnet_linalg::{LinalgError, TruncSvdParams};
pub use ops::{EigResult, EighResult, LqResult, QrResult, SvdResult, TruncSvdResult};

// Re-export the native backend
pub use arnet_native::NativeBackend;
