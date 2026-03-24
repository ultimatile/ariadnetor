//! Ariadnetor: tensor network framework in Rust
//!
//! # Example
//!
//! ```
//! use arnet::Tensor;
//!
//! let a = Tensor::<f64>::zeros(vec![2, 3]);
//! let b = Tensor::<f64>::zeros(vec![3, 2]);
//!
//! assert_eq!(a.shape(), &[2, 3]);
//! assert_eq!(b.shape(), &[3, 2]);
//! ```

pub mod diag_tensor;
pub mod expr;
#[cfg(feature = "mps")]
pub mod mps;
pub mod ops;
pub mod runtime;
pub mod tensor;

// Main types
pub use diag_tensor::DiagTensor;
pub use expr::ExpressionComputeGraph;
pub use tensor::{DenseTensor, Tensor, TensorStorage};

// Re-export from ariadnetor-core
pub use arnet_core::{ComputeBackend, ContractionError, EinsumExpr, LabelId, Scalar};

// High-level free functions (backend extracted from Tensor)
pub use ops::{
    contract, diag, eig, eigh, eigvals, eigvalsh, einsum, expm, expm_antihermitian, expm_hermitian,
    inverse, linear_combine, lq, norm, normalize, qr, scale, solve, svd, trace, transpose,
    trunc_svd,
};

// Re-export result types from linalg
pub use arnet_linalg::{
    EigResult, EighResult, LqResult, QrResult, SvdResult, TruncSvdParams, TruncSvdResult,
};

// Re-export the native backend
pub use arnet_native::NativeBackend;
