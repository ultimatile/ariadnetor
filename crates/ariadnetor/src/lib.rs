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
mod tensor;

// Main types
pub use tensor::{BlockSparseTensor, DenseTensor, Tensor};

// Re-export from ariadnetor-core
pub use arnet_core::{ComputeBackend, ContractionError, EinsumExpr, LabelId, MemoryOrder, Scalar};

// High-level free functions (backend extracted from Tensor)
pub use ops::{
    contract, diag, eig, eigh, eigvals, eigvalsh, einsum, expm, expm_antihermitian, expm_hermitian,
    inverse, linear_combine, lq, norm, normalize, qr, scale, solve, trace, transpose,
};

// Linalg-level error type and SVD parameters.
pub use arnet_linalg::{LinalgError, TruncSvdParams};
pub use ops::{EigResult, EighResult, LqResult, QrResult};

// Re-export the native backend
pub use arnet_native::NativeBackend;
