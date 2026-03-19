//! Ariadnetor: tensor network framework in Rust
//!
//! # Example
//!
//! ```
//! use arnet::{Tensor, einsum};
//!
//! let a = Tensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
//! let b = Tensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
//!
//! let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
//! assert_eq!(c.shape(), &[2, 2]);
//! ```

pub mod expr;
pub mod ops;
pub mod runtime;
pub mod tensor;

// Main types
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
