//! Ariadnetor: tensor network framework in Rust
//!
//! # Architecture
//!
//! ```text
//! Einsum DSL → Tensor Operations → BLAS/LAPACK
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use arnet::{einsum, NativeBackend};
//! use arnet_tensor::DenseTensor;
//!
//! let backend = NativeBackend::new();
//! let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
//! let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
//!
//! // Matrix multiplication
//! let c = einsum(&backend, &[&a, &b], "ij,jk->ik").unwrap();
//! ```

pub mod expr;
pub mod runtime;
pub mod tensor;

// Main types
pub use tensor::{DenseTensor, Tensor, TensorStorage};
pub use expr::ExpressionComputeGraph;

// Re-export from ariadnetor-core
pub use arnet_core::{ComputeBackend, ContractionError, EinsumExpr, LabelId, Scalar};

// Re-export backend-agnostic linear algebra operations
pub use arnet_linalg::{contract, einsum, transpose};

// Re-export the native backend
pub use arnet_native::NativeBackend;
