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
//! use arnet::{einsum, CpuBackend};
//! use arnet_tensor::DenseTensor;
//!
//! let backend = CpuBackend::new();
//! let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
//! let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
//!
//! // Matrix multiplication
//! let c = einsum(&backend, &[&a, &b], "ij,jk->ik").unwrap();
//! ```

pub mod expr;
pub mod runtime;
pub mod tensor;

// Re-export EinsumExpr from core (unified in #29)
pub use arnet_core::EinsumExpr;
pub use expr::ExpressionComputeGraph;
pub use tensor::Tensor;

// Re-export from ariadnetor-core
pub use arnet_core::{ComputeBackend, ContractionError, LabelId, Scalar};

// Re-export backend-agnostic linear algebra operations
pub use arnet_linalg::{contract, einsum, transpose};

// Re-export the CPU backend
pub use arnet_cpu::CpuBackend;
