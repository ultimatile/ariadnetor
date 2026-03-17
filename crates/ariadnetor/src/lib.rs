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
//! use arnet::{Tensor, einsum};
//!
//! // Matrix multiplication using einsum notation
//! let a = Tensor::new(vec![100, 200]);
//! let b = Tensor::new(vec![200, 300]);
//!
//! let c = einsum("ij,jk->ik", vec![&a, &b]);
//! ```

pub mod einsum;
pub mod expr;
pub mod runtime;
pub mod tensor;

pub use einsum::EinsumExpr;
pub use expr::ExpressionComputeGraph;
pub use tensor::Tensor;

// Re-export from ariadnetor-core
pub use arnet_core::{ComputeBackend, ContractionError, LabelId, Scalar};

// Re-export backend-agnostic linear algebra operations
pub use arnet_linalg::{contract, transpose};

// Re-export the CPU backend
pub use arnet_cpu::CpuBackend;
