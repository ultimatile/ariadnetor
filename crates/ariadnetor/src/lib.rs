//! Ariadnetor: MLIR-based Distributed Tensor Network Framework
//!
//! Ariadnetor provides high-performance tensor operations with MLIR compilation.
//!
//! # Architecture
//!
//! ```text
//! Einsum DSL → Tensor Dialect → LinAlg → LLVM
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
pub mod tensor;
pub mod runtime;

pub use einsum::EinsumExpr;
pub use tensor::Tensor;

// Re-export from ariadnetor-tensor-compute-dialect
pub use ariadnetor_tensor_compute_dialect::{TCDialect, TCBuilder, MemRefDescriptor};
