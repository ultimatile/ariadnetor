//! TN-MLIR: Tensor Network MLIR Dialect
//!
//! A distributed tensor network library with MLIR compilation frontend.
//!
//! # Architecture
//!
//! ```text
//! DSL (Einsum) → TN-Compute Dialect → LinAlg → LLVM → JIT/AOT
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use tn_mlir::{TNCompiler, Tensor};
//!
//! let mut compiler = TNCompiler::new();
//!
//! // Matrix multiplication using einsum notation
//! let a = Tensor::new(vec![100, 200]);
//! let b = Tensor::new(vec![200, 300]);
//!
//! let c = compiler.contract("ij,jk->ik", vec![&a, &b]);
//! ```

pub mod dialect;
pub mod builder;
pub mod jit;
pub mod tensor;

pub use dialect::TNDialect;
pub use builder::TNBuilder;
pub use jit::TNJITCompiler;
pub use tensor::Tensor;

use anyhow::Result;

/// Initialize the TN-Compute dialect and register it with MLIR
///
/// # Panics
///
/// This function is not yet implemented and will panic when called.
/// MLIR C++ integration is required first.
pub fn initialize() -> Result<()> {
    todo!("MLIR dialect initialization requires C++ integration")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        // This test will fail until MLIR C++ integration is implemented
        let result = initialize();
        assert!(result.is_ok());
    }
}
