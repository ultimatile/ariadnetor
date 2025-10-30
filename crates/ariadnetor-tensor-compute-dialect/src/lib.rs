//! Ariadnetor Tensor Dialect
//!
//! MLIR dialect for tensor operations in the Ariadnetor framework.
//!
//! This crate provides:
//! - Tensor Dialect definition (C++ side, to be implemented)
//! - IR Builder for tensor operations
//! - MemRef descriptor utilities

pub mod builder;
pub mod dialect;
pub mod memref;

#[cfg(feature = "mlir")]
pub mod ffi;

pub use builder::TCBuilder;
pub use dialect::TCDialect;
pub use memref::MemRefDescriptor;

use anyhow::Result;

/// Initialize the Tensor Dialect
///
/// This is a placeholder until C++ integration is complete.
pub fn initialize() -> Result<()> {
    todo!("MLIR dialect initialization requires C++ integration")
}
