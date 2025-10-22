//! Tensor Compute Dialect registration and initialization
//!
//! This module provides the Rust interface to the Tensor Compute MLIR dialect.

use anyhow::Result;

#[cfg(feature = "mlir")]
use crate::ffi;

/// Tensor Compute Dialect wrapper
pub struct TCDialect {
    #[cfg(feature = "mlir")]
    handle: ffi::MlirDialectHandle,
}

impl TCDialect {
    /// Create a new TC dialect instance
    ///
    /// Returns a handle to the TC dialect that can be used for registration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use tc_mlir::TCDialect;
    ///
    /// let dialect = TCDialect::new()?;
    /// // Use with melior context registration
    /// ```
    pub fn new() -> Result<Self> {
        #[cfg(feature = "mlir")]
        {
            let handle = unsafe { ffi::mlirGetDialectHandle__tc__() };
            Ok(Self { handle })
        }

        #[cfg(not(feature = "mlir"))]
        {
            anyhow::bail!("TCDialect requires 'mlir' feature to be enabled")
        }
    }

    #[cfg(feature = "mlir")]
    pub fn handle(&self) -> &ffi::MlirDialectHandle {
        &self.handle
    }
}

impl Default for TCDialect {
    fn default() -> Self {
        Self::new().expect("Failed to create TC dialect")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "mlir")]
    fn test_dialect_creation() {
        // This test will pass when mlir feature is enabled
        let dialect = TCDialect::new();
        assert!(dialect.is_ok());
    }

    #[test]
    #[cfg(not(feature = "mlir"))]
    fn test_dialect_requires_feature() {
        let dialect = TCDialect::new();
        assert!(dialect.is_err());
    }
}
