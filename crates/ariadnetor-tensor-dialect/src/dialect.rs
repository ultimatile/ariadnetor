//! TN-Compute Dialect registration and initialization
//!
//! This module provides the Rust interface to the TN-Compute MLIR dialect.

use anyhow::Result;

#[cfg(feature = "mlir")]
use crate::ffi;

/// TN-Compute Dialect wrapper
pub struct TNDialect {
    #[cfg(feature = "mlir")]
    handle: ffi::MlirDialectHandle,
}

impl TNDialect {
    /// Create a new TN dialect instance
    ///
    /// Returns a handle to the TN dialect that can be used for registration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use tn_mlir::TNDialect;
    ///
    /// let dialect = TNDialect::new()?;
    /// // Use with melior context registration
    /// ```
    pub fn new() -> Result<Self> {
        #[cfg(feature = "mlir")]
        {
            let handle = unsafe { ffi::mlirGetDialectHandle__tn__() };
            Ok(Self { handle })
        }

        #[cfg(not(feature = "mlir"))]
        {
            anyhow::bail!("TNDialect requires 'mlir' feature to be enabled")
        }
    }

    #[cfg(feature = "mlir")]
    pub fn handle(&self) -> &ffi::MlirDialectHandle {
        &self.handle
    }
}

impl Default for TNDialect {
    fn default() -> Self {
        Self::new().expect("Failed to create TN dialect")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "mlir")]
    fn test_dialect_creation() {
        // This test will pass when mlir feature is enabled
        let dialect = TNDialect::new();
        assert!(dialect.is_ok());
    }

    #[test]
    #[cfg(not(feature = "mlir"))]
    fn test_dialect_requires_feature() {
        let dialect = TNDialect::new();
        assert!(dialect.is_err());
    }
}
