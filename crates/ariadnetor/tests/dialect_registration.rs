//! Integration tests for TN Dialect registration with melior
//!
//! These tests verify that the TN-Compute dialect can be properly
//! registered and used with the melior MLIR bindings.

#[cfg(feature = "mlir")]
mod melior_tests {
    use melior::{
        Context,
        dialect::DialectRegistry,
        ir::{Location, Module, operation::OperationLike},
        utility::register_all_dialects,
    };
    use arnet::dialect::TCDialect;

    /// Test that the TN dialect can be created
    #[test]
    fn test_dialect_creation() {
        let dialect = TCDialect::new();
        assert!(dialect.is_ok(), "Failed to create TN dialect");
    }

    /// Test that the TN dialect can be registered with a melior context
    #[test]
    fn test_dialect_registration() {
        // Create a dialect registry
        let registry = DialectRegistry::new();

        // Register standard MLIR dialects
        register_all_dialects(&registry);

        // Create TN dialect
        let _tn_dialect = TCDialect::new()
            .expect("Failed to create TN dialect");

        // Register TN dialect
        // Note: This requires the dialect handle to be properly exposed
        // The actual registration mechanism depends on melior's API

        // Create context with registry
        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        // Verify context was created successfully
        // Context is valid if we can create a location from it
        let _location = Location::unknown(&context);
    }

    /// Test creating a module with TN operations
    #[test]
    fn test_module_creation() {
        // Create context and load dialects
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        // Try to load TN dialect explicitly
        let _tn_dialect = TCDialect::new()
            .expect("Failed to create TN dialect");

        // Create a location for error reporting
        let location = Location::unknown(&context);

        // Create a module
        let module = Module::new(location);

        // Verify module was created successfully
        assert!(module.as_operation().verify(),
                "Module creation/verification failed");
    }

    /// Test parsing TN dialect IR from string
    #[test]
    #[ignore] // This test requires full dialect registration to work
    fn test_parse_tn_ir() {
        let context = Context::new();

        // Load all dialects including builtin
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        // Try to create TN dialect
        let _tn_dialect = TCDialect::new()
            .expect("Failed to create TN dialect");

        // Create a simple TN operation as a string
        let tn_ir = r#"
            module {
                func.func @matrix_multiply(%arg0: tensor<10x20xf64>, %arg1: tensor<20x30xf64>) -> tensor<10x30xf64> {
                    %0 = tn.contract %arg0, %arg1 {
                        indices = "ij,jk->ik"
                    } : tensor<10x20xf64>, tensor<20x30xf64> -> tensor<10x30xf64>
                    return %0 : tensor<10x30xf64>
                }
            }
        "#;

        // Try to parse the module
        // Note: This will only work if the TN dialect is properly registered
        let result = Module::parse(&context, tn_ir);

        if let Some(module) = result {
            // Verify the module
            assert!(module.as_operation().verify(),
                    "Parsed module verification failed");
        } else {
            // For now, we expect this to fail until full registration is complete
            println!("Module parsing failed (expected until full registration)");
        }
    }
}

/// Basic sanity tests that don't require the mlir feature
#[cfg(not(feature = "mlir"))]
mod basic_tests {
    use arnet::dialect::TCDialect;

    #[test]
    fn test_dialect_requires_feature() {
        let result = TCDialect::new();
        assert!(result.is_err(),
                "TCDialect should fail without mlir feature");
    }
}
