//! End-to-end pipeline tests
//!
//! Tests the full pipeline from einsum notation to IR generation.

#[cfg(feature = "mlir")]
mod end_to_end_tests {
    use melior::{
        Context,
        dialect::DialectRegistry,
        utility::register_all_dialects,
    };
    use tn_mlir::{TNJITCompiler, TNDialect, Tensor};

    fn setup_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        // Load TN dialect
        let _tn_dialect = TNDialect::new().expect("Failed to create TN dialect");

        context
    }

    #[test]
    fn test_compile_and_execute_ir_generation() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Create test tensors
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        // Attempt to compile and execute
        // This will fail at execution step, but should successfully generate IR
        let result = compiler.compile_and_execute("ij,jk->ik", &a, &b);

        // We expect an error about JIT execution not being fully implemented
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("JIT execution") || err_msg.contains("not yet fully implemented"),
            "Expected error about unimplemented JIT execution, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_compile_and_execute_invalid_einsum() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        // Invalid einsum notation
        let result = compiler.compile_and_execute("invalid", &a, &b);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Invalid") || err_msg.contains("expected format"),
            "Expected parse error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_compile_and_execute_dimension_mismatch() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0], vec![2, 2]); // Should be [3, 2]

        // Dimension mismatch
        let result = compiler.compile_and_execute("ij,jk->ik", &a, &b);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Dimension mismatch") || err_msg.contains("mismatch"),
            "Expected dimension mismatch error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_compile_and_execute_batch_matmul() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Create batched tensors (batch=2, 2x3 and 3x2 matrices)
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![2, 2, 3]
        );
        let b = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![2, 3, 2]
        );

        let result = compiler.compile_and_execute("bij,bjk->bik", &a, &b);

        // Should fail at execution, but IR generation should work
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("JIT execution") || err_msg.contains("not yet fully implemented"),
            "Expected JIT execution error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_tensor_pointer_access() {
        // Test that Tensor FFI methods work correctly
        let mut tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Test const pointer
        let ptr = tensor.as_ptr();
        assert!(!ptr.is_null());

        // Test mutable pointer
        let mut_ptr = tensor.as_mut_ptr();
        assert!(!mut_ptr.is_null());

        // Verify we can modify through the pointer
        unsafe {
            *mut_ptr = 42.0;
        }
        assert_eq!(tensor.get(&[0, 0]), 42.0);
    }

    #[test]
    fn test_tensor_shape_i64_conversion() {
        let tensor = Tensor::new(vec![10, 20, 30]);
        let shape_i64 = tensor.shape_i64();

        assert_eq!(shape_i64, vec![10i64, 20i64, 30i64]);
    }
}
