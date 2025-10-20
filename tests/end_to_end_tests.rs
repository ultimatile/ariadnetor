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
        let mut a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        // Attempt to compile and execute
        let result = compiler.compile_and_execute("ij,jk->ik", &mut a, &mut b);

        // We expect success now that the implementation is complete
        assert!(result.is_ok(), "Compilation and execution failed: {:?}", result.err());
    }

    #[test]
    fn test_compile_and_execute_numerical_correctness() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Test case: 2x2 matrix multiplication
        // A = [[1, 2],    B = [[5, 6],    Expected C = [[19, 22],
        //      [3, 4]]         [7, 8]]                   [43, 50]]
        let mut a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let result = compiler.compile_and_execute("ij,jk->ik", &mut a, &mut b)
            .expect("Compilation and execution failed");

        // Verify result shape
        assert_eq!(result.shape(), &[2, 2]);

        // Verify numerical correctness
        // C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 5 + 14 = 19
        assert_eq!(result.get(&[0, 0]), 19.0, "C[0,0] mismatch");

        // C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 6 + 16 = 22
        assert_eq!(result.get(&[0, 1]), 22.0, "C[0,1] mismatch");

        // C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*5 + 4*7 = 15 + 28 = 43
        assert_eq!(result.get(&[1, 0]), 43.0, "C[1,0] mismatch");

        // C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 3*6 + 4*8 = 18 + 32 = 50
        assert_eq!(result.get(&[1, 1]), 50.0, "C[1,1] mismatch");
    }

    #[test]
    fn test_compile_and_execute_non_square_matrices() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Test case: 2x3 @ 3x2 matrix multiplication
        // A = [[1, 2, 3],    B = [[7,  8],     Expected C = [[58,  64],
        //      [4, 5, 6]]         [9,  10],                  [139, 154]]
        //                         [11, 12]]
        let mut a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let mut b = Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let result = compiler.compile_and_execute("ij,jk->ik", &mut a, &mut b)
            .expect("Compilation and execution failed");

        // Verify result shape
        assert_eq!(result.shape(), &[2, 2]);

        // Verify numerical correctness
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        assert_eq!(result.get(&[0, 0]), 58.0, "C[0,0] mismatch");

        // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        assert_eq!(result.get(&[0, 1]), 64.0, "C[0,1] mismatch");

        // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        assert_eq!(result.get(&[1, 0]), 139.0, "C[1,0] mismatch");

        // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        assert_eq!(result.get(&[1, 1]), 154.0, "C[1,1] mismatch");
    }

    #[test]
    fn test_compile_and_execute_invalid_einsum() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        let mut a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        // Invalid einsum notation
        let result = compiler.compile_and_execute("invalid", &mut a, &mut b);

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

        let mut a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let mut b = Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0], vec![2, 2]); // Should be [3, 2]

        // Dimension mismatch
        let result = compiler.compile_and_execute("ij,jk->ik", &mut a, &mut b);

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
        let mut a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![2, 2, 3]
        );
        let mut b = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![2, 3, 2]
        );

        let result = compiler.compile_and_execute("bij,bjk->bik", &mut a, &mut b);

        // This is not matrix multiplication (has batch dimension), so should fail
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("matrix multiplication") || err_msg.contains("ij,jk->ik"),
            "Expected error about only supporting matrix multiplication, got: {}",
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
