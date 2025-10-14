//! Integration tests for TNBuilder API
//!
//! These tests verify that the TNBuilder can construct TN dialect operations correctly.

#[cfg(feature = "mlir")]
mod builder_tests {
    use melior::{
        Context,
        dialect::DialectRegistry,
        ir::{
            Block, BlockLike,
            Location,
            operation::OperationLike,
            r#type::{RankedTensorType, Type},
        },
        utility::register_all_dialects,
    };
    use tn_mlir::{TNBuilder, TNDialect};

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

    /// Helper function to create a function block with tensor arguments for testing
    fn create_test_block<'c>(
        context: &'c Context,
        arg_shapes: &[&[u64]],
        location: Location<'c>,
    ) -> Block<'c> {
        let f64_type = Type::float64(context);

        let arg_types: Vec<_> = arg_shapes
            .iter()
            .map(|shape| {
                let tensor_type = RankedTensorType::new(shape, f64_type, None);
                (tensor_type.into(), location)
            })
            .collect();

        Block::new(&arg_types)
    }

    #[test]
    fn test_builder_creation() {
        let context = setup_context();
        let builder = TNBuilder::new(&context);

        // Verify module was created
        assert!(builder.module().as_operation().verify());
    }

    #[test]
    fn test_contract_operation() {
        let context = setup_context();
        let builder = TNBuilder::new(&context);
        let location = builder.location();

        // Create a test block with two tensor arguments (10x20 and 20x30)
        let block = create_test_block(&context, &[&[10, 20], &[20, 30]], location);

        let lhs = block.argument(0).expect("Failed to get arg 0").into();
        let rhs = block.argument(1).expect("Failed to get arg 1").into();

        // Create result type (10x30 tensor)
        let f64_type = Type::float64(&context);
        let result_type = RankedTensorType::new(&[10, 30], f64_type, None).into();

        // Build contract operation
        let result = builder.contract(lhs, rhs, result_type, "ij,jk->ik");

        assert!(result.is_ok(), "Contract operation build failed: {:?}", result.err());
    }

    #[test]
    fn test_svd_operation() {
        let context = setup_context();
        let builder = TNBuilder::new(&context);
        let location = builder.location();

        // Create a test block with one tensor argument (100x200)
        let block = create_test_block(&context, &[&[100, 200], &[100, 200]], location);

        let input = block.argument(0).expect("Failed to get arg 0").into();
        let input2 = block.argument(1).expect("Failed to get arg 1").into();

        // Create result types
        let f64_type = Type::float64(&context);
        let u_type = RankedTensorType::new(&[100, 100], f64_type, None).into();
        let s_type = RankedTensorType::new(&[100], f64_type, None).into();
        let v_type = RankedTensorType::new(&[200, 100], f64_type, None).into();

        // Build SVD operation without optional parameters
        let result = builder.svd(input, u_type, s_type, v_type, None, None);

        assert!(result.is_ok(), "SVD operation build failed: {:?}", result.err());

        // Test with optional parameters
        let result_with_params = builder.svd(
            input2,
            u_type,
            s_type,
            v_type,
            Some(50),
            Some(1e-10),
        );

        assert!(result_with_params.is_ok(), "SVD operation with params build failed: {:?}", result_with_params.err());
    }

    #[test]
    fn test_qr_operation() {
        let context = setup_context();
        let builder = TNBuilder::new(&context);
        let location = builder.location();

        // Create a test block with one tensor argument (100x50)
        let block = create_test_block(&context, &[&[100, 50]], location);

        let input = block.argument(0).expect("Failed to get arg 0").into();

        // Create result types
        let f64_type = Type::float64(&context);
        let q_type = RankedTensorType::new(&[100, 50], f64_type, None).into();
        let r_type = RankedTensorType::new(&[50, 50], f64_type, None).into();

        // Build QR operation
        let result = builder.qr(input, q_type, r_type);

        assert!(result.is_ok(), "QR operation build failed: {:?}", result.err());
    }

    #[test]
    fn test_transpose_operation() {
        let context = setup_context();
        let builder = TNBuilder::new(&context);
        let location = builder.location();

        // Create a test block with one tensor argument (10x20)
        let block = create_test_block(&context, &[&[10, 20]], location);

        let input = block.argument(0).expect("Failed to get arg 0").into();

        // Create result type (20x10 tensor, transposed)
        let f64_type = Type::float64(&context);
        let result_type = RankedTensorType::new(&[20, 10], f64_type, None).into();

        // Build transpose operation
        let result = builder.transpose(input, result_type, &[1, 0]);

        assert!(result.is_ok(), "Transpose operation build failed: {:?}", result.err());
    }

    #[test]
    fn test_reshape_operation() {
        let context = setup_context();
        let builder = TNBuilder::new(&context);
        let location = builder.location();

        // Create a test block with one tensor argument (100x20)
        let block = create_test_block(&context, &[&[100, 20]], location);

        let input = block.argument(0).expect("Failed to get arg 0").into();

        // Create result type (10x200 tensor, reshaped)
        let f64_type = Type::float64(&context);
        let result_type = RankedTensorType::new(&[10, 200], f64_type, None).into();

        // Build reshape operation
        let result = builder.reshape(input, result_type);

        assert!(result.is_ok(), "Reshape operation build failed: {:?}", result.err());
    }

    #[test]
    fn test_truncate_operation() {
        let context = setup_context();
        let builder = TNBuilder::new(&context);
        let location = builder.location();

        // Create a test block with two tensor arguments (100x200 each)
        let block = create_test_block(&context, &[&[100, 200], &[100, 200]], location);

        let input = block.argument(0).expect("Failed to get arg 0").into();
        let input2 = block.argument(1).expect("Failed to get arg 1").into();

        // Create result type (same shape, but may be truncated internally)
        let f64_type = Type::float64(&context);
        let result_type = RankedTensorType::new(&[100, 200], f64_type, None).into();

        // Build truncate operation without parameters
        let result1 = builder.truncate(input, result_type, None, None);
        assert!(result1.is_ok(), "Truncate operation build failed: {:?}", result1.err());

        // Build truncate operation with parameters
        let result2 = builder.truncate(input2, result_type, Some(50), Some(1e-8));
        assert!(result2.is_ok(), "Truncate operation with params build failed: {:?}", result2.err());
    }
}
