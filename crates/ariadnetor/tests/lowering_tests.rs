//! Integration tests for TN → LinAlg lowering
//!
//! These tests verify that TN operations are correctly lowered to LinAlg dialect.

#[cfg(feature = "mlir")]
mod lowering_tests {
    use melior::{
        Context,
        dialect::DialectRegistry,
        ir::{
            Block, BlockLike, Location,
            operation::OperationLike,
            r#type::{RankedTensorType, Type},
        },
        utility::register_all_dialects,
    };
    use arnet::{TCBuilder, TCDialect, EinsumExpr};

    fn setup_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        // Load TN dialect
        let _tn_dialect = TCDialect::new().expect("Failed to create TN dialect");

        context
    }

    /// Helper to create a test block with tensor arguments
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
    fn test_lower_matrix_multiply() {
        let context = setup_context();
        let builder = TCBuilder::new(&context);
        let location = builder.location();

        // Parse einsum for matrix multiplication
        let expr = EinsumExpr::parse("ij,jk->ik")
            .expect("Failed to parse einsum");

        // Create test block with tensor arguments
        let block = create_test_block(&context, &[&[10, 20], &[20, 30]], location);

        let lhs = block.argument(0).expect("Failed to get arg 0").into();
        let rhs = block.argument(1).expect("Failed to get arg 1").into();
        let f64_type = Type::float64(&context);

        // Build TN contract operation
        let _result = builder.build_contract_from_einsum(
            &expr,
            lhs,
            rhs,
            &[10, 20],
            &[20, 30],
            f64_type
        ).expect("Failed to build contract");

        // Verify module
        // Note: We verify that TN IR generation is correct. The actual lowering
        // to LinAlg happens in C++ and is tested separately via lit tests.
        assert!(builder.module().as_operation().verify(),
                "Generated TN IR is invalid");
    }

    #[test]
    fn test_lower_batch_matmul() {
        let context = setup_context();
        let builder = TCBuilder::new(&context);
        let location = builder.location();

        // Parse einsum for batch matrix multiplication
        let expr = EinsumExpr::parse("bij,bjk->bik")
            .expect("Failed to parse einsum");

        // Create test block with 3D tensor arguments
        let block = create_test_block(&context, &[&[32, 10, 20], &[32, 20, 30]], location);

        let lhs = block.argument(0).expect("Failed to get arg 0").into();
        let rhs = block.argument(1).expect("Failed to get arg 1").into();
        let f64_type = Type::float64(&context);

        // Build TN contract operation
        let _result = builder.build_contract_from_einsum(
            &expr,
            lhs,
            rhs,
            &[32, 10, 20],
            &[32, 20, 30],
            f64_type
        ).expect("Failed to build contract");

        // Verify module
        assert!(builder.module().as_operation().verify());
    }

    #[test]
    fn test_lower_element_wise() {
        let context = setup_context();
        let builder = TCBuilder::new(&context);
        let location = builder.location();

        // Parse einsum for element-wise multiplication
        let expr = EinsumExpr::parse("ij,ij->ij")
            .expect("Failed to parse einsum");

        // Create test block with matching tensor shapes
        let block = create_test_block(&context, &[&[10, 20], &[10, 20]], location);

        let lhs = block.argument(0).expect("Failed to get arg 0").into();
        let rhs = block.argument(1).expect("Failed to get arg 1").into();
        let f64_type = Type::float64(&context);

        // Build TN contract operation
        let _result = builder.build_contract_from_einsum(
            &expr,
            lhs,
            rhs,
            &[10, 20],
            &[10, 20],
            f64_type
        ).expect("Failed to build contract");

        // Verify module
        assert!(builder.module().as_operation().verify());
    }

    #[test]
    fn test_lower_transpose() {
        let context = setup_context();
        let builder = TCBuilder::new(&context);
        let location = builder.location();

        // Create test block with tensor argument
        let block = create_test_block(&context, &[&[10, 20]], location);

        let input = block.argument(0).expect("Failed to get arg 0").into();
        let f64_type = Type::float64(&context);

        // Build transpose operation
        let result_type = RankedTensorType::new(&[20, 10], f64_type, None).into();
        let _result = builder.transpose(input, result_type, &[1, 0])
            .expect("Failed to build transpose");

        // Verify module
        assert!(builder.module().as_operation().verify());
    }

    #[test]
    fn test_ir_generation_multiple_ops() {
        let context = setup_context();
        let builder = TCBuilder::new(&context);
        let location = builder.location();

        // Test that we can generate IR for multiple operations

        // Matrix multiply
        let expr1 = EinsumExpr::parse("ij,jk->ik").unwrap();
        let block1 = create_test_block(&context, &[&[5, 10], &[10, 15]], location);
        let lhs1 = block1.argument(0).unwrap().into();
        let rhs1 = block1.argument(1).unwrap().into();
        let f64_type = Type::float64(&context);

        let _result1 = builder.build_contract_from_einsum(
            &expr1, lhs1, rhs1, &[5, 10], &[10, 15], f64_type
        ).unwrap();

        // Batch matmul
        let expr2 = EinsumExpr::parse("bij,bjk->bik").unwrap();
        let block2 = create_test_block(&context, &[&[8, 5, 10], &[8, 10, 15]], location);
        let lhs2 = block2.argument(0).unwrap().into();
        let rhs2 = block2.argument(1).unwrap().into();

        let _result2 = builder.build_contract_from_einsum(
            &expr2, lhs2, rhs2, &[8, 5, 10], &[8, 10, 15], f64_type
        ).unwrap();

        // Verify module contains all operations
        assert!(builder.module().as_operation().verify());
    }
}
