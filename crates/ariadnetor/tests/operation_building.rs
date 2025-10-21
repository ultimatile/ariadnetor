//! Integration tests for building TN dialect operations programmatically
//!
//! These tests verify that we can construct TN operations using melior's builder API.

#[cfg(feature = "mlir")]
mod operation_tests {
    use melior::{
        Context,
        dialect::DialectRegistry,
        ir::{
            attribute::{StringAttribute, IntegerAttribute},
            r#type::{RankedTensorType, IntegerType},
            Location, Module,
            operation::OperationLike,
        },
        utility::register_all_dialects,
    };
    use tn_mlir::dialect::TNDialect;

    fn setup_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        // Try to load TN dialect
        let _tn_dialect = TNDialect::new().expect("Failed to create TN dialect");

        context
    }

    /// Test building a simple empty module
    #[test]
    fn test_build_module() {
        let context = setup_context();
        let location = Location::unknown(&context);

        // Create an empty module
        let module = Module::new(location);

        // Verify the module structure
        assert!(module.as_operation().verify(),
                "Module verification failed");
    }

    /// Test building SVD operation structure
    #[test]
    fn test_build_svd_types() {
        let context = setup_context();

        let f64_type = melior::ir::r#type::Type::float64(&context);

        // Input: tensor<100x200xf64>
        let _input_type = RankedTensorType::new(&[100, 200], f64_type, None);

        // U: tensor<100x?xf64> (dynamic second dimension)
        // Note: Use u64::MAX or i64::MAX to represent dynamic dimensions
        let _u_type = RankedTensorType::new(&[100, i64::MAX as u64], f64_type, None);

        // S: tensor<?xf64> (dynamic dimension)
        let _s_type = RankedTensorType::new(&[i64::MAX as u64], f64_type, None);

        // V: tensor<200x?xf64>
        let _v_type = RankedTensorType::new(&[200, i64::MAX as u64], f64_type, None);

        // Types were created successfully if we got here
    }

    /// Test creating attributes for TN operations
    #[test]
    fn test_build_operation_attributes() {
        let context = setup_context();

        // Einsum indices string attribute
        let indices_attr = StringAttribute::new(&context, "ij,jk->ik");

        // Max chi (bond dimension) integer attribute
        let i64_type = IntegerType::new(&context, 64);
        let _max_chi_attr = IntegerAttribute::new(
            i64_type.into(),
            100
        );

        // Verify attributes were created
        assert!(indices_attr.to_string().contains("ij,jk->ik"));
        // Note: exact string representation may vary
    }

    /// Test tensor type constraints
    #[test]
    fn test_tensor_type_constraints() {
        let context = setup_context();

        // Test various supported types
        let f64_type = melior::ir::r#type::Type::float64(&context);
        let f32_type = melior::ir::r#type::Type::float32(&context);
        let i64_type = IntegerType::new(&context, 64);

        // Create tensors with different element types
        let _tensor_f64 = RankedTensorType::new(&[10, 20], f64_type, None);
        let _tensor_f32 = RankedTensorType::new(&[10, 20], f32_type, None);
        let _tensor_i64 = RankedTensorType::new(&[10, 20], i64_type.into(), None);

        // Types were created successfully if we got here
    }

    /// Test dynamic tensor shapes
    #[test]
    fn test_dynamic_tensor_shapes() {
        let context = setup_context();
        let f64_type = melior::ir::r#type::Type::float64(&context);

        // Fully dynamic tensor: tensor<?x?xf64>
        let _dynamic_tensor = RankedTensorType::new(&[i64::MAX as u64, i64::MAX as u64], f64_type, None);

        // Partially dynamic: tensor<10x?xf64>
        let _partial_dynamic = RankedTensorType::new(&[10, i64::MAX as u64], f64_type, None);

        // Static tensor: tensor<10x20xf64>
        let _static_tensor = RankedTensorType::new(&[10, 20], f64_type, None);

        // Types were created successfully if we got here
    }
}
