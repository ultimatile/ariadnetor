//! MemRef execution test with loops
//!
//! Tests more complex MemRef operations including:
//! 1. Creating MemRef descriptors from Tensors
//! 2. Passing them to JIT-compiled functions with loops
//! 3. Verifying numerical correctness across all elements

#[cfg(feature = "mlir")]
mod execution_engine_memref_tests {
    use arnet::{MemRefDescriptor, Tensor};
    use melior::{
        Context, ExecutionEngine,
        dialect::DialectRegistry,
        ir::{Location, Module, operation::OperationLike},
        pass::PassManager,
        utility::{register_all_dialects, register_all_llvm_translations},
    };

    fn setup_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        register_all_llvm_translations(&context);

        context
    }

    /// Test scalar addition to all MemRef elements using a loop
    #[test]
    fn test_memref_scalar_add() {
        let context = setup_context();
        let _location = Location::unknown(&context);

        // Create module with function that adds a scalar to all memref elements
        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @add_scalar(
                    %arg0: memref<4xf64>,
                    %arg1: f64,
                    %result: memref<4xf64>
                ) attributes { llvm.emit_c_interface } {
                    %c0 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    %c4 = arith.constant 4 : index

                    scf.for %i = %c0 to %c4 step %c1 {
                        %v = memref.load %arg0[%i] : memref<4xf64>
                        %sum = arith.addf %v, %arg1 : f64
                        memref.store %sum, %result[%i] : memref<4xf64>
                    }

                    return
                }
            }
            "#,
        )
        .expect("Failed to parse module");

        // Apply lowering passes
        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(melior::pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(melior::pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(melior::pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(melior::pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(melior::pass::conversion::create_finalize_mem_ref_to_llvm());
        pass_manager.add_pass(melior::pass::conversion::create_to_llvm());
        pass_manager.add_pass(melior::pass::conversion::create_reconcile_unrealized_casts());

        pass_manager.run(&mut module).expect("Failed to run passes");

        // Verify module
        assert!(module.as_operation().verify());

        // Create ExecutionEngine
        let engine = ExecutionEngine::new(&module, 2, &[], false);

        // Create input tensor
        let mut input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let mut output = Tensor::new(vec![4]);

        // Create MemRef descriptors
        let mut input_desc = MemRefDescriptor::<1>::from_tensor_mut(&mut input);
        let mut output_desc = MemRefDescriptor::<1>::from_tensor_mut(&mut output);
        let mut scalar = 10.0f64;

        // Invoke function using direct function pointer call
        // We use lookup + transmute to call the C interface wrapper directly
        let func_ptr = engine.lookup("_mlir_ciface_add_scalar");
        assert!(
            !func_ptr.is_null(),
            "Function _mlir_ciface_add_scalar not found!"
        );

        type FuncType =
            unsafe extern "C" fn(*mut MemRefDescriptor<1>, f64, *mut MemRefDescriptor<1>);

        unsafe {
            let func: FuncType = std::mem::transmute(func_ptr);
            func(&mut input_desc, scalar, &mut output_desc);
        }

        // Verify results
        assert_eq!(output.get(&[0]), 11.0);
        assert_eq!(output.get(&[1]), 12.0);
        assert_eq!(output.get(&[2]), 13.0);
        assert_eq!(output.get(&[3]), 14.0);
    }

    /// Test 2x2 matrix multiplication with linalg.matmul
    #[test]
    #[ignore = "linalg lowering pass not available in melior 0.25 - needs TN->LinAlg->LLVM pipeline"]
    fn test_memref_matmul_2x2() {
        // TODO: Implement when linalg lowering passes are available
    }

    /// Test with larger matrices (10x20 x 20x30)
    #[test]
    #[ignore = "linalg lowering pass not available in melior 0.25 - needs TN->LinAlg->LLVM pipeline"]
    fn test_memref_matmul_large() {
        // TODO: Implement when linalg lowering passes are available
    }
}
