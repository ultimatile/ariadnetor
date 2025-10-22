//! Basic MemRef execution test
//!
//! Tests the integration between TN-MLIR's Tensor/MemRefDescriptor types
//! and MLIR ExecutionEngine for JIT-compiled functions.

#[cfg(feature = "mlir")]
#[test]
fn test_memref_read_write() {
    use melior::{
        dialect::DialectRegistry,
        ir::{operation::OperationLike, Module},
        pass::PassManager,
        utility::{register_all_dialects, register_all_llvm_translations},
        Context, ExecutionEngine,
    };
    use arnet::{MemRefDescriptor, Tensor};

    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);

    // Create simple function: read from memref[0], add 10, write to output[0]
    let mut module = Module::parse(
        &context,
        r#"
        module {
            func.func @memref_add_ten(
                %input: memref<4xf64>,
                %output: memref<4xf64>
            ) attributes { llvm.emit_c_interface } {
                %c0 = arith.constant 0 : index
                %c10 = arith.constant 10.0 : f64

                %val = memref.load %input[%c0] : memref<4xf64>
                %result = arith.addf %val, %c10 : f64
                memref.store %result, %output[%c0] : memref<4xf64>

                return
            }
        }
        "#,
    )
    .expect("Failed to parse module");

    // Apply lowering passes
    let pass_manager = PassManager::new(&context);
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

    // Create tensors
    let mut input = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![4]);
    let mut output = Tensor::new(vec![4]);

    // Create MemRef descriptors
    let mut input_desc = MemRefDescriptor::<1>::from_tensor_mut(&mut input);
    let mut output_desc = MemRefDescriptor::<1>::from_tensor_mut(&mut output);

    // Invoke function using direct function pointer
    let func_ptr = engine.lookup("_mlir_ciface_memref_add_ten");
    assert!(!func_ptr.is_null(), "Function not found!");

    type FuncType = unsafe extern "C" fn(*mut MemRefDescriptor<1>, *mut MemRefDescriptor<1>);

    unsafe {
        let func: FuncType = std::mem::transmute(func_ptr);
        func(&mut input_desc, &mut output_desc);
    }
    assert_eq!(output.get(&[0]), 15.0);
}
