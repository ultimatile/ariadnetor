//! Minimal ExecutionEngine test without any TN dialect dependencies
//!
//! This test file does NOT import tn_mlir at all, to isolate ExecutionEngine issues.

#[cfg(feature = "mlir")]
#[test]
fn test_minimal_execution_engine() {
    use melior::{
        Context, ExecutionEngine,
        dialect::DialectRegistry,
        ir::Module,
        pass::PassManager,
        utility::{register_all_dialects, register_all_llvm_translations},
    };

    // Setup context
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);

    // Parse module
    let mut module = Module::parse(
        &context,
        r#"
        module {
            func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
                %res = arith.addi %arg0, %arg0 : i32
                return %res : i32
            }
        }
        "#,
    )
    .unwrap();

    // Apply passes
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(melior::pass::conversion::create_to_llvm());
    assert_eq!(pass_manager.run(&mut module), Ok(()));

    // Create ExecutionEngine
    println!("Creating ExecutionEngine (minimal test, no TN dialect)...");
    let engine = ExecutionEngine::new(&module, 2, &[], false);
    println!("ExecutionEngine created successfully!");

    // Invoke function
    let mut argument = 42;
    let mut result = -1;

    println!("Invoking function...");
    let invoke_result = unsafe {
        engine.invoke_packed(
            "add",
            &mut [
                &mut argument as *mut i32 as *mut (),
                &mut result as *mut i32 as *mut (),
            ],
        )
    };

    assert_eq!(invoke_result, Ok(()));
    assert_eq!(argument, 42);
    assert_eq!(result, 84);
    println!("Test passed - ExecutionEngine works without TN dialect!");
}
