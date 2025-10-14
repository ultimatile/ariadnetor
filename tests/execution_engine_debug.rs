//! Debug tests for ExecutionEngine segfault investigation
//!
//! These tests replicate melior's working examples to identify differences.

#[cfg(feature = "mlir")]
mod execution_engine_debug {
    use melior::{
        Context,
        ExecutionEngine,
        dialect::DialectRegistry,
        ir::Module,
        pass::{self, PassManager},
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

    /// Test 1: Exact replica of melior's invoke_packed test
    #[test]
    fn test_melior_exact_replica() {
        let context = setup_context();

        // EXACTLY as melior's test
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

        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));

        // This is where segfault might occur
        println!("Creating ExecutionEngine...");
        let engine = ExecutionEngine::new(&module, 2, &[], false);
        println!("ExecutionEngine created successfully!");

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
        println!("Test passed!");
    }

    /// Test 2: Without llvm.emit_c_interface attribute
    #[test]
    fn test_without_c_interface_attribute() {
        let context = setup_context();

        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @add(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
            }
            "#,
        )
        .unwrap();

        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));

        println!("Creating ExecutionEngine (without c_interface)...");
        let _engine = ExecutionEngine::new(&module, 2, &[], false);
        println!("ExecutionEngine created successfully!");
    }

    /// Test 3: Using our multi-pass pipeline with llvm.emit_c_interface
    #[test]
    fn test_multi_pass_with_c_interface() {
        let context = setup_context();

        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @add(%arg0 : i32, %arg1 : i32) -> i32 attributes { llvm.emit_c_interface } {
                    %result = arith.addi %arg0, %arg1 : i32
                    return %result : i32
                }
            }
            "#,
        )
        .unwrap();

        // Our original pass pipeline
        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

        assert_eq!(pass_manager.run(&mut module), Ok(()));

        println!("Creating ExecutionEngine (multi-pass)...");
        let _engine = ExecutionEngine::new(&module, 2, &[], false);
        println!("ExecutionEngine created successfully!");
    }

    /// Test 4: Minimal module
    #[test]
    fn test_minimal_module() {
        let context = setup_context();

        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @identity(%arg0 : i64) -> i64 attributes { llvm.emit_c_interface } {
                    return %arg0 : i64
                }
            }
            "#,
        )
        .unwrap();

        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));

        println!("Creating ExecutionEngine (minimal)...");
        let _engine = ExecutionEngine::new(&module, 2, &[], false);
        println!("ExecutionEngine created successfully!");
    }

    /// Test 5: Test with float operations
    #[test]
    fn test_float_operations() {
        let context = setup_context();

        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @multiply(%arg0 : f64, %arg1 : f64) -> f64 attributes { llvm.emit_c_interface } {
                    %result = arith.mulf %arg0, %arg1 : f64
                    return %result : f64
                }
            }
            "#,
        )
        .unwrap();

        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));

        println!("Creating ExecutionEngine (float ops)...");
        let _engine = ExecutionEngine::new(&module, 2, &[], false);
        println!("ExecutionEngine created successfully!");
    }

    /// Test 6: Dump to object file (doesn't require actual execution)
    #[test]
    fn test_dump_to_object_file() {
        let context = setup_context();

        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @test(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
            }
            "#,
        )
        .unwrap();

        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));

        println!("Creating ExecutionEngine (with object dump)...");
        let engine = ExecutionEngine::new(&module, 2, &[], true);
        println!("Dumping to object file...");

        // Create temp directory if needed
        std::fs::create_dir_all("/tmp/tn-mlir").ok();
        engine.dump_to_object_file("/tmp/tn-mlir/test.o");
        println!("Object file dumped successfully!");
    }
}
