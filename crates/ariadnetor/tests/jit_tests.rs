//! Integration tests for JIT Compiler
//!
//! These tests verify JIT compilation and execution of MLIR modules.
//!
//! NOTE: JIT compiler is not yet implemented. These tests are placeholders
//! for future implementation.

#[cfg(feature = "mlir")]
#[cfg(feature = "jit")] // JIT compiler not yet implemented
mod jit_tests {
    use melior::{
        Context,
        dialect::DialectRegistry,
        ir::operation::OperationLike,
        utility::{register_all_dialects, register_all_llvm_translations},
    };
    // use arnet::JITCompiler; // TODO: Implement JIT compiler

    fn setup_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        // Register LLVM translations for ExecutionEngine support
        register_all_llvm_translations(&context);

        context
    }

    #[test]
    fn test_compiler_creation() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Verify compiler was created with default optimization level
        assert_eq!(compiler.optimization_level(), 2);
    }

    #[test]
    fn test_optimization_level() {
        let context = setup_context();

        // Test setting optimization level
        let compiler = TNJITCompiler::new(&context)
            .with_optimization_level(3);
        assert_eq!(compiler.optimization_level(), 3);

        // Test clamping to max level 3
        let compiler_clamped = TNJITCompiler::new(&context)
            .with_optimization_level(10);
        assert_eq!(compiler_clamped.optimization_level(), 3);
    }

    #[test]
    fn test_compile_simple_function() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Simple function that doubles an i32 value
        let mlir_source = r#"
            module {
                func.func @double(%arg0: i32) -> i32 {
                    %result = arith.addi %arg0, %arg0 : i32
                    return %result : i32
                }
            }
        "#;

        let result = compiler.compile_module(mlir_source);
        assert!(result.is_ok(), "Failed to compile simple function: {:?}", result.err());
    }

    #[test]
    fn test_compile_and_lookup_function() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Simple function that adds two i32 values
        let mlir_source = r#"
            module {
                func.func @add(%arg0: i32, %arg1: i32) -> i32 {
                    %result = arith.addi %arg0, %arg1 : i32
                    return %result : i32
                }
            }
        "#;

        let engine = compiler.compile_module(mlir_source)
            .expect("Failed to compile module");

        // Lookup the function (returns raw pointer)
        let add_ptr = engine.lookup("add");

        // Verify we got a non-null pointer
        assert!(!add_ptr.is_null(), "Function lookup returned null pointer");

        // Note: Actually invoking the function requires unsafe code and proper ABI handling
        // For now, we just verify the function can be looked up after compilation
    }

    #[test]
    fn test_compile_float_function() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Function that multiplies two f64 values
        let mlir_source = r#"
            module {
                func.func @multiply(%arg0: f64, %arg1: f64) -> f64 {
                    %result = arith.mulf %arg0, %arg1 : f64
                    return %result : f64
                }
            }
        "#;

        let result = compiler.compile_module(mlir_source);
        assert!(result.is_ok(), "Failed to compile float function: {:?}", result.err());
    }

    #[test]
    fn test_compile_invalid_module() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Invalid MLIR source (syntax error)
        let mlir_source = r#"
            module {
                this is not valid MLIR
            }
        "#;

        let result = compiler.compile_module(mlir_source);
        assert!(result.is_err(), "Should fail to compile invalid module");
    }

    #[test]
    fn test_compile_with_optimization() {
        let context = setup_context();

        // Test with different optimization levels
        for opt_level in 0..=3 {
            let compiler = TNJITCompiler::new(&context)
                .with_optimization_level(opt_level);

            let mlir_source = r#"
                module {
                    func.func @identity(%arg0: i64) -> i64 {
                        return %arg0 : i64
                    }
                }
            "#;

            let result = compiler.compile_module(mlir_source);
            assert!(
                result.is_ok(),
                "Failed to compile with optimization level {}: {:?}",
                opt_level,
                result.err()
            );
        }
    }

    #[test]
    fn test_compile_function_with_control_flow() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Function with conditional logic
        let mlir_source = r#"
            module {
                func.func @max(%arg0: i32, %arg1: i32) -> i32 {
                    %cmp = arith.cmpi sgt, %arg0, %arg1 : i32
                    %result = arith.select %cmp, %arg0, %arg1 : i32
                    return %result : i32
                }
            }
        "#;

        let result = compiler.compile_module(mlir_source);
        assert!(result.is_ok(), "Failed to compile function with control flow: {:?}", result.err());
    }

    #[test]
    #[ignore = "Requires tensor runtime support"]
    fn test_compile_tensor_function() {
        let context = setup_context();
        let compiler = TNJITCompiler::new(&context);

        // Simple tensor operation (this may fail without proper runtime setup)
        let mlir_source = r#"
            module {
                func.func @tensor_identity(%arg0: tensor<10xf64>) -> tensor<10xf64> {
                    return %arg0 : tensor<10xf64>
                }
            }
        "#;

        let result = compiler.compile_module(mlir_source);
        // Note: This may fail without memref lowering and runtime library setup
        // Mark as ignored for now
        if result.is_err() {
            println!("Expected: Tensor function requires additional runtime setup");
        }
    }

    #[test]
    fn test_module_parsing() {
        use melior::ir::Module;

        let context = setup_context();

        // Test that we can parse a valid MLIR module
        let mlir_source = r#"
            module {
                func.func @test(%arg0: i32) -> i32 {
                    return %arg0 : i32
                }
            }
        "#;

        let module_result = Module::parse(&context, mlir_source);
        assert!(module_result.is_some(), "Failed to parse valid MLIR module");

        let module = module_result.unwrap();
        assert!(module.as_operation().verify(), "Module verification failed");
    }

    #[test]
    fn test_lowering_passes() {
        use melior::{
            ir::Module,
            pass::{self, PassManager},
        };

        let context = setup_context();

        // Create a simple module with arith operations
        let mlir_source = r#"
            module {
                func.func @add(%arg0: i32, %arg1: i32) -> i32 {
                    %result = arith.addi %arg0, %arg1 : i32
                    return %result : i32
                }
            }
        "#;

        let mut module = Module::parse(&context, mlir_source)
            .expect("Failed to parse module");

        // Create pass manager and apply lowering passes
        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

        // Run passes
        let result = pass_manager.run(&mut module);
        assert!(result.is_ok(), "Failed to run lowering passes");

        // Verify the lowered module
        assert!(module.as_operation().verify(), "Module verification failed after lowering");
    }

    #[test]
    fn test_optimization_passes_on_module() {
        use melior::{
            ir::Module,
            pass::{self, PassManager},
        };

        let context = setup_context();

        let mlir_source = r#"
            module {
                func.func @identity(%arg0: i64) -> i64 {
                    return %arg0 : i64
                }
            }
        "#;

        let mut module = Module::parse(&context, mlir_source)
            .expect("Failed to parse module");

        // Apply optimization and lowering passes
        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

        let result = pass_manager.run(&mut module);
        assert!(result.is_ok(), "Failed to run optimization passes");
        assert!(module.as_operation().verify(), "Module verification failed");
    }
}
