//! JIT Compiler for TN-Compute dialect
//!
//! Provides JIT compilation of tensor network operations with dynamic shape support.

use anyhow::Result;

#[cfg(feature = "mlir")]
use crate::tensor::Tensor;

#[cfg(feature = "mlir")]
use melior::{
    Context,
    ExecutionEngine,
    ir::{Module, operation::OperationLike},
    pass::{self, PassManager},
};

/// JIT Compiler for tensor network operations
#[cfg(feature = "mlir")]
pub struct TNJITCompiler<'c> {
    context: &'c Context,
    optimization_level: usize,
}

#[cfg(not(feature = "mlir"))]
pub struct TNJITCompiler {
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(feature = "mlir")]
impl<'c> TNJITCompiler<'c> {
    /// Create a new JIT compiler instance
    ///
    /// # Arguments
    ///
    /// * `context` - MLIR context for compilation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use melior::Context;
    /// use tn_mlir::TNJITCompiler;
    ///
    /// let context = Context::new();
    /// let compiler = TNJITCompiler::new(&context);
    /// ```
    pub fn new(context: &'c Context) -> Self {
        Self {
            context,
            optimization_level: 2,
        }
    }

    /// Set optimization level (0-3)
    pub fn with_optimization_level(mut self, level: usize) -> Self {
        self.optimization_level = level.min(3);
        self
    }

    /// Get the current optimization level
    pub fn optimization_level(&self) -> usize {
        self.optimization_level
    }

    /// Apply lowering passes to convert TN operations to executable code
    ///
    /// Pipeline: TN → LinAlg → LLVM IR
    ///
    /// # Arguments
    ///
    /// * `module` - Module containing TN operations
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of pass execution
    fn apply_lowering_passes(&self, module: &mut Module) -> Result<()> {
        let pass_manager = PassManager::new(self.context);

        // TODO: Add TN → LinAlg conversion pass when implemented
        // For now, we assume the module already contains LinAlg or lower-level IR

        // Convert LinAlg to Standard dialect
        // pass_manager.add_pass(pass::conversion::create_linalg_to_standard());

        // Convert Standard/Arith to LLVM
        pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

        pass_manager
            .run(module)
            .map_err(|_| anyhow::anyhow!("Failed to run lowering passes"))?;

        Ok(())
    }

    /// Create an execution engine for a module
    ///
    /// # Arguments
    ///
    /// * `module` - Lowered module ready for execution
    ///
    /// # Returns
    ///
    /// Execution engine instance
    fn create_execution_engine(&self, module: &Module) -> ExecutionEngine {
        ExecutionEngine::new(
            module,
            self.optimization_level,
            &[],  // No additional shared libraries
            false, // Disable object dump
        )
    }

    /// Compile a module from MLIR source
    ///
    /// This is a low-level method for advanced users.
    ///
    /// # Arguments
    ///
    /// * `mlir_source` - MLIR textual representation
    ///
    /// # Returns
    ///
    /// Compiled module ready for execution
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let source = r#"
    ///     module {
    ///         func.func @add(%arg0: i32) -> i32 {
    ///             %res = arith.addi %arg0, %arg0 : i32
    ///             return %res : i32
    ///         }
    ///     }
    /// "#;
    /// let engine = compiler.compile_module(source)?;
    /// ```
    pub fn compile_module(&self, mlir_source: &str) -> Result<ExecutionEngine> {
        // Parse the module
        let mut module = Module::parse(self.context, mlir_source)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse MLIR module"))?;

        // Apply lowering passes
        self.apply_lowering_passes(&mut module)?;

        // Verify the module
        if !module.as_operation().verify() {
            anyhow::bail!("Module verification failed after lowering");
        }

        // Create execution engine
        Ok(self.create_execution_engine(&module))
    }

    /// Compile and execute a tensor contraction expression
    ///
    /// This provides a high-level interface for end-to-end tensor operations:
    /// parsing einsum notation, generating IR, JIT compiling, and executing
    /// with real tensor data.
    ///
    /// # Arguments
    ///
    /// * `einsum_expr` - Einsum notation (e.g., "ij,jk->ik")
    /// * `lhs` - Left-hand side input tensor
    /// * `rhs` - Right-hand side input tensor
    ///
    /// # Returns
    ///
    /// Result tensor from the computation
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Einsum expression is invalid
    /// - Tensor shapes are incompatible
    /// - IR generation fails
    /// - Compilation fails
    /// - Execution fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use melior::Context;
    /// use tn_mlir::{TNJITCompiler, Tensor};
    ///
    /// let context = Context::new();
    /// let compiler = TNJITCompiler::new(&context);
    ///
    /// let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    ///
    /// let c = compiler.compile_and_execute("ij,jk->ik", &a, &b)?;
    /// // c = a @ b (matrix multiplication)
    /// ```
    pub fn compile_and_execute(
        &self,
        einsum_expr: &str,
        lhs: &Tensor,
        rhs: &Tensor,
    ) -> Result<Tensor> {
        use crate::einsum::EinsumExpr;
        use melior::ir::{
            Module, Location, Block, BlockLike, RegionLike,
            r#type::{Type, RankedTensorType, FunctionType},
            operation::OperationBuilder,
            attribute::StringAttribute,
            Identifier,
        };
        use melior::ir::attribute::TypeAttribute;

        // Step 1: Parse einsum expression
        let expr = EinsumExpr::parse(einsum_expr)?;

        // Step 2: Infer output shape
        let lhs_shape = lhs.shape_i64();
        let rhs_shape = rhs.shape_i64();
        let output_shape = expr.infer_output_shape(&lhs_shape, &rhs_shape)?;

        // Step 3: Build module with wrapper function
        let location = Location::unknown(self.context);
        let module = Module::new(location);

        let f64_type = Type::float64(self.context);
        let lhs_type = RankedTensorType::new(
            &lhs_shape.iter().map(|&s| s as u64).collect::<Vec<_>>(),
            f64_type,
            None
        );
        let rhs_type = RankedTensorType::new(
            &rhs_shape.iter().map(|&s| s as u64).collect::<Vec<_>>(),
            f64_type,
            None
        );
        let result_type = RankedTensorType::new(
            &output_shape.iter().map(|&s| s as u64).collect::<Vec<_>>(),
            f64_type,
            None
        );

        // Create function type
        let func_type = FunctionType::new(
            self.context,
            &[lhs_type.into(), rhs_type.into()],
            &[result_type.into()]
        );

        // Build function
        let func_name = StringAttribute::new(self.context, "main");
        let func_name_id = Identifier::new(self.context, "sym_name");
        let func_type_attr = TypeAttribute::new(func_type.into());
        let func_type_id = Identifier::new(self.context, "function_type");

        // Allow unregistered dialects (TODO: properly register TN dialect)
        unsafe {
            mlir_sys::mlirContextSetAllowUnregisteredDialects(self.context.to_raw(), true);
        }

        let region = {
            let region = melior::ir::Region::new();
            let block = Block::new(&[
                (lhs_type.into(), location),
                (rhs_type.into(), location),
            ]);

            let lhs_val = block.argument(0)?.into();
            let rhs_val = block.argument(1)?.into();

            // Build TN contract operation directly (not using TNBuilder's module)
            let lhs_indices: String = expr.lhs_indices().iter().collect();
            let rhs_indices: String = expr.rhs_indices().iter().collect();
            let out_indices: String = expr.out_indices().iter().collect();
            let indices_str = format!("{},{}->{}", lhs_indices, rhs_indices, out_indices);

            let indices_attr = StringAttribute::new(self.context, &indices_str);
            let indices_id = Identifier::new(self.context, "indices");

            let contract_op = OperationBuilder::new("tn.contract", location)
                .add_operands(&[lhs_val, rhs_val])
                .add_attributes(&[(indices_id, indices_attr.into())])
                .add_results(&[result_type.into()])
                .build()?;

            // IMPORTANT: Append operation to block FIRST, then get result
            let contract_ref = block.append_operation(contract_op);
            let result = contract_ref.result(0)?.into();

            // Add return
            let return_op = OperationBuilder::new("func.return", location)
                .add_operands(&[result])
                .build()?;

            block.append_operation(return_op);
            region.append_block(block);
            region
        };

        let func_op = OperationBuilder::new("func.func", location)
            .add_attributes(&[
                (func_name_id, func_name.into()),
                (func_type_id, func_type_attr.into()),
            ])
            .add_regions([region])
            .build()?;

        module.body().append_operation(func_op);

        // Verify module
        if !module.as_operation().verify() {
            anyhow::bail!("Generated module failed verification");
        }

        // Step 4: Apply lowering passes
        // Note: Lowering to LLVM is skipped for now due to stability issues
        // The TN IR generation is complete and verified

        // For now, return a placeholder indicating the pipeline is incomplete
        let output_size: usize = output_shape.iter().map(|&s| s as usize).product();
        let _result_tensor = Tensor::new(output_shape.iter().map(|&s| s as usize).collect());

        anyhow::bail!(
            "JIT execution with tensor data not yet fully implemented. \
             Module generation successful. Expected output shape: {:?}, size: {}. \
             ExecutionEngine invocation with MemRef descriptors requires additional FFI work.",
            output_shape, output_size
        )
    }
}

// Non-mlir stub implementation
#[cfg(not(feature = "mlir"))]
impl TNJITCompiler {
    pub fn new() -> Result<Self> {
        anyhow::bail!("TNJITCompiler requires 'mlir' feature to be enabled")
    }

    pub fn compile_module(&self, _mlir_source: &str) -> Result<()> {
        anyhow::bail!("TNJITCompiler requires 'mlir' feature to be enabled")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "mlir")]
    fn test_compiler_creation() {
        use melior::Context;

        let context = Context::new();
        let compiler = TNJITCompiler::new(&context);

        // Verify compiler was created with correct settings
        assert_eq!(compiler.optimization_level, 2);
    }

    #[test]
    #[cfg(feature = "mlir")]
    fn test_optimization_level() {
        use melior::Context;

        let context = Context::new();
        let compiler = TNJITCompiler::new(&context).with_optimization_level(3);

        assert_eq!(compiler.optimization_level, 3);

        // Test clamping
        let compiler2 = TNJITCompiler::new(&context).with_optimization_level(10);
        assert_eq!(compiler2.optimization_level, 3);
    }

    #[test]
    #[cfg(not(feature = "mlir"))]
    fn test_compiler_requires_feature() {
        let result = TNJITCompiler::new();
        assert!(result.is_err());
    }
}
