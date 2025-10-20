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

    /// Apply lowering passes to convert SCF/MemRef operations to executable LLVM IR
    ///
    /// Pipeline: SCF → CF → LLVM IR
    ///
    /// # Arguments
    ///
    /// * `module` - Module containing SCF and MemRef operations
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of pass execution
    fn apply_lowering_passes(&self, module: &mut Module) -> Result<()> {
        let pass_manager = PassManager::new(self.context);

        // Convert SCF (Structured Control Flow) to Control Flow
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());

        // Convert to LLVM dialect
        pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
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
    /// Currently supports: matrix multiplication (ij,jk->ik)
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
        lhs: &mut Tensor,
        rhs: &mut Tensor,
    ) -> Result<Tensor> {
        use crate::einsum::EinsumExpr;
        use melior::ir::Module;

        // Step 1: Parse einsum expression
        let expr = EinsumExpr::parse(einsum_expr)?;

        // Step 2: Infer output shape
        let lhs_shape = lhs.shape_i64();
        let rhs_shape = rhs.shape_i64();
        let output_shape = expr.infer_output_shape(&lhs_shape, &rhs_shape)?;

        // Step 3: Verify this is matrix multiplication (currently the only supported pattern)
        if !expr.is_matrix_multiply() {
            anyhow::bail!(
                "Only matrix multiplication (ij,jk->ik) is currently supported, got: {}",
                einsum_expr
            );
        }

        // Step 4: Build module with matrix multiplication loops
        // Generate explicit loops instead of linalg.matmul since linalg lowering passes
        // aren't available in melior yet
        let m = lhs_shape[0];
        let k = lhs_shape[1];
        let n = rhs_shape[1];

        let mlir_source = format!(
            r#"
            module {{
                func.func @matmul(%arg0: memref<{}x{}xf64>, %arg1: memref<{}x{}xf64>, %arg2: memref<{}x{}xf64>) attributes {{llvm.emit_c_interface}} {{
                    %c0 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    %c_m = arith.constant {} : index
                    %c_k = arith.constant {} : index
                    %c_n = arith.constant {} : index
                    %zero = arith.constant 0.0 : f64

                    // Initialize output to zero
                    scf.for %i = %c0 to %c_m step %c1 {{
                        scf.for %j = %c0 to %c_n step %c1 {{
                            memref.store %zero, %arg2[%i, %j] : memref<{}x{}xf64>
                        }}
                    }}

                    // Compute C[i,j] = sum_k A[i,k] * B[k,j]
                    scf.for %i = %c0 to %c_m step %c1 {{
                        scf.for %j = %c0 to %c_n step %c1 {{
                            scf.for %k_idx = %c0 to %c_k step %c1 {{
                                %a = memref.load %arg0[%i, %k_idx] : memref<{}x{}xf64>
                                %b = memref.load %arg1[%k_idx, %j] : memref<{}x{}xf64>
                                %c_old = memref.load %arg2[%i, %j] : memref<{}x{}xf64>
                                %prod = arith.mulf %a, %b : f64
                                %c_new = arith.addf %c_old, %prod : f64
                                memref.store %c_new, %arg2[%i, %j] : memref<{}x{}xf64>
                            }}
                        }}
                    }}

                    return
                }}
            }}
            "#,
            m, k, k, n, m, n,
            m, k, n,
            m, n,
            m, k, k, n, m, n, m, n
        );

        let mut module = Module::parse(self.context, &mlir_source)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse LinAlg module"))?;

        // Verify module before lowering
        if !module.as_operation().verify() {
            anyhow::bail!("Generated module failed verification before lowering");
        }

        // Step 5: Apply lowering passes to LLVM
        self.apply_lowering_passes(&mut module)?;

        // Verify module after lowering
        if !module.as_operation().verify() {
            anyhow::bail!("Module verification failed after lowering");
        }

        // Step 6: Create execution engine
        let engine = self.create_execution_engine(&module);

        // Step 7: Execute with MemRef descriptors
        use crate::memref::MemRefDescriptor;

        // Create input MemRef descriptors (2D matrices)
        let lhs_desc = MemRefDescriptor::<2>::from_tensor_mut(lhs);
        let rhs_desc = MemRefDescriptor::<2>::from_tensor_mut(rhs);

        // Create output tensor and descriptor (2D matrix)
        let mut result_tensor = Tensor::new(output_shape.iter().map(|&s| s as usize).collect());
        let mut result_desc = MemRefDescriptor::<2>::from_tensor_mut(&mut result_tensor);

        // Look up the C interface wrapper function
        let wrapper_name = "_mlir_ciface_matmul";
        let func_ptr = engine.lookup(wrapper_name);

        // Call the function using direct function pointer
        // Function signature: (input1, input2, output) -> ()
        type MatmulFn = unsafe extern "C" fn(
            *const MemRefDescriptor<2>,
            *const MemRefDescriptor<2>,
            *mut MemRefDescriptor<2>
        );

        unsafe {
            let matmul_fn: MatmulFn = std::mem::transmute(func_ptr);
            matmul_fn(&lhs_desc, &rhs_desc, &mut result_desc);
        }

        // Step 8: Return result
        // The result_tensor is modified in place through the MemRefDescriptor
        Ok(result_tensor)
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
