//! JIT Compiler for TN-Compute dialect
//!
//! Provides JIT compilation of tensor network operations with dynamic shape support.

use anyhow::Result;
use std::collections::HashMap;

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

    /// Generate MLIR IR for arbitrary einsum contraction patterns
    ///
    /// # Arguments
    ///
    /// * `expr` - Parsed einsum expression
    /// * `lhs_shape` - Shape of left-hand side tensor
    /// * `rhs_shape` - Shape of right-hand side tensor
    /// * `output_shape` - Shape of output tensor
    ///
    /// # Returns
    ///
    /// MLIR module source code as a string
    fn generate_einsum_mlir(
        &self,
        expr: &crate::einsum::EinsumExpr,
        lhs_shape: &[i64],
        rhs_shape: &[i64],
        output_shape: &[i64],
    ) -> Result<String> {
        use std::collections::HashMap;

        // Build index -> dimension mapping
        let mut index_dims: HashMap<char, i64> = HashMap::new();

        for (idx, &dim) in expr.lhs_indices().iter().zip(lhs_shape.iter()) {
            index_dims.insert(*idx, dim);
        }

        for (idx, &dim) in expr.rhs_indices().iter().zip(rhs_shape.iter()) {
            if let Some(&existing_dim) = index_dims.get(idx) {
                if existing_dim != dim {
                    anyhow::bail!(
                        "Dimension mismatch for index '{}': {} vs {}",
                        idx, existing_dim, dim
                    );
                }
            } else {
                index_dims.insert(*idx, dim);
            }
        }

        // Identify output and contracted indices
        let output_indices = expr.out_indices();
        let contracted_indices = expr.contracted_indices();

        // Generate memref type strings
        let lhs_type = self.generate_memref_type(lhs_shape);
        let rhs_type = self.generate_memref_type(rhs_shape);
        let output_type = self.generate_memref_type(output_shape);

        // Generate constant declarations for loop bounds
        let mut constants = String::from("    %c0 = arith.constant 0 : index\n");
        constants.push_str("    %c1 = arith.constant 1 : index\n");
        for (idx, &dim) in &index_dims {
            constants.push_str(&format!("    %c_{} = arith.constant {} : index\n", idx, dim));
        }
        constants.push_str("    %zero = arith.constant 0.0 : f64\n");

        // Generate initialization loops
        let init_loops = self.generate_nested_loops(
            output_indices,
            &index_dims,
            &format!("memref.store %zero, %arg2[{}] : {}",
                self.generate_index_list(output_indices),
                output_type)
        );

        // Generate computation loops
        let lhs_access = self.generate_index_list(expr.lhs_indices());
        let rhs_access = self.generate_index_list(expr.rhs_indices());
        let out_access = self.generate_index_list(output_indices);

        let compute_body = format!(
            "      %lhs_val = memref.load %arg0[{}] : {}\n\
             %rhs_val = memref.load %arg1[{}] : {}\n\
             %out_old = memref.load %arg2[{}] : {}\n\
             %prod = arith.mulf %lhs_val, %rhs_val : f64\n\
             %out_new = arith.addf %out_old, %prod : f64\n\
             memref.store %out_new, %arg2[{}] : {}",
            lhs_access, lhs_type,
            rhs_access, rhs_type,
            out_access, output_type,
            out_access, output_type
        );

        // Combine output and contracted indices for full loop nest
        let mut all_loop_indices = output_indices.to_vec();
        all_loop_indices.extend(&contracted_indices);

        let compute_loops = self.generate_nested_loops(
            &all_loop_indices,
            &index_dims,
            &compute_body
        );

        // Assemble the complete module
        let mlir_source = format!(
            r#"
            module {{
                func.func @einsum_contract(%arg0: {}, %arg1: {}, %arg2: {}) attributes {{llvm.emit_c_interface}} {{
{}
                    // Initialize output to zero
{}
                    // Compute tensor contraction
{}
                    return
                }}
            }}
            "#,
            lhs_type, rhs_type, output_type,
            constants,
            init_loops,
            compute_loops
        );

        Ok(mlir_source)
    }

    /// Generate memref type string from shape
    fn generate_memref_type(&self, shape: &[i64]) -> String {
        let dims = shape.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        format!("memref<{}xf64>", dims)
    }

    /// Generate index access list for memref
    fn generate_index_list(&self, indices: &[char]) -> String {
        indices.iter()
            .map(|idx| format!("%{}", idx))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Generate nested SCF for loops
    fn generate_nested_loops(
        &self,
        indices: &[char],
        _index_dims: &HashMap<char, i64>,
        body: &str,
    ) -> String {
        if indices.is_empty() {
            return format!("    {}\n", body);
        }

        let mut result = String::new();
        let indent = "    ";

        // Generate nested loops
        for (depth, &idx) in indices.iter().enumerate() {
            let loop_indent = indent.repeat(depth + 1);
            result.push_str(&format!(
                "{}scf.for %{} = %c0 to %c_{} step %c1 {{\n",
                loop_indent, idx, idx
            ));
        }

        // Add body with proper indentation
        let body_indent = indent.repeat(indices.len() + 1);
        for line in body.lines() {
            result.push_str(&format!("{}{}\n", body_indent, line.trim()));
        }

        // Close loops
        for depth in (0..indices.len()).rev() {
            let loop_indent = indent.repeat(depth + 1);
            result.push_str(&format!("{}}}\n", loop_indent));
        }

        result
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

        // Step 3: Build module with general einsum contraction loops
        let mlir_source = self.generate_einsum_mlir(&expr, &lhs_shape, &rhs_shape, &output_shape)?;

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
        let mut result_tensor = Tensor::new(output_shape.iter().map(|&s| s as usize).collect());

        // Look up the C interface wrapper function
        let wrapper_name = "_mlir_ciface_einsum_contract";
        let func_ptr = engine.lookup(wrapper_name);

        // Call function with appropriate rank-specific descriptors
        let lhs_rank = lhs_shape.len();
        let rhs_rank = rhs_shape.len();
        let out_rank = output_shape.len();

        // Dispatch based on tensor ranks
        self.execute_with_memref(
            func_ptr,
            lhs, rhs, &mut result_tensor,
            lhs_rank, rhs_rank, out_rank
        )?;

        Ok(result_tensor)
    }

    /// Execute JIT-compiled function with MemRef descriptors
    ///
    /// Dispatches to rank-specific implementations based on tensor ranks
    #[allow(clippy::too_many_arguments)]
    fn execute_with_memref(
        &self,
        func_ptr: *mut (),
        lhs: &mut Tensor,
        rhs: &mut Tensor,
        result: &mut Tensor,
        lhs_rank: usize,
        rhs_rank: usize,
        out_rank: usize,
    ) -> Result<()> {
        use crate::memref::MemRefDescriptor;

        // Macro to generate rank-specific execution code
        macro_rules! execute_rank_combination {
            ($lhs_rank:expr, $rhs_rank:expr, $out_rank:expr) => {{
                let lhs_desc = MemRefDescriptor::<$lhs_rank>::from_tensor_mut(lhs);
                let rhs_desc = MemRefDescriptor::<$rhs_rank>::from_tensor_mut(rhs);
                let mut out_desc = MemRefDescriptor::<$out_rank>::from_tensor_mut(result);

                type FnType = unsafe extern "C" fn(
                    *const MemRefDescriptor<$lhs_rank>,
                    *const MemRefDescriptor<$rhs_rank>,
                    *mut MemRefDescriptor<$out_rank>
                );

                unsafe {
                    let f: FnType = std::mem::transmute(func_ptr);
                    f(&lhs_desc, &rhs_desc, &mut out_desc);
                }
            }};
        }

        // Match on all rank combinations we support
        match (lhs_rank, rhs_rank, out_rank) {
            // Same rank cases (1D-10D)
            (1, 1, 1) => execute_rank_combination!(1, 1, 1),
            (2, 2, 2) => execute_rank_combination!(2, 2, 2),
            (3, 3, 3) => execute_rank_combination!(3, 3, 3),
            (4, 4, 4) => execute_rank_combination!(4, 4, 4),
            (5, 5, 5) => execute_rank_combination!(5, 5, 5),
            (6, 6, 6) => execute_rank_combination!(6, 6, 6),
            (7, 7, 7) => execute_rank_combination!(7, 7, 7),
            (8, 8, 8) => execute_rank_combination!(8, 8, 8),
            (9, 9, 9) => execute_rank_combination!(9, 9, 9),
            (10, 10, 10) => execute_rank_combination!(10, 10, 10),

            // Mixed rank: output lower than inputs (common contractions)
            (2, 2, 1) => execute_rank_combination!(2, 2, 1),
            (3, 3, 1) => execute_rank_combination!(3, 3, 1),
            (3, 3, 2) => execute_rank_combination!(3, 3, 2),
            (4, 4, 1) => execute_rank_combination!(4, 4, 1),
            (4, 4, 2) => execute_rank_combination!(4, 4, 2),
            (4, 4, 3) => execute_rank_combination!(4, 4, 3),
            (5, 5, 1) => execute_rank_combination!(5, 5, 1),
            (5, 5, 2) => execute_rank_combination!(5, 5, 2),
            (5, 5, 3) => execute_rank_combination!(5, 5, 3),
            (5, 5, 4) => execute_rank_combination!(5, 5, 4),
            (6, 6, 1) => execute_rank_combination!(6, 6, 1),
            (6, 6, 2) => execute_rank_combination!(6, 6, 2),
            (6, 6, 3) => execute_rank_combination!(6, 6, 3),
            (6, 6, 4) => execute_rank_combination!(6, 6, 4),
            (6, 6, 5) => execute_rank_combination!(6, 6, 5),
            (7, 7, 1) => execute_rank_combination!(7, 7, 1),
            (7, 7, 2) => execute_rank_combination!(7, 7, 2),
            (7, 7, 3) => execute_rank_combination!(7, 7, 3),
            (7, 7, 4) => execute_rank_combination!(7, 7, 4),
            (7, 7, 5) => execute_rank_combination!(7, 7, 5),
            (7, 7, 6) => execute_rank_combination!(7, 7, 6),
            (8, 8, 1) => execute_rank_combination!(8, 8, 1),
            (8, 8, 2) => execute_rank_combination!(8, 8, 2),
            (8, 8, 3) => execute_rank_combination!(8, 8, 3),
            (8, 8, 4) => execute_rank_combination!(8, 8, 4),
            (8, 8, 5) => execute_rank_combination!(8, 8, 5),
            (8, 8, 6) => execute_rank_combination!(8, 8, 6),
            (8, 8, 7) => execute_rank_combination!(8, 8, 7),
            (9, 9, 1) => execute_rank_combination!(9, 9, 1),
            (9, 9, 2) => execute_rank_combination!(9, 9, 2),
            (9, 9, 3) => execute_rank_combination!(9, 9, 3),
            (9, 9, 4) => execute_rank_combination!(9, 9, 4),
            (9, 9, 5) => execute_rank_combination!(9, 9, 5),
            (9, 9, 6) => execute_rank_combination!(9, 9, 6),
            (9, 9, 7) => execute_rank_combination!(9, 9, 7),
            (9, 9, 8) => execute_rank_combination!(9, 9, 8),
            (10, 10, 1) => execute_rank_combination!(10, 10, 1),
            (10, 10, 2) => execute_rank_combination!(10, 10, 2),
            (10, 10, 3) => execute_rank_combination!(10, 10, 3),
            (10, 10, 4) => execute_rank_combination!(10, 10, 4),
            (10, 10, 5) => execute_rank_combination!(10, 10, 5),
            (10, 10, 6) => execute_rank_combination!(10, 10, 6),
            (10, 10, 7) => execute_rank_combination!(10, 10, 7),
            (10, 10, 8) => execute_rank_combination!(10, 10, 8),
            (10, 10, 9) => execute_rank_combination!(10, 10, 9),

            // Mixed rank: different input ranks (3D x 4D -> 5D, etc.)
            (2, 3, 3) => execute_rank_combination!(2, 3, 3),
            (2, 3, 4) => execute_rank_combination!(2, 3, 4),
            (2, 4, 4) => execute_rank_combination!(2, 4, 4),
            (2, 4, 5) => execute_rank_combination!(2, 4, 5),
            (2, 5, 5) => execute_rank_combination!(2, 5, 5),
            (2, 5, 6) => execute_rank_combination!(2, 5, 6),
            (3, 2, 3) => execute_rank_combination!(3, 2, 3),
            (3, 2, 4) => execute_rank_combination!(3, 2, 4),
            (3, 4, 4) => execute_rank_combination!(3, 4, 4),
            (3, 4, 5) => execute_rank_combination!(3, 4, 5),
            (3, 4, 6) => execute_rank_combination!(3, 4, 6),
            (3, 5, 5) => execute_rank_combination!(3, 5, 5),
            (3, 5, 6) => execute_rank_combination!(3, 5, 6),
            (3, 5, 7) => execute_rank_combination!(3, 5, 7),
            (4, 2, 4) => execute_rank_combination!(4, 2, 4),
            (4, 2, 5) => execute_rank_combination!(4, 2, 5),
            (4, 3, 4) => execute_rank_combination!(4, 3, 4),
            (4, 3, 5) => execute_rank_combination!(4, 3, 5),
            (4, 3, 6) => execute_rank_combination!(4, 3, 6),
            (4, 5, 5) => execute_rank_combination!(4, 5, 5),
            (4, 5, 6) => execute_rank_combination!(4, 5, 6),
            (4, 5, 7) => execute_rank_combination!(4, 5, 7),
            (4, 5, 8) => execute_rank_combination!(4, 5, 8),
            (5, 2, 5) => execute_rank_combination!(5, 2, 5),
            (5, 2, 6) => execute_rank_combination!(5, 2, 6),
            (5, 3, 5) => execute_rank_combination!(5, 3, 5),
            (5, 3, 6) => execute_rank_combination!(5, 3, 6),
            (5, 3, 7) => execute_rank_combination!(5, 3, 7),
            (5, 4, 5) => execute_rank_combination!(5, 4, 5),
            (5, 4, 6) => execute_rank_combination!(5, 4, 6),
            (5, 4, 7) => execute_rank_combination!(5, 4, 7),
            (5, 4, 8) => execute_rank_combination!(5, 4, 8),
            (5, 6, 6) => execute_rank_combination!(5, 6, 6),
            (5, 6, 7) => execute_rank_combination!(5, 6, 7),
            (5, 6, 8) => execute_rank_combination!(5, 6, 8),
            (5, 6, 9) => execute_rank_combination!(5, 6, 9),
            (6, 2, 6) => execute_rank_combination!(6, 2, 6),
            (6, 2, 7) => execute_rank_combination!(6, 2, 7),
            (6, 3, 6) => execute_rank_combination!(6, 3, 6),
            (6, 3, 7) => execute_rank_combination!(6, 3, 7),
            (6, 3, 8) => execute_rank_combination!(6, 3, 8),
            (6, 4, 6) => execute_rank_combination!(6, 4, 6),
            (6, 4, 7) => execute_rank_combination!(6, 4, 7),
            (6, 4, 8) => execute_rank_combination!(6, 4, 8),
            (6, 4, 9) => execute_rank_combination!(6, 4, 9),
            (6, 5, 6) => execute_rank_combination!(6, 5, 6),
            (6, 5, 7) => execute_rank_combination!(6, 5, 7),
            (6, 5, 8) => execute_rank_combination!(6, 5, 8),
            (6, 5, 9) => execute_rank_combination!(6, 5, 9),
            (6, 5, 10) => execute_rank_combination!(6, 5, 10),

            _ => {
                anyhow::bail!(
                    "Unsupported tensor rank combination: lhs={}, rhs={}, output={}. \
                     This rank combination has not been implemented yet. \
                     Supported patterns: same ranks (1-10), contractions reducing rank up to 10D, \
                     and many mixed-rank operations. If you encounter this error, the specific \
                     combination can be added - the underlying infrastructure supports arbitrary ranks.",
                    lhs_rank, rhs_rank, out_rank
                );
            }
        }

        Ok(())
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
