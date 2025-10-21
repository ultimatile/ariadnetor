//! Basic usage example of TN-MLIR library
//!
//! This example demonstrates:
//! - Creating tensors
//! - Building operations
//! - JIT compilation (when implemented)

use tn_mlir::Tensor;
// Note: TNBuilder and TNJITCompiler are not yet implemented
// use tn_mlir::{TNBuilder, TNJITCompiler};

fn main() {
    println!("TN-MLIR Basic Usage Example");
    println!("============================\n");

    // Example 1: Create tensors
    println!("1. Creating tensors");
    let a = Tensor::new(vec![3, 4]);
    let b = Tensor::ones(vec![4, 5]);
    println!("  Tensor A: {}", a);
    println!("  Tensor B: {}", b);

    // Example 2: Tensor indexing
    println!("\n2. Tensor indexing");
    let mut tensor = Tensor::new(vec![2, 3]);
    tensor.set(&[0, 0], 1.0);
    tensor.set(&[0, 1], 2.0);
    tensor.set(&[1, 2], 3.0);
    println!("  tensor[0, 0] = {}", tensor.get(&[0, 0]));
    println!("  tensor[0, 1] = {}", tensor.get(&[0, 1]));
    println!("  tensor[1, 2] = {}", tensor.get(&[1, 2]));

    // Example 3: Builder API (not yet implemented)
    println!("\n3. Building operations (NOT YET IMPLEMENTED)");
    println!("  TNBuilder requires MLIR melior integration");
    println!("  Once implemented, operations will include:");
    println!("    - contract: tensor contractions (einsum)");
    println!("    - svd: singular value decomposition");
    println!("    - qr: QR decomposition");
    println!("    - transpose, reshape, truncate");
    // Note: TNBuilder::new() will panic until MLIR integration is complete
    // Future:
    // let builder = TNBuilder::new()?;
    // let result = builder.contract(&a.data(), &b.data(), "ij,jk->ik")?;

    // Example 4: JIT compilation (not yet implemented)
    println!("\n4. JIT Compilation (NOT YET IMPLEMENTED)");
    println!("  TNJITCompiler requires MLIR ExecutionEngine integration");
    println!("  Once implemented, you'll be able to:");
    println!("    - Compile einsum expressions to machine code");
    println!("    - Execute with dynamic tensor shapes");
    println!("    - Cache compiled functions for reuse");
    // Note: TNJITCompiler::new() will panic until MLIR integration is complete
    // Future:
    // let mut compiler = TNJITCompiler::new()?;
    // let c = compiler.compile_and_execute("ij,jk->ik", vec![a, b])?;

    println!("\n✓ Example completed successfully!");
    println!("\nNext steps:");
    println!("  1. Build the MLIR dialect with CMake");
    println!("  2. Integrate with melior for MLIR bindings");
    println!("  3. Implement JIT compilation pipeline");
}
