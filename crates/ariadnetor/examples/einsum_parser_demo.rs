//! Demonstration of the Einsum Parser
//!
//! This example shows how to use the `EinsumExpr` parser to parse
//! and validate Einstein summation notation.
//!
//! Run with:
//!   cargo run --example einsum_parser_demo

use arnet::EinsumExpr;

fn main() {
    println!("=== Einsum Parser Demo ===\n");

    // Example 1: Matrix multiplication
    println!("1. Matrix Multiplication: ij,jk->ik");
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    println!("   LHS indices: {:?}", expr.lhs_indices());
    println!("   RHS indices: {:?}", expr.rhs_indices());
    println!("   Output indices: {:?}", expr.out_indices());
    println!("   Contracted indices: {:?}", expr.contracted_indices());
    println!("   Is matrix multiply: {}", expr.is_matrix_multiply());

    // Infer output shape
    let lhs_shape = vec![10, 20];
    let rhs_shape = vec![20, 30];
    let output_shape = expr.infer_output_shape(&lhs_shape, &rhs_shape).unwrap();
    println!("   Input shapes: {:?} x {:?}", lhs_shape, rhs_shape);
    println!("   Output shape: {:?}\n", output_shape);

    // Example 2: Higher-dimensional contraction
    println!("2. Tensor Contraction: ijk,jkl->il");
    let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    println!("   LHS indices: {:?}", expr.lhs_indices());
    println!("   RHS indices: {:?}", expr.rhs_indices());
    println!("   Output indices: {:?}", expr.out_indices());
    println!("   Contracted indices: {:?}", expr.contracted_indices());

    let lhs_shape = vec![5, 10, 15];
    let rhs_shape = vec![10, 15, 20];
    let output_shape = expr.infer_output_shape(&lhs_shape, &rhs_shape).unwrap();
    println!("   Input shapes: {:?} x {:?}", lhs_shape, rhs_shape);
    println!("   Output shape: {:?}\n", output_shape);

    // Example 3: Element-wise multiplication
    println!("3. Element-wise: ij,ij->ij");
    let expr = EinsumExpr::parse("ij,ij->ij").unwrap();
    println!("   LHS indices: {:?}", expr.lhs_indices());
    println!("   RHS indices: {:?}", expr.rhs_indices());
    println!("   Output indices: {:?}", expr.out_indices());
    println!("   Contracted indices: {:?}", expr.contracted_indices());
    println!("   Is matrix multiply: {}\n", expr.is_matrix_multiply());

    // Example 4: Error handling - invalid notation
    println!("4. Error Handling:");
    match EinsumExpr::parse("ij,jk->im") {
        Ok(_) => println!("   Unexpected success!"),
        Err(e) => println!("   ✓ Caught error: {}", e),
    }

    match EinsumExpr::parse("invalid") {
        Ok(_) => println!("   Unexpected success!"),
        Err(e) => println!("   ✓ Caught error: {}", e),
    }

    println!("\n=== Demo Complete ===");
}
