//! Demonstration of the Einsum Parser
//!
//! This example shows how to use the `EinsumExpr` parser to parse
//! and validate Einstein summation notation.
//!
//! Run with:
//!   cargo run --example einsum_parser_demo

use arnet::EinsumExpr;

/// Format u8 index slice as chars for display
fn fmt_indices(indices: &[u8]) -> Vec<char> {
    indices.iter().map(|&b| b as char).collect()
}

fn main() {
    println!("=== Einsum Parser Demo ===\n");

    // Example 1: Matrix multiplication
    println!("1. Matrix Multiplication: ij,jk->ik");
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    println!("   LHS indices: {:?}", fmt_indices(expr.lhs_indices()));
    println!("   RHS indices: {:?}", fmt_indices(expr.rhs_indices()));
    println!("   Output indices: {:?}", fmt_indices(expr.out_indices()));
    println!(
        "   Contracted indices: {:?}",
        fmt_indices(&expr.contracted_indices())
    );
    println!("   Is matrix multiply: {}", expr.is_matrix_multiply());

    let output_shape = expr.infer_output_shape(&[&[10, 20], &[20, 30]]).unwrap();
    println!("   Input shapes: [10, 20] x [20, 30]");
    println!("   Output shape: {:?}\n", output_shape);

    // Example 2: Higher-dimensional contraction
    println!("2. Tensor Contraction: ijk,jkl->il");
    let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    println!("   LHS indices: {:?}", fmt_indices(expr.lhs_indices()));
    println!("   RHS indices: {:?}", fmt_indices(expr.rhs_indices()));
    println!("   Output indices: {:?}", fmt_indices(expr.out_indices()));
    println!(
        "   Contracted indices: {:?}",
        fmt_indices(&expr.contracted_indices())
    );

    let output_shape = expr
        .infer_output_shape(&[&[5, 10, 15], &[10, 15, 20]])
        .unwrap();
    println!("   Input shapes: [5, 10, 15] x [10, 15, 20]");
    println!("   Output shape: {:?}\n", output_shape);

    // Example 3: Element-wise multiplication
    println!("3. Element-wise: ij,ij->ij");
    let expr = EinsumExpr::parse("ij,ij->ij").unwrap();
    println!("   LHS indices: {:?}", fmt_indices(expr.lhs_indices()));
    println!("   RHS indices: {:?}", fmt_indices(expr.rhs_indices()));
    println!("   Output indices: {:?}", fmt_indices(expr.out_indices()));
    println!(
        "   Contracted indices: {:?}",
        fmt_indices(&expr.contracted_indices())
    );
    println!("   Is matrix multiply: {}\n", expr.is_matrix_multiply());

    // Example 4: Implicit output inference
    println!("4. Implicit Output: ij,jk (no ->)");
    let expr = EinsumExpr::parse("ij,jk").unwrap();
    println!(
        "   Inferred output: {:?}",
        fmt_indices(expr.out_indices())
    );
    println!("   Num inputs: {}\n", expr.num_inputs());

    // Example 5: Single tensor trace
    println!("5. Trace: ii->");
    let expr = EinsumExpr::parse("ii->").unwrap();
    println!("   Num inputs: {}", expr.num_inputs());
    println!("   Indices: {:?}", fmt_indices(expr.lhs_indices()));
    println!("   Output: {:?}\n", fmt_indices(expr.out_indices()));

    // Example 6: Error handling
    println!("6. Error Handling:");
    match EinsumExpr::parse("ij,jk->im") {
        Ok(_) => println!("   Unexpected success!"),
        Err(e) => println!("   Caught error: {}", e),
    }

    match EinsumExpr::parse("i1,jk->ik") {
        Ok(_) => println!("   Unexpected success!"),
        Err(e) => println!("   Caught error: {}", e),
    }

    println!("\n=== Demo Complete ===");
}
