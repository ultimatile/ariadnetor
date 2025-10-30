//! Demonstration of IR generation from Einsum expressions
//!
//! This example shows how to use the TCBuilder with parsed einsum expressions
//! to automatically generate TN-Compute dialect IR.
//!
//! Run with:
//!   MLIR_SYS_200_PREFIX=/opt/homebrew/opt/llvm@20 cargo run --features mlir --example einsum_ir_builder_demo

#[cfg(feature = "mlir")]
fn main() -> anyhow::Result<()> {
    use arnet::{EinsumExpr, TCBuilder, TCDialect};
    use melior::{
        Context,
        dialect::DialectRegistry,
        ir::{
            Block, BlockLike,
            r#type::{RankedTensorType, Type},
        },
        utility::register_all_dialects,
    };

    println!("=== Einsum IR Builder Demo ===\n");

    // Setup MLIR context
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Load TN dialect
    let _tn_dialect = TCDialect::new()?;

    // Create builder
    let builder = TCBuilder::new(&context);
    let location = builder.location();

    println!("1. Matrix Multiplication: ij,jk->ik\n");

    // Parse einsum expression
    let expr = EinsumExpr::parse("ij,jk->ik")?;
    println!("   Parsed expression:");
    println!("     LHS indices: {:?}", expr.lhs_indices());
    println!("     RHS indices: {:?}", expr.rhs_indices());
    println!("     Output indices: {:?}", expr.out_indices());
    println!("     Contracted: {:?}", expr.contracted_indices());

    // Create test block with tensor arguments
    let f64_type = Type::float64(&context);
    let lhs_type = RankedTensorType::new(&[10, 20], f64_type, None);
    let rhs_type = RankedTensorType::new(&[20, 30], f64_type, None);

    let block = Block::new(&[(lhs_type.into(), location), (rhs_type.into(), location)]);

    let lhs = block.argument(0)?.into();
    let rhs = block.argument(1)?.into();

    // Build IR from einsum expression
    let _result =
        builder.build_contract_from_einsum(&expr, lhs, rhs, &[10, 20], &[20, 30], f64_type)?;

    println!("   ✓ Generated tn.contract operation");
    println!("   Input shapes: [10, 20] x [20, 30]");
    println!("   Output shape: [10, 30]\n");

    // Example 2: Higher-dimensional contraction
    println!("2. Higher-Dimensional: ijk,jkl->il\n");

    let expr2 = EinsumExpr::parse("ijk,jkl->il")?;
    println!("   Parsed expression:");
    println!("     LHS indices: {:?}", expr2.lhs_indices());
    println!("     RHS indices: {:?}", expr2.rhs_indices());
    println!("     Output indices: {:?}", expr2.out_indices());
    println!("     Contracted: {:?}", expr2.contracted_indices());

    let lhs_type2 = RankedTensorType::new(&[5, 10, 15], f64_type, None);
    let rhs_type2 = RankedTensorType::new(&[10, 15, 20], f64_type, None);

    let block2 = Block::new(&[(lhs_type2.into(), location), (rhs_type2.into(), location)]);

    let lhs2 = block2.argument(0)?.into();
    let rhs2 = block2.argument(1)?.into();

    let _result2 = builder.build_contract_from_einsum(
        &expr2,
        lhs2,
        rhs2,
        &[5, 10, 15],
        &[10, 15, 20],
        f64_type,
    )?;

    println!("   ✓ Generated tn.contract operation");
    println!("   Input shapes: [5, 10, 15] x [10, 15, 20]");
    println!("   Output shape: [5, 20]\n");

    // Example 3: Batch matrix multiplication
    println!("3. Batch Matrix Multiply: bij,bjk->bik\n");

    let expr3 = EinsumExpr::parse("bij,bjk->bik")?;
    println!("   Parsed expression:");
    println!("     LHS indices: {:?}", expr3.lhs_indices());
    println!("     RHS indices: {:?}", expr3.rhs_indices());
    println!("     Output indices: {:?}", expr3.out_indices());
    println!("     Contracted: {:?}", expr3.contracted_indices());

    let lhs_type3 = RankedTensorType::new(&[32, 10, 20], f64_type, None);
    let rhs_type3 = RankedTensorType::new(&[32, 20, 30], f64_type, None);

    let block3 = Block::new(&[(lhs_type3.into(), location), (rhs_type3.into(), location)]);

    let lhs3 = block3.argument(0)?.into();
    let rhs3 = block3.argument(1)?.into();

    let _result3 = builder.build_contract_from_einsum(
        &expr3,
        lhs3,
        rhs3,
        &[32, 10, 20],
        &[32, 20, 30],
        f64_type,
    )?;

    println!("   ✓ Generated tn.contract operation");
    println!("   Input shapes: [32, 10, 20] x [32, 20, 30]");
    println!("   Output shape: [32, 10, 30]");
    println!("   (32 batches of 10x20 @ 20x30 matrix multiplications)\n");

    // Example 4: Error handling - dimension mismatch
    println!("4. Error Handling: Dimension Mismatch\n");

    let expr4 = EinsumExpr::parse("ij,jk->ik")?;

    let lhs_type4 = RankedTensorType::new(&[10, 20], f64_type, None);
    let rhs_type4 = RankedTensorType::new(&[25, 30], f64_type, None);

    let block4 = Block::new(&[(lhs_type4.into(), location), (rhs_type4.into(), location)]);

    let lhs4 = block4.argument(0)?.into();
    let rhs4 = block4.argument(1)?.into();

    match builder.build_contract_from_einsum(
        &expr4,
        lhs4,
        rhs4,
        &[10, 20],
        &[25, 30], // Mismatched: 20 != 25
        f64_type,
    ) {
        Ok(_) => println!("   ✗ Unexpected success!"),
        Err(e) => println!("   ✓ Caught error: {}", e),
    }

    println!("\n=== Demo Complete ===");
    println!("\nKey Features Demonstrated:");
    println!("  • Automatic shape inference from einsum expressions");
    println!("  • Type-safe IR generation");
    println!("  • Support for various tensor operations (matmul, batch, higher-dim)");
    println!("  • Built-in dimension validation\n");

    Ok(())
}

#[cfg(not(feature = "mlir"))]
fn main() {
    eprintln!("This example requires the 'mlir' feature to be enabled.");
    eprintln!(
        "Run with: MLIR_SYS_200_PREFIX=/opt/homebrew/opt/llvm@20 cargo run --features mlir --example einsum_ir_builder_demo"
    );
    std::process::exit(1);
}
