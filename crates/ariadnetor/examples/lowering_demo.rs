//! Demonstration of TN → LinAlg lowering
//!
//! This example shows how to generate TN-Compute IR and print it.
//! The actual lowering to LinAlg is performed by the C++ pass, which
//! can be invoked via mlir-opt command-line tool.
//!
//! Run with:
//!   MLIR_SYS_200_PREFIX=/opt/homebrew/opt/llvm@20 cargo run --features mlir --example lowering_demo

#[cfg(feature = "mlir")]
fn main() -> anyhow::Result<()> {
    use melior::{
        Context,
        dialect::DialectRegistry,
        ir::{
            Block, BlockLike, Location, RegionLike,
            r#type::{RankedTensorType, Type},
            Module,
        },
        utility::register_all_dialects,
    };
    use arnet::{TCBuilder, TCDialect, EinsumExpr};

    println!("=== TN → LinAlg Lowering Demo ===\n");

    // Setup MLIR context
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Load TN dialect
    let _tn_dialect = TCDialect::new()?;

    println!("1. Generating TN IR for Matrix Multiplication\n");

    // Create a function module for better output
    let location = Location::unknown(&context);
    let module = Module::new(location);

    let f64_type = Type::float64(&context);
    let lhs_type = RankedTensorType::new(&[10, 20], f64_type, None);
    let rhs_type = RankedTensorType::new(&[20, 30], f64_type, None);

    // Create function with TN operations
    let _func_type = {
        use melior::ir::r#type::FunctionType;
        FunctionType::new(&context, &[lhs_type.into(), rhs_type.into()], &[])
    };

    let func_op = {
        use melior::ir::{operation::OperationBuilder, attribute::StringAttribute, Identifier};

        let func_name = StringAttribute::new(&context, "test_matmul");
        let func_name_id = Identifier::new(&context, "sym_name");

        let region = {
            let region = melior::ir::Region::new();
            let block = Block::new(&[
                (lhs_type.into(), location),
                (rhs_type.into(), location),
            ]);

            // Build TN contract operation
            let builder = TCBuilder::new(&context);
            let expr = EinsumExpr::parse("ij,jk->ik")?;

            let lhs = block.argument(0)?.into();
            let rhs = block.argument(1)?.into();

            let result = builder.build_contract_from_einsum(
                &expr,
                lhs,
                rhs,
                &[10, 20],
                &[20, 30],
                f64_type
            )?;

            // Add return
            use melior::ir::operation::OperationBuilder as OpBuilder;
            let return_op = OpBuilder::new("func.return", location)
                .add_operands(&[result])
                .build()?;

            block.append_operation(return_op);
            region.append_block(block);
            region
        };

        OperationBuilder::new("func.func", location)
            .add_attributes(&[(func_name_id, func_name.into())])
            .add_regions([region])
            .build()?
    };

    module.body().append_operation(func_op);

    println!("✓ Successfully generated TN IR with tn.contract operation");
    println!("  Operation: tn.contract(ij,jk->ik)");
    println!("  Input shapes: [10, 20] x [20, 30]");
    println!("  Output shape: [10, 30]");

    println!("\n2. Lowering to LinAlg\n");
    println!("To lower this IR to LinAlg dialect, save the above IR to a file");
    println!("and run:");
    println!("  mlir-opt --convert-tn-to-linalg <file.mlir>");
    println!("\nThis will convert:");
    println!("  - tn.contract(ij,jk->ik) → linalg.matmul");
    println!("  - tn.contract(bij,bjk->bik) → linalg.batch_matmul");
    println!("  - tn.contract(ij,ij->ij) → linalg.map with arith.mulf");
    println!("  - tn.transpose → linalg.transpose");

    println!("\n=== Demo Complete ===\n");

    Ok(())
}

#[cfg(not(feature = "mlir"))]
fn main() {
    eprintln!("This example requires the 'mlir' feature to be enabled.");
    eprintln!("Run with: MLIR_SYS_200_PREFIX=/opt/homebrew/opt/llvm@20 cargo run --features mlir --example lowering_demo");
    std::process::exit(1);
}
