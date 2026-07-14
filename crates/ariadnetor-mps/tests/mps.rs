//! Tests for MPS/MPO data structures and TensorChain trait

#[path = "mps/helpers.rs"]
mod helpers;

#[path = "mps/apply.rs"]
mod apply;
#[path = "mps/apply_block_sparse.rs"]
mod apply_block_sparse;
#[path = "mps/apply_successive_randomized.rs"]
mod apply_successive_randomized;
#[path = "mps/apply_variational.rs"]
mod apply_variational;
#[path = "mps/apply_variational_block_sparse.rs"]
mod apply_variational_block_sparse;
#[path = "mps/basic.rs"]
mod basic;
#[path = "mps/canonicalize.rs"]
mod canonicalize;
#[path = "mps/canonicalize_block_sparse.rs"]
mod canonicalize_block_sparse;
#[path = "mps/col_major.rs"]
mod col_major;
#[path = "mps/inner_block_sparse.rs"]
mod inner_block_sparse;
#[path = "mps/inner_product.rs"]
mod inner_product;
#[path = "mps/operators.rs"]
mod operators;
#[path = "mps/truncate.rs"]
mod truncate;
#[path = "mps/truncate_block_sparse.rs"]
mod truncate_block_sparse;

// Mutation-testing coverage modules
#[path = "mps/canonicalize_mutant.rs"]
mod canonicalize_mutant;
#[path = "mps/chain_mutant.rs"]
mod chain_mutant;
#[path = "mps/inner_mutant.rs"]
mod inner_mutant;
#[path = "mps/site_ops_mutant.rs"]
mod site_ops_mutant;
#[path = "mps/truncate_mutant.rs"]
mod truncate_mutant;
