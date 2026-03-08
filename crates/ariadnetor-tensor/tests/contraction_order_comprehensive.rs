// Comprehensive test for contracted indices ordering fix
// This verifies that the critical bug fix is working correctly
//
// Tests that use actual tensor contraction (contract_naive) have been
// migrated to ariadnetor-linalg/tests/contraction.rs since contraction
// now goes through ComputeBackend.

use arnet_tensor::einsum::{ContractionPlan, EinsumExpr};

#[test]
fn test_contracted_indices_preserve_lhs_order() {
    // Test Case: "ikj,jkl->il"
    // LHS has indices in order: i, k, j
    // Contracted indices should be [k, j] (LHS order), NOT [j, k] (ASCII order)
    let expr = EinsumExpr::parse("ikj,jkl->il").unwrap();
    let plan = ContractionPlan::from_expr(&expr);

    assert_eq!(
        plan.contracted,
        vec![b'k', b'j'],
        "Contracted indices must preserve LHS order: [k, j], not ASCII order [j, k]"
    );
    assert_eq!(plan.free_lhs, vec![b'i']);
    assert_eq!(plan.free_rhs, vec![b'l']);
}

#[test]
fn test_contracted_indices_another_order() {
    // Test Case: "jki,kij->i"
    // LHS has indices: j, k, i
    // Contracted indices should be [j, k] (LHS order)
    let expr = EinsumExpr::parse("jki,kij->i").unwrap();
    let plan = ContractionPlan::from_expr(&expr);

    assert_eq!(
        plan.contracted,
        vec![b'j', b'k'],
        "Contracted indices must preserve LHS order: [j, k]"
    );
}

#[test]
fn test_three_contracted_indices() {
    // Test with three contracted indices: "abcd,dcba->nothing" (all contracted)
    let expr = EinsumExpr::parse("abcd,dcba->").unwrap();
    let plan = ContractionPlan::from_expr(&expr);

    // Should preserve LHS order: a, b, c, d
    assert_eq!(
        plan.contracted,
        vec![b'a', b'b', b'c', b'd'],
        "All contracted indices should preserve LHS order"
    );
    assert!(plan.free_lhs.is_empty());
    assert!(plan.free_rhs.is_empty());
}
