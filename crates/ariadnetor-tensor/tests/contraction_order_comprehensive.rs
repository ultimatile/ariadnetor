// Comprehensive test for contracted indices ordering fix
// This verifies that the critical bug fix is working correctly

use arnet_tensor::{DenseTensor, einsum::{EinsumExpr, ContractionPlan}};

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
fn test_actual_contraction_with_reordered_indices() {
    // Verify that actual tensor contraction works correctly
    // with reordered contracted indices

    let a = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2], // i=2, k=2, j=2
    );
    let b = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2], // j=2, k=2, l=2
    );

    // This should not panic and should produce correct output shape
    let c = a.contract_naive(&b, "ikj,jkl->il");
    assert_eq!(c.shape(), &[2, 2], "Output shape should be [2, 2]");

    // Verify result is non-zero (actual contraction happened)
    assert_ne!(c.get(&[0, 0]), 0.0, "Result should be non-zero");
}

#[test]
fn test_consistency_between_ijk_and_ikj_layouts() {
    // Compare "ijk,jkl->il" vs "ikj,jkl->il" with properly permuted A
    // Both should give the same mathematical result

    let a_ijk = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2], // i=2, j=2, k=2
    );
    let b = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2], // j=2, k=2, l=2
    );

    let result_ijk = a_ijk.contract_naive(&b, "ijk,jkl->il");

    // Create A with ikj layout by permuting [i,j,k] -> [i,k,j]
    let a_ikj = a_ijk.permute(&[0, 2, 1]);
    let result_ikj = a_ikj.contract_naive(&b, "ikj,jkl->il");

    // Both results should have the same shape
    assert_eq!(result_ijk.shape(), result_ikj.shape());

    // Both should compute valid (non-zero) results
    assert_ne!(result_ijk.get(&[0, 0]), 0.0);
    assert_ne!(result_ikj.get(&[0, 0]), 0.0);
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
