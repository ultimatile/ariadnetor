//! Integration tests for Einsum parser
//!
//! These tests verify the EinsumExpr API from a user perspective,
//! testing the public interface as it would be used in real code.

use tn_mlir::EinsumExpr;

#[test]
fn test_matrix_multiplication_end_to_end() {
    // Parse matrix multiplication notation
    let expr = EinsumExpr::parse("ij,jk->ik").expect("Failed to parse matrix multiply");

    // Verify indices are correct
    assert_eq!(expr.lhs_indices(), &['i', 'j']);
    assert_eq!(expr.rhs_indices(), &['j', 'k']);
    assert_eq!(expr.out_indices(), &['i', 'k']);

    // Verify contracted index detection
    let contracted = expr.contracted_indices();
    assert_eq!(contracted.len(), 1);
    assert_eq!(contracted[0], 'j');

    // Verify pattern detection
    assert!(expr.is_matrix_multiply());

    // Verify shape inference
    let output_shape = expr.infer_output_shape(&[10, 20], &[20, 30])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![10, 30]);
}

#[test]
fn test_higher_dimensional_contraction() {
    let expr = EinsumExpr::parse("ijk,jkl->il")
        .expect("Failed to parse higher dimensional contraction");

    assert_eq!(expr.lhs_indices(), &['i', 'j', 'k']);
    assert_eq!(expr.rhs_indices(), &['j', 'k', 'l']);
    assert_eq!(expr.out_indices(), &['i', 'l']);

    // Two indices are contracted
    let contracted = expr.contracted_indices();
    assert_eq!(contracted.len(), 2);
    assert!(contracted.contains(&'j'));
    assert!(contracted.contains(&'k'));

    // Not a simple matrix multiply
    assert!(!expr.is_matrix_multiply());

    // Shape inference with 3D tensors
    let output_shape = expr.infer_output_shape(&[5, 10, 15], &[10, 15, 20])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![5, 20]);
}

#[test]
fn test_element_wise_multiplication() {
    let expr = EinsumExpr::parse("ij,ij->ij")
        .expect("Failed to parse element-wise multiplication");

    assert_eq!(expr.lhs_indices(), &['i', 'j']);
    assert_eq!(expr.rhs_indices(), &['i', 'j']);
    assert_eq!(expr.out_indices(), &['i', 'j']);

    // No contracted indices
    let contracted = expr.contracted_indices();
    assert_eq!(contracted.len(), 0);

    // Not a matrix multiply (no contraction)
    assert!(!expr.is_matrix_multiply());

    // Output shape matches input shapes
    let output_shape = expr.infer_output_shape(&[10, 20], &[10, 20])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![10, 20]);
}

#[test]
fn test_batch_matrix_multiplication() {
    // Batch matrix multiply: batch dimension preserved, inner dims contracted
    let expr = EinsumExpr::parse("bij,bjk->bik")
        .expect("Failed to parse batch matrix multiplication");

    assert_eq!(expr.lhs_indices(), &['b', 'i', 'j']);
    assert_eq!(expr.rhs_indices(), &['b', 'j', 'k']);
    assert_eq!(expr.out_indices(), &['b', 'i', 'k']);

    // Only 'j' is contracted
    let contracted = expr.contracted_indices();
    assert_eq!(contracted, vec!['j']);

    // Shape inference with batch dimension
    let output_shape = expr.infer_output_shape(&[32, 10, 20], &[32, 20, 30])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![32, 10, 30]);
}

#[test]
fn test_tensor_outer_product() {
    // Outer product: no contracted indices
    let expr = EinsumExpr::parse("ij,kl->ijkl")
        .expect("Failed to parse outer product");

    assert_eq!(expr.lhs_indices(), &['i', 'j']);
    assert_eq!(expr.rhs_indices(), &['k', 'l']);
    assert_eq!(expr.out_indices(), &['i', 'j', 'k', 'l']);

    // No contraction
    let contracted = expr.contracted_indices();
    assert_eq!(contracted.len(), 0);

    // Shape inference: output has all dimensions
    let output_shape = expr.infer_output_shape(&[10, 20], &[30, 40])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![10, 20, 30, 40]);
}

#[test]
fn test_partial_trace() {
    // Partial trace: contract one pair of indices in first tensor
    let expr = EinsumExpr::parse("iij,jk->ik")
        .expect("Failed to parse partial trace");

    assert_eq!(expr.lhs_indices(), &['i', 'i', 'j']);
    assert_eq!(expr.rhs_indices(), &['j', 'k']);
    assert_eq!(expr.out_indices(), &['i', 'k']);

    // 'j' is contracted between tensors
    let contracted = expr.contracted_indices();
    assert_eq!(contracted, vec!['j']);

    // Shape inference: repeated index must have same dimension
    let output_shape = expr.infer_output_shape(&[10, 10, 20], &[20, 30])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![10, 30]);
}

#[test]
fn test_error_invalid_format_no_arrow() {
    let result = EinsumExpr::parse("ij,jk");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("expected format"));
}

#[test]
fn test_error_invalid_format_single_input() {
    let result = EinsumExpr::parse("ij->ik");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("two input tensors"));
}

#[test]
fn test_error_invalid_character_uppercase() {
    let result = EinsumExpr::parse("iJ,jk->ik");
    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("lowercase") ||
        err_msg.contains("Invalid") ||
        err_msg.contains("Failed to parse")
    );
}

#[test]
fn test_error_invalid_character_number() {
    let result = EinsumExpr::parse("i1,jk->ik");
    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("lowercase") ||
        err_msg.contains("Invalid") ||
        err_msg.contains("Failed to parse")
    );
}

#[test]
fn test_error_output_index_not_in_input() {
    let result = EinsumExpr::parse("ij,jk->im");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("does not appear"));
}

#[test]
fn test_error_dimension_mismatch_contracted() {
    let expr = EinsumExpr::parse("ij,jk->ik")
        .expect("Failed to parse");

    // 'j' dimension mismatch: 20 vs 25
    let result = expr.infer_output_shape(&[10, 20], &[25, 30]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Dimension mismatch"));
}

#[test]
fn test_error_dimension_mismatch_repeated() {
    let expr = EinsumExpr::parse("iij,jk->ik")
        .expect("Failed to parse");

    // 'i' appears twice in LHS but with different dimensions
    let result = expr.infer_output_shape(&[10, 15, 20], &[20, 30]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Dimension mismatch"));
}

#[test]
fn test_error_shape_rank_mismatch() {
    let expr = EinsumExpr::parse("ij,jk->ik")
        .expect("Failed to parse");

    // Shape has wrong rank (3 instead of 2)
    let result = expr.infer_output_shape(&[10, 20, 30], &[20, 30]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("does not match index count"));
}

#[test]
fn test_whitespace_handling() {
    // Parser should handle whitespace gracefully
    let expr1 = EinsumExpr::parse("ij,jk->ik").unwrap();
    let expr2 = EinsumExpr::parse("  ij , jk -> ik  ").unwrap();
    let expr3 = EinsumExpr::parse("ij ,jk-> ik").unwrap();

    assert_eq!(expr1.lhs_indices(), expr2.lhs_indices());
    assert_eq!(expr1.lhs_indices(), expr3.lhs_indices());
    assert_eq!(expr1.rhs_indices(), expr2.rhs_indices());
    assert_eq!(expr1.rhs_indices(), expr3.rhs_indices());
    assert_eq!(expr1.out_indices(), expr2.out_indices());
    assert_eq!(expr1.out_indices(), expr3.out_indices());
}

#[test]
fn test_complex_multidimensional_contraction() {
    // Test a complex case: 4D tensor contraction
    let expr = EinsumExpr::parse("abcd,cdef->abef")
        .expect("Failed to parse complex contraction");

    assert_eq!(expr.lhs_indices(), &['a', 'b', 'c', 'd']);
    assert_eq!(expr.rhs_indices(), &['c', 'd', 'e', 'f']);
    assert_eq!(expr.out_indices(), &['a', 'b', 'e', 'f']);

    // Two contracted indices
    let contracted = expr.contracted_indices();
    assert_eq!(contracted.len(), 2);
    assert!(contracted.contains(&'c'));
    assert!(contracted.contains(&'d'));

    // Shape inference
    let output_shape = expr.infer_output_shape(&[2, 3, 4, 5], &[4, 5, 6, 7])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![2, 3, 6, 7]);
}
