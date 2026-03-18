//! Integration tests for unified EinsumExpr (re-exported from arnet_core)
//!
//! These tests verify the EinsumExpr API from a user perspective,
//! testing the public interface as it would be used in real code.

use arnet::EinsumExpr;

#[test]
fn test_matrix_multiplication_end_to_end() {
    let expr = EinsumExpr::parse("ij,jk->ik").expect("Failed to parse matrix multiply");

    assert_eq!(expr.lhs_indices(), &[b'i', b'j']);
    assert_eq!(expr.rhs_indices(), &[b'j', b'k']);
    assert_eq!(expr.out_indices(), &[b'i', b'k']);

    let contracted = expr.contracted_indices();
    assert_eq!(contracted.len(), 1);
    assert_eq!(contracted[0], b'j');

    assert!(expr.is_matrix_multiply());

    let output_shape = expr
        .infer_output_shape(&[&[10, 20], &[20, 30]])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![10, 30]);
}

#[test]
fn test_higher_dimensional_contraction() {
    let expr =
        EinsumExpr::parse("ijk,jkl->il").expect("Failed to parse higher dimensional contraction");

    assert_eq!(expr.lhs_indices(), &[b'i', b'j', b'k']);
    assert_eq!(expr.rhs_indices(), &[b'j', b'k', b'l']);
    assert_eq!(expr.out_indices(), &[b'i', b'l']);

    let contracted = expr.contracted_indices();
    assert_eq!(contracted.len(), 2);
    assert!(contracted.contains(&b'j'));
    assert!(contracted.contains(&b'k'));

    assert!(!expr.is_matrix_multiply());

    let output_shape = expr
        .infer_output_shape(&[&[5, 10, 15], &[10, 15, 20]])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![5, 20]);
}

#[test]
fn test_element_wise_multiplication() {
    let expr = EinsumExpr::parse("ij,ij->ij").expect("Failed to parse element-wise multiplication");

    assert_eq!(expr.lhs_indices(), &[b'i', b'j']);
    assert_eq!(expr.rhs_indices(), &[b'i', b'j']);
    assert_eq!(expr.out_indices(), &[b'i', b'j']);

    let contracted = expr.contracted_indices();
    assert_eq!(contracted.len(), 0);

    assert!(!expr.is_matrix_multiply());

    let output_shape = expr
        .infer_output_shape(&[&[10, 20], &[10, 20]])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![10, 20]);
}

#[test]
fn test_batch_matrix_multiplication() {
    let expr =
        EinsumExpr::parse("bij,bjk->bik").expect("Failed to parse batch matrix multiplication");

    assert_eq!(expr.lhs_indices(), &[b'b', b'i', b'j']);
    assert_eq!(expr.rhs_indices(), &[b'b', b'j', b'k']);
    assert_eq!(expr.out_indices(), &[b'b', b'i', b'k']);

    let contracted = expr.contracted_indices();
    assert_eq!(contracted, vec![b'j']);

    let output_shape = expr
        .infer_output_shape(&[&[32, 10, 20], &[32, 20, 30]])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![32, 10, 30]);
}

#[test]
fn test_tensor_outer_product() {
    let expr = EinsumExpr::parse("ij,kl->ijkl").expect("Failed to parse outer product");

    assert_eq!(expr.lhs_indices(), &[b'i', b'j']);
    assert_eq!(expr.rhs_indices(), &[b'k', b'l']);
    assert_eq!(expr.out_indices(), &[b'i', b'j', b'k', b'l']);

    let contracted = expr.contracted_indices();
    assert_eq!(contracted.len(), 0);

    let output_shape = expr
        .infer_output_shape(&[&[10, 20], &[30, 40]])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![10, 20, 30, 40]);
}

#[test]
fn test_partial_trace() {
    let expr = EinsumExpr::parse("iij,jk->ik").expect("Failed to parse partial trace");

    assert_eq!(expr.lhs_indices(), &[b'i', b'i', b'j']);
    assert_eq!(expr.rhs_indices(), &[b'j', b'k']);
    assert_eq!(expr.out_indices(), &[b'i', b'k']);

    let contracted = expr.contracted_indices();
    assert_eq!(contracted, vec![b'j']);

    let output_shape = expr
        .infer_output_shape(&[&[10, 10, 20], &[20, 30]])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![10, 30]);
}

#[test]
fn test_single_tensor_trace() {
    let expr = EinsumExpr::parse("ii->").expect("Failed to parse trace");
    assert_eq!(expr.num_inputs(), 1);
    assert_eq!(expr.lhs_indices(), &[b'i', b'i']);
    assert_eq!(expr.out_indices(), &[] as &[u8]);
}

#[test]
fn test_single_tensor_transpose() {
    let expr = EinsumExpr::parse("ij->ji").expect("Failed to parse transpose");
    assert_eq!(expr.num_inputs(), 1);
    assert_eq!(expr.lhs_indices(), &[b'i', b'j']);
    assert_eq!(expr.out_indices(), &[b'j', b'i']);
}

#[test]
fn test_implicit_output_inference() {
    // "ij,jk" → free indices i,k sorted → "ij,jk->ik"
    let expr = EinsumExpr::parse("ij,jk").expect("Failed to parse implicit output");
    assert_eq!(expr.out_indices(), &[b'i', b'k']);
}

#[test]
fn test_error_output_index_not_in_input() {
    let result = EinsumExpr::parse("ij,jk->im");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("does not appear"));
}

#[test]
fn test_error_single_input_invalid_output() {
    // Single input "ij" with output index 'k' not in input
    let result = EinsumExpr::parse("ij->ik");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("does not appear"));
}

#[test]
fn test_uppercase_indices_are_valid() {
    // Core EinsumExpr supports both A-Z and a-z as index characters
    let expr = EinsumExpr::parse("iJ,Jk->ik").unwrap();
    assert_eq!(expr.lhs_indices(), &[b'i', b'J']);
    assert_eq!(expr.rhs_indices(), &[b'J', b'k']);
}

#[test]
fn test_error_invalid_character_number() {
    let result = EinsumExpr::parse("i1,jk->ik");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("Invalid"));
}

#[test]
fn test_error_dimension_mismatch_contracted() {
    let expr = EinsumExpr::parse("ij,jk->ik").expect("Failed to parse");

    let result = expr.infer_output_shape(&[&[10, 20], &[25, 30]]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("Dimension mismatch"));
}

#[test]
fn test_error_dimension_mismatch_repeated() {
    let expr = EinsumExpr::parse("iij,jk->ik").expect("Failed to parse");

    let result = expr.infer_output_shape(&[&[10, 15, 20], &[20, 30]]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("Dimension mismatch"));
}

#[test]
fn test_error_shape_rank_mismatch() {
    let expr = EinsumExpr::parse("ij,jk->ik").expect("Failed to parse");

    let result = expr.infer_output_shape(&[&[10, 20, 30], &[20, 30]]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("does not match index count"));
}

#[test]
fn test_whitespace_handling() {
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
    let expr = EinsumExpr::parse("abcd,cdef->abef").expect("Failed to parse complex contraction");

    assert_eq!(expr.lhs_indices(), &[b'a', b'b', b'c', b'd']);
    assert_eq!(expr.rhs_indices(), &[b'c', b'd', b'e', b'f']);
    assert_eq!(expr.out_indices(), &[b'a', b'b', b'e', b'f']);

    let contracted = expr.contracted_indices();
    assert_eq!(contracted.len(), 2);
    assert!(contracted.contains(&b'c'));
    assert!(contracted.contains(&b'd'));

    let output_shape = expr
        .infer_output_shape(&[&[2, 3, 4, 5], &[4, 5, 6, 7]])
        .expect("Failed to infer output shape");
    assert_eq!(output_shape, vec![2, 3, 6, 7]);
}
