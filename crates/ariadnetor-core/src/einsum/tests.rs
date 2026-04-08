use super::*;

// ---- Accessor methods ----

#[test]
fn accessors_matmul() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    assert_eq!(expr.inputs(), &[vec![b'i', b'j'], vec![b'j', b'k']]);
    assert_eq!(expr.out_indices(), b"ik");
    assert_eq!(expr.num_inputs(), 2);
    assert_eq!(expr.lhs_indices(), b"ij");
    assert_eq!(expr.rhs_indices(), b"jk");
}

#[test]
fn accessors_three_index_contraction() {
    let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    assert_eq!(expr.inputs().len(), 2);
    assert_eq!(expr.num_inputs(), 2);
    assert_eq!(expr.lhs_indices(), b"ijk");
    assert_eq!(expr.rhs_indices(), b"jkl");
    assert_eq!(expr.out_indices(), b"il");
}

#[test]
fn accessors_single_input_trace() {
    let expr = EinsumExpr::parse("ii->").unwrap();
    assert_eq!(expr.num_inputs(), 1);
    assert_eq!(expr.inputs(), &[vec![b'i', b'i']]);
    assert_eq!(expr.lhs_indices(), b"ii");
    assert_eq!(expr.out_indices(), &[] as &[u8]);
}

#[test]
fn accessors_single_input_no_contraction() {
    let expr = EinsumExpr::parse("ij->ji").unwrap();
    assert_eq!(expr.num_inputs(), 1);
    assert_eq!(expr.inputs(), &[vec![b'i', b'j']]);
    assert_eq!(expr.out_indices(), b"ji");
}

// ---- all_indices ----

#[test]
fn all_indices_matmul() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    let all = expr.all_indices();
    assert_eq!(all.len(), 3);
    assert!(all.contains(&b'i'));
    assert!(all.contains(&b'j'));
    assert!(all.contains(&b'k'));
}

#[test]
fn all_indices_trace() {
    let expr = EinsumExpr::parse("ii->").unwrap();
    let all = expr.all_indices();
    assert_eq!(all.len(), 1);
    assert!(all.contains(&b'i'));
}

// ---- contracted_indices ----

#[test]
fn contracted_indices_matmul() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    assert_eq!(expr.contracted_indices(), vec![b'j']);
}

#[test]
fn contracted_indices_multi() {
    let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    assert_eq!(expr.contracted_indices(), vec![b'j', b'k']);
}

#[test]
fn contracted_indices_preserves_lhs_order() {
    let expr = EinsumExpr::parse("ikj,jkl->il").unwrap();
    // k appears before j in lhs
    assert_eq!(expr.contracted_indices(), vec![b'k', b'j']);
}

#[test]
fn contracted_indices_none() {
    // Outer product: no contracted indices
    let expr = EinsumExpr::parse("ij,kl->ijkl").unwrap();
    assert!(expr.contracted_indices().is_empty());
}

#[test]
fn contracted_indices_trace() {
    let expr = EinsumExpr::parse("ii->").unwrap();
    assert_eq!(expr.contracted_indices(), vec![b'i']);
}

// ---- infer_output (implicit output) ----

#[test]
fn infer_output_two_inputs() {
    let expr = EinsumExpr::parse("ij,jk").unwrap();
    // Free indices sorted alphabetically: i, k
    assert_eq!(expr.out_indices(), b"ik");
}

#[test]
fn infer_output_all_contracted() {
    // Both indices appear twice -> empty output (scalar)
    let expr = EinsumExpr::parse("ij,ij").unwrap();
    assert_eq!(expr.out_indices(), &[] as &[u8]);
}

#[test]
fn infer_output_sorted() {
    // Free indices should be c, a sorted -> a, c
    let expr = EinsumExpr::parse("cb,ba").unwrap();
    // b appears twice -> contracted; a, c appear once -> free
    assert_eq!(expr.out_indices(), b"ac");
}

// ---- parse_indices ----

#[test]
fn parse_indices_valid() {
    let result = EinsumExpr::parse_indices("ijk");
    assert_eq!(result.unwrap(), vec![b'i', b'j', b'k']);
}

#[test]
fn parse_indices_uppercase() {
    let result = EinsumExpr::parse_indices("AB");
    assert_eq!(result.unwrap(), vec![b'A', b'B']);
}

#[test]
fn parse_indices_empty() {
    let result = EinsumExpr::parse_indices("");
    assert_eq!(result.unwrap(), vec![]);
}

#[test]
fn parse_indices_invalid_digit() {
    let result = EinsumExpr::parse_indices("i1j");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid index '1'"));
}

#[test]
fn parse_indices_invalid_special() {
    let result = EinsumExpr::parse_indices("i+j");
    assert!(result.is_err());
}

// ---- validate ----

#[test]
fn validate_ok() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    assert!(expr.validate().is_ok());
}

#[test]
fn validate_output_index_not_in_input() {
    let expr = EinsumExpr {
        inputs: vec![vec![b'i', b'j']],
        out_indices: vec![b'i', b'z'],
    };
    let result = expr.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("'z'"));
}

// ---- is_matrix_multiply ----

#[test]
fn is_matmul_standard() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    assert!(expr.is_matrix_multiply());
}

#[test]
fn is_matmul_reversed_contraction() {
    let expr = EinsumExpr::parse("ji,ik->jk").unwrap();
    assert!(expr.is_matrix_multiply());
}

#[test]
fn is_matmul_false_single_input() {
    let expr = EinsumExpr::parse("ij->ji").unwrap();
    assert!(!expr.is_matrix_multiply());
}

#[test]
fn is_matmul_false_three_inputs() {
    let expr = EinsumExpr::parse("ij,jk,kl->il").unwrap();
    assert!(!expr.is_matrix_multiply());
}

#[test]
fn is_matmul_false_too_many_indices() {
    // 4 unique indices -> not matmul
    let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    assert!(!expr.is_matrix_multiply());
}

#[test]
fn is_matmul_false_rank3_inputs() {
    // Each input has 3 indices, not 2
    let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    assert!(!expr.is_matrix_multiply());
}

#[test]
fn is_matmul_false_rank1_times_rank2() {
    // One input has 1 index, 3 unique indices, but not a matrix multiply
    let expr = EinsumExpr::parse("i,ij->j").unwrap();
    assert!(!expr.is_matrix_multiply());
}

#[test]
fn is_matmul_false_scalar_output() {
    // Output has 0 indices
    let expr = EinsumExpr::parse("ij,ji->").unwrap();
    assert!(!expr.is_matrix_multiply());
}

#[test]
fn is_matmul_false_single_output_index() {
    // Output has 1 index (vector), not 2
    let expr = EinsumExpr::parse("ij,j->i").unwrap();
    assert!(!expr.is_matrix_multiply());
}

#[test]
fn is_matmul_false_outer_product() {
    // No contracted index: ij,kl->ijkl has 4 unique indices
    let expr = EinsumExpr::parse("ij,kl->ijkl").unwrap();
    assert!(!expr.is_matrix_multiply());
}

// ---- infer_output_shape ----

#[test]
fn infer_output_shape_matmul() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    let shape = expr.infer_output_shape(&[&[2, 3], &[3, 4]]).unwrap();
    assert_eq!(shape, vec![2, 4]);
}

#[test]
fn infer_output_shape_trace() {
    let expr = EinsumExpr::parse("ii->").unwrap();
    let shape = expr.infer_output_shape(&[&[3, 3]]).unwrap();
    assert_eq!(shape, vec![]);
}

#[test]
fn infer_output_shape_transpose() {
    let expr = EinsumExpr::parse("ij->ji").unwrap();
    let shape = expr.infer_output_shape(&[&[2, 5]]).unwrap();
    assert_eq!(shape, vec![5, 2]);
}

#[test]
fn infer_output_shape_three_index() {
    let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    let shape = expr.infer_output_shape(&[&[2, 3, 4], &[3, 4, 5]]).unwrap();
    assert_eq!(shape, vec![2, 5]);
}

#[test]
fn infer_output_shape_wrong_num_shapes() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    let result = expr.infer_output_shape(&[&[2, 3]]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Expected 2"));
}

#[test]
fn infer_output_shape_rank_mismatch() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    // First input has rank 3 but "ij" expects rank 2
    let result = expr.infer_output_shape(&[&[2, 3, 4], &[3, 4]]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("rank"));
}

#[test]
fn infer_output_shape_dimension_mismatch() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    // j=3 in lhs but j=5 in rhs
    let result = expr.infer_output_shape(&[&[2, 3], &[5, 4]]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Dimension mismatch"));
}

// ---- ContractionPlan::from_expr ----

#[test]
fn contraction_plan_matmul() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    let plan = ContractionPlan::from_expr(&expr);
    assert!(plan.batch.is_empty());
    assert_eq!(plan.contracted, vec![b'j']);
    assert_eq!(plan.free_lhs, vec![b'i']);
    assert_eq!(plan.free_rhs, vec![b'k']);
}

#[test]
fn contraction_plan_multi_contract() {
    let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    let plan = ContractionPlan::from_expr(&expr);
    assert!(plan.batch.is_empty());
    assert_eq!(plan.contracted, vec![b'j', b'k']);
    assert_eq!(plan.free_lhs, vec![b'i']);
    assert_eq!(plan.free_rhs, vec![b'l']);
}

#[test]
fn contraction_plan_with_batch() {
    // b is in both inputs AND in output -> batch
    let expr = EinsumExpr::parse("bij,bjk->bik").unwrap();
    let plan = ContractionPlan::from_expr(&expr);
    assert_eq!(plan.batch, vec![b'b']);
    assert_eq!(plan.contracted, vec![b'j']);
    assert_eq!(plan.free_lhs, vec![b'i']);
    assert_eq!(plan.free_rhs, vec![b'k']);
}

#[test]
fn contraction_plan_outer_product() {
    let expr = EinsumExpr::parse("ij,kl->ijkl").unwrap();
    let plan = ContractionPlan::from_expr(&expr);
    assert!(plan.batch.is_empty());
    assert!(plan.contracted.is_empty());
    assert_eq!(plan.free_lhs, vec![b'i', b'j']);
    assert_eq!(plan.free_rhs, vec![b'k', b'l']);
}

#[test]
fn contraction_plan_scalar_result() {
    let expr = EinsumExpr::parse("ij,ij->").unwrap();
    let plan = ContractionPlan::from_expr(&expr);
    assert!(plan.batch.is_empty());
    assert_eq!(plan.contracted, vec![b'i', b'j']);
    assert!(plan.free_lhs.is_empty());
    assert!(plan.free_rhs.is_empty());
}

// ---- lhs_permutation / rhs_permutation ----

#[test]
fn lhs_permutation_identity() {
    // lhs is already in [batch, free_lhs, contracted] order
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    let plan = ContractionPlan::from_expr(&expr);
    // Target: [free_lhs=i, contracted=j] = [i, j], same as lhs
    let perm = plan.lhs_permutation(expr.lhs_indices(), expr.rhs_indices());
    assert_eq!(perm, None); // identity -> None
}

#[test]
fn lhs_permutation_needed() {
    // lhs = [j, i], target = [free_lhs=i, contracted=j] = [i, j]
    let expr = EinsumExpr::parse("ji,jk->ik").unwrap();
    let plan = ContractionPlan::from_expr(&expr);
    let perm = plan.lhs_permutation(expr.lhs_indices(), expr.rhs_indices());
    assert_eq!(perm, Some(vec![1, 0])); // swap i and j
}

#[test]
fn rhs_permutation_identity() {
    let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    let plan = ContractionPlan::from_expr(&expr);
    // Target: [contracted=j, free_rhs=k] = [j, k], same as rhs
    let perm = plan.rhs_permutation(expr.rhs_indices());
    assert_eq!(perm, None);
}

#[test]
fn rhs_permutation_needed() {
    // rhs = [k, j], target = [contracted=j, free_rhs=k] = [j, k]
    let expr = EinsumExpr::parse("ij,kj->ik").unwrap();
    let plan = ContractionPlan::from_expr(&expr);
    let perm = plan.rhs_permutation(expr.rhs_indices());
    assert_eq!(perm, Some(vec![1, 0]));
}

#[test]
fn permutations_with_batch() {
    let expr = EinsumExpr::parse("bij,bjk->bik").unwrap();
    let plan = ContractionPlan::from_expr(&expr);
    // lhs target: [batch=b, free_lhs=i, contracted=j] = [b, i, j] = same as lhs
    let lhs_perm = plan.lhs_permutation(expr.lhs_indices(), expr.rhs_indices());
    assert_eq!(lhs_perm, None);
    // rhs target: [batch=b, contracted=j, free_rhs=k] = [b, j, k] = same as rhs
    let rhs_perm = plan.rhs_permutation(expr.rhs_indices());
    assert_eq!(rhs_perm, None);
}

// ---- compute_permutation ----

#[test]
fn compute_permutation_identity() {
    let result = compute_permutation(b"ijk", b"ijk");
    assert_eq!(result, None);
}

#[test]
fn compute_permutation_swap() {
    let result = compute_permutation(b"ij", b"ji");
    assert_eq!(result, Some(vec![1, 0]));
}

#[test]
fn compute_permutation_reverse() {
    let result = compute_permutation(b"abc", b"cba");
    assert_eq!(result, Some(vec![2, 1, 0]));
}

#[test]
fn compute_permutation_rotate() {
    let result = compute_permutation(b"abc", b"bca");
    assert_eq!(result, Some(vec![1, 2, 0]));
}

// ---- Parse edge cases ----

#[test]
fn parse_whitespace_ignored() {
    let expr = EinsumExpr::parse("i j , j k -> i k").unwrap();
    assert_eq!(expr.out_indices(), b"ik");
    assert_eq!(expr.num_inputs(), 2);
}

#[test]
fn parse_three_inputs() {
    let expr = EinsumExpr::parse("ij,jk,kl->il").unwrap();
    assert_eq!(expr.num_inputs(), 3);
    assert_eq!(expr.inputs().len(), 3);
    assert_eq!(expr.inputs()[2], vec![b'k', b'l']);
}

#[test]
fn parse_invalid_index_error() {
    let result = EinsumExpr::parse("i1,jk->ik");
    assert!(result.is_err());
}

#[test]
fn parse_output_index_not_in_input() {
    let result = EinsumExpr::parse("ij->iz");
    assert!(result.is_err());
}
