use super::*;

fn spec(notation: &str) -> ContractSpec {
    ContractSpec::from_notation(notation).expect("valid notation")
}

#[test]
fn matmul_axes_and_natural_order() {
    let s = spec("ab,bc->ac");
    assert_eq!(s.axes_lhs, vec![1]);
    assert_eq!(s.axes_rhs, vec![0]);
    assert_eq!(s.natural_labels, b"ac".to_vec());
    assert_eq!(s.out_labels, b"ac".to_vec());
}

#[test]
fn natural_order_is_input_axis_order_not_output_order() {
    // Reordering notation: rhs-free label `e` requested between lhs-free labels.
    let s = spec("abcd,ebf->aecdf");
    // contracted: b (lhs axis 1, rhs axis 1)
    assert_eq!(s.axes_lhs, vec![1]);
    assert_eq!(s.axes_rhs, vec![1]);
    // natural = free-lhs (a,c,d) then free-rhs (e,f)
    assert_eq!(s.natural_labels, b"acdef".to_vec());
    // requested differs → caller permutes
    assert_eq!(s.out_labels, b"aecdf".to_vec());
}

#[test]
fn multi_axis_contraction_pairing() {
    // lhs bd contracted against rhs bd (shared labels b,d).
    let s = spec("bcde,bfdg->cefg");
    assert_eq!(s.axes_lhs, vec![0, 2]); // b at 0, d at 2 in "bcde"
    assert_eq!(s.axes_rhs, vec![0, 2]); // b at 0, d at 2 in "bfdg"
    assert_eq!(s.natural_labels, b"cefg".to_vec());
    assert_eq!(s.out_labels, b"cefg".to_vec());
}

#[test]
fn full_contraction_has_empty_orders() {
    let s = spec("ab,ab->");
    assert_eq!(s.axes_lhs, vec![0, 1]);
    assert_eq!(s.axes_rhs, vec![0, 1]);
    assert!(s.natural_labels.is_empty());
    assert!(s.out_labels.is_empty());
}

#[test]
fn rejects_single_operand_notation() {
    assert!(ContractSpec::from_notation("ii->").is_err());
}

#[test]
fn rejects_partial_trace_repeat_within_operand() {
    // `i` repeated within the left operand.
    let result = ContractSpec::from_notation("iij,jk->ik");
    assert!(matches!(result, Err(LinalgError::InvalidArgument(_))));
}

#[test]
fn rejects_batch_index() {
    // `a` shared by both operands and kept in the output.
    let result = ContractSpec::from_notation("abc,abd->abcd");
    assert!(matches!(result, Err(LinalgError::InvalidArgument(_))));
}

#[test]
fn rejects_implicit_reduction() {
    // `b` is in the left operand only and not in the output.
    let result = ContractSpec::from_notation("ab,cd->ac");
    assert!(matches!(result, Err(LinalgError::InvalidArgument(_))));
}
