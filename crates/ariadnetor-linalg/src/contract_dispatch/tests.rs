use crate::error::LinalgError;
use crate::{contract, tensordot};
use arnet_native::NativeBackend;
use arnet_tensor::DenseTensor;

/// Dense tensor of the given shape with distinct, non-degenerate entries
/// (`1, 2, 3, …` over the flat buffer) so a contraction result depends on every
/// element. The fill order is irrelevant to the equivalence checks: `tensordot`
/// and `contract` consume the same tensor, so any consistent fill exposes a
/// divergence between the two paths.
fn filled(shape: Vec<usize>) -> DenseTensor<f64> {
    let mut t = DenseTensor::<f64>::zeros(shape);
    for (i, v) in t.data_slice_mut().iter_mut().enumerate() {
        *v = (i + 1) as f64;
    }
    t
}

/// Assert `tensordot(axes)` equals `contract(natural_notation)` in shape and
/// values. The axis-native dense path must emit exactly what the notation path
/// produces for the equivalent natural-order notation — this is the oracle for
/// the natural-order leg ordering.
fn assert_tensordot_matches_contract(
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
    natural_notation: &str,
) {
    let be = NativeBackend::new();
    let a = filled(lhs_shape);
    let b = filled(rhs_shape);
    let td = tensordot(&be, &a, &b, axes_lhs, axes_rhs).unwrap();
    let ct = contract(&be, &a, &b, natural_notation).unwrap();
    assert_eq!(
        td.shape(),
        ct.shape(),
        "tensordot vs contract shape ({natural_notation})"
    );
    for (x, y) in td.data_slice().iter().zip(ct.data_slice()) {
        assert!(
            (x - y).abs() < 1e-12,
            "tensordot vs contract value ({natural_notation}): {x} vs {y}"
        );
    }
}

#[test]
fn dense_tensordot_matmul_matches_contract() {
    assert_tensordot_matches_contract(vec![2, 3], vec![3, 2], &[1], &[0], "ab,bc->ac");
}

#[test]
fn dense_tensordot_rank_generic_matches_contract() {
    // Output rank 3 exercises the rank-greater-than-2 reshape path.
    assert_tensordot_matches_contract(vec![2, 3], vec![3, 4, 5], &[1], &[0], "ab,bcd->acd");
}

#[test]
fn dense_tensordot_multi_axis_matches_contract() {
    // Two contracted axes, non-adjacent on the left (0 and 2).
    assert_tensordot_matches_contract(
        vec![2, 3, 4],
        vec![2, 4, 5],
        &[0, 2],
        &[0, 1],
        "abc,acd->bd",
    );
}

#[test]
fn dense_tensordot_interleaved_rhs_axes_matches_contract() {
    // Right contracted axes interleave with free axes (axes_rhs = [1, 3], free
    // = [0, 2]), so rhs_perm is a non-trivial transpose that does NOT leave the
    // contracted axes as a leading prefix — the case the other multi-axis tests
    // (all axes_rhs = [0, 1]) never reach.
    assert_tensordot_matches_contract(
        vec![2, 3, 4],
        vec![5, 3, 7, 4],
        &[1, 2],
        &[1, 3],
        "abc,dbec->ade",
    );
}

#[test]
fn dense_tensordot_rank4_output_matches_contract() {
    // Output rank 4 drives the rank-greater-than-2 reshape one level deeper than
    // the rank-3 case, and the single contracted right axis (axes_rhs = [2]) sits
    // between free axes, giving a non-prefix rhs_perm.
    assert_tensordot_matches_contract(vec![2, 3], vec![4, 5, 3, 6], &[1], &[2], "ab,cdbe->acde");
}

#[test]
fn dense_tensordot_full_contraction_matches_contract() {
    // No free legs → rank-0 scalar result.
    assert_tensordot_matches_contract(vec![2, 3], vec![2, 3], &[0, 1], &[0, 1], "ab,ab->");
    let be = NativeBackend::new();
    let a = filled(vec![2, 3]);
    let td = tensordot(&be, &a, &a, &[0, 1], &[0, 1]).unwrap();
    assert_eq!(td.shape(), &[] as &[usize]);
}

#[test]
fn dense_tensordot_has_no_leg_count_cap() {
    // The notation path rejected `lhs_rank + free_rhs > 26` (a single-letter
    // label artifact). The axis-native path has no such cap: a rank-27 operand
    // (all dims 1, so one element) contracts cleanly. Output rank is 26 free
    // left legs + 1 free right leg = 27.
    let be = NativeBackend::new();
    let a = filled(vec![1; 27]);
    let b = filled(vec![1, 3]);
    let td = tensordot(&be, &a, &b, &[26], &[0]).unwrap();
    assert_eq!(td.shape().len(), 27);
    assert_eq!(td.shape()[26], 3);
    // a's lone element is 1; the result is a's element times each b entry.
    let got: Vec<f64> = td.data_slice().to_vec();
    assert_eq!(got, vec![1.0, 2.0, 3.0]);
}

#[test]
fn dense_tensordot_rejects_length_mismatch() {
    let be = NativeBackend::new();
    let a = DenseTensor::<f64>::zeros(vec![2, 2, 2]);
    assert!(matches!(
        tensordot(&be, &a, &a, &[0, 1], &[0]),
        Err(LinalgError::InvalidArgument(_))
    ));
}

#[test]
fn dense_tensordot_rejects_out_of_range_axis() {
    let be = NativeBackend::new();
    let a = DenseTensor::<f64>::zeros(vec![2, 2]);
    assert!(matches!(
        tensordot(&be, &a, &a, &[2], &[0]),
        Err(LinalgError::InvalidArgument(_))
    ));
}

#[test]
fn dense_tensordot_rejects_duplicate_axis() {
    let be = NativeBackend::new();
    let a = DenseTensor::<f64>::zeros(vec![2, 2, 2]);
    assert!(matches!(
        tensordot(&be, &a, &a, &[1, 1], &[0, 2]),
        Err(LinalgError::InvalidArgument(_))
    ));
    assert!(matches!(
        tensordot(&be, &a, &a, &[0, 2], &[1, 1]),
        Err(LinalgError::InvalidArgument(_))
    ));
}

#[test]
fn dense_tensordot_rejects_mismatched_contracted_extents() {
    // Paired contracted axes with different extents (lhs axis 1 has dim 3, rhs
    // axis 0 has dim 4) must return InvalidArgument, not panic in the reshape.
    let be = NativeBackend::new();
    let a = DenseTensor::<f64>::zeros(vec![2, 3]);
    let b = DenseTensor::<f64>::zeros(vec![4, 5]);
    assert!(matches!(
        tensordot(&be, &a, &b, &[1], &[0]),
        Err(LinalgError::InvalidArgument(_))
    ));

    // Two contracted pairs whose extents differ per axis (2 vs 3, 6 vs 4) yet
    // whose products coincide (2*6 == 3*4 == 12). This pins the check to a
    // per-pair extent comparison: a coarser `k_lhs == k_rhs` product check would
    // wrongly accept these and panic downstream.
    let c = DenseTensor::<f64>::zeros(vec![2, 6, 7]);
    let d = DenseTensor::<f64>::zeros(vec![3, 4, 8]);
    assert!(matches!(
        tensordot(&be, &c, &d, &[0, 1], &[0, 1]),
        Err(LinalgError::InvalidArgument(_))
    ));
}

#[test]
fn dense_tensordot_zero_length_contracted_axis() {
    // A zero-extent contracted axis is an empty sum: the result is the zero tensor
    // of the free shape, not a panic. The paired `0 == 0` extents pass the per-pair
    // check, so this exercises the degenerate-GEMM short-circuit.
    let be = NativeBackend::new();
    let a = DenseTensor::<f64>::zeros(vec![2, 0]);
    let b = DenseTensor::<f64>::zeros(vec![0, 3]);
    let td = tensordot(&be, &a, &b, &[1], &[0]).unwrap();
    assert_eq!(td.shape(), &[2, 3]);
    assert!(td.data_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn dense_tensordot_zero_length_free_axis() {
    // A zero-extent free axis yields an empty tensor (the shape carries the zero),
    // again via the short-circuit rather than a panic in the GEMM.
    let be = NativeBackend::new();
    let a = DenseTensor::<f64>::zeros(vec![0, 3]);
    let b = DenseTensor::<f64>::zeros(vec![3, 4]);
    let td = tensordot(&be, &a, &b, &[1], &[0]).unwrap();
    assert_eq!(td.shape(), &[0, 4]);
    assert_eq!(td.data_slice().len(), 0);
}

#[test]
fn dense_contract_rejects_single_operand_notation_without_panic() {
    // The dense auto-policy path computes a GEMM-size plan before validating;
    // a single-operand notation must return InvalidArgument, not panic inside
    // ContractionPlan::from_expr (which assumes two operands).
    let be = NativeBackend::new();
    let a = DenseTensor::<f64>::zeros(vec![2, 2]);
    let err = contract(&be, &a, &a, "ii->").unwrap_err();
    assert!(matches!(err, LinalgError::InvalidArgument(_)));
}
