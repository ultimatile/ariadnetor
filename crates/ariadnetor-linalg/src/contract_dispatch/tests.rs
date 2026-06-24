use super::tensordot_notation;
use crate::contract_spec::ContractSpec;
use crate::error::LinalgError;
use crate::{contract, tensordot};
use arnet_native::NativeBackend;
use arnet_tensor::DenseTensor;

fn notation(lhs_rank: usize, rhs_rank: usize, al: &[usize], ar: &[usize]) -> String {
    tensordot_notation(lhs_rank, rhs_rank, al, ar).expect("valid axes")
}

#[test]
fn matmul_notation() {
    assert_eq!(notation(2, 2, &[1], &[0]), "ab,bc->ac");
}

#[test]
fn keeps_rank_generic_site_legs() {
    // absorb_from_left_bsp: factor(2) axis 1 vs next(3) axis 0 → output rank 3.
    assert_eq!(notation(2, 3, &[1], &[0]), "ab,bcd->acd");
    // Same contraction against a rank-4 next stays general.
    assert_eq!(notation(2, 4, &[1], &[0]), "ab,bcde->acde");
}

#[test]
fn multi_axis_natural_order() {
    // env(3) axes 0,2 vs phi(3) axes 0,1 (braket step 3).
    let n = notation(3, 3, &[0, 2], &[0, 1]);
    // lhs labels abc; contracted a(0),c(2) inherit to rhs axes 0,1 → rhs a,c,?;
    // rhs free axis 2 → fresh 'd'. out = lhs free (b) then rhs free (d).
    assert_eq!(n, "abc,acd->bd");
}

#[test]
fn full_contraction_empty_output() {
    assert_eq!(notation(2, 2, &[0, 1], &[0, 1]), "ab,ab->");
}

#[test]
fn notation_roundtrips_through_spec_with_no_reorder() {
    // The notation tensordot emits must request exactly the natural order, so a
    // ContractSpec parse yields natural == out (no permute pass).
    for (lr, rr, al, ar) in [
        (2usize, 3usize, &[1usize][..], &[0usize][..]),
        (3, 3, &[0, 2][..], &[0, 1][..]),
        (3, 4, &[1, 2][..], &[1, 3][..]),
    ] {
        let n = notation(lr, rr, al, ar);
        let spec = ContractSpec::from_notation(&n).expect("valid");
        assert_eq!(
            spec.natural_labels, spec.out_labels,
            "tensordot notation {n} must be natural-order"
        );
        assert_eq!(spec.axes_lhs, al);
        assert_eq!(spec.axes_rhs, ar);
    }
}

#[test]
fn rejects_length_mismatch() {
    assert!(matches!(
        tensordot_notation(3, 3, &[0, 1], &[0]),
        Err(LinalgError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_out_of_range_axis() {
    assert!(matches!(
        tensordot_notation(2, 2, &[2], &[0]),
        Err(LinalgError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_duplicate_axis() {
    assert!(matches!(
        tensordot_notation(3, 3, &[1, 1], &[0, 2]),
        Err(LinalgError::InvalidArgument(_))
    ));
    assert!(matches!(
        tensordot_notation(3, 3, &[0, 2], &[1, 1]),
        Err(LinalgError::InvalidArgument(_))
    ));
}

#[test]
fn dense_tensordot_matches_contract_with_natural_notation() {
    // The dense `tensordot` impl builds natural-order notation and routes
    // through `contract`; confirm it equals the explicit natural notation.
    let be = NativeBackend::new();
    let mut a = DenseTensor::<f64>::zeros(vec![2, 3]);
    for (i, v) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().enumerate() {
        a.set([i / 3, i % 3], v);
    }
    let mut b = DenseTensor::<f64>::zeros(vec![3, 2]);
    for (i, v) in [7.0, 8.0, 9.0, 10.0, 11.0, 12.0].into_iter().enumerate() {
        b.set([i / 2, i % 2], v);
    }

    let td = tensordot(&be, &a, &b, &[1], &[0]).unwrap();
    let ct = contract(&be, &a, &b, "ab,bc->ac").unwrap();
    assert_eq!(td.shape(), ct.shape());
    for (x, y) in td.data_slice().iter().zip(ct.data_slice()) {
        assert!((x - y).abs() < 1e-12, "tensordot vs contract: {x} vs {y}");
    }
}
