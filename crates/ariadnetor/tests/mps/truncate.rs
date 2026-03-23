//! Truncation operation tests.

use approx::assert_abs_diff_eq;
use arnet::mps::{self, CanonicalForm, Mps, TensorChain};
use arnet_tensor::{DenseTensor, MemoryOrder, TensorStorage};

use super::helpers::{is_left_canonical, make_4site_mps, mps_to_dense};

#[test]
fn test_truncate_no_change_within_tolerance() {
    // Build a small MPS, orthogonalize, then truncate with large chi_max
    // Bond dims should stay the same since no truncation is needed
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 2);

    let bond_dims_before = mps.bond_dims();

    let params = mps::TruncSvdParams {
        chi_max: Some(100),
        target_trunc_err: None,
    };
    let err = mps::truncate(&mut mps, &params);

    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Canonicalized { center: 2 }
    );
    // Error should be zero (no truncation)
    assert_abs_diff_eq!(err, 0.0, epsilon = 1e-12);
    // Bond dims should not increase (may decrease if rank-deficient)
    for (before, after) in bond_dims_before.iter().zip(mps.bond_dims().iter()) {
        assert!(*after <= *before);
    }
}

#[test]
fn test_truncate_reduces_bond_dim() {
    // Build MPS with large bond dims, orthogonalize, then truncate to chi_max=2
    let storages = vec![
        TensorStorage::Dense(DenseTensor::from_data_with_order(
            (1..=8).map(|i| i as f64 * 0.1).collect(),
            vec![1, 2, 4],
            MemoryOrder::RowMajor,
        )),
        TensorStorage::Dense(DenseTensor::from_data_with_order(
            (1..=32).map(|i| i as f64 * 0.1).collect(),
            vec![4, 2, 4],
            MemoryOrder::RowMajor,
        )),
        TensorStorage::Dense(DenseTensor::from_data_with_order(
            (1..=32).map(|i| i as f64 * 0.01).collect(),
            vec![4, 2, 4],
            MemoryOrder::RowMajor,
        )),
        TensorStorage::Dense(DenseTensor::from_data_with_order(
            (1..=8).map(|i| i as f64 * 0.1).collect(),
            vec![4, 2, 1],
            MemoryOrder::RowMajor,
        )),
    ];
    let mut mps = Mps::from_storages(storages);
    mps::orthogonalize(&mut mps, 1);

    let params = mps::TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    };
    let err = mps::truncate(&mut mps, &params);

    // Bond dims should all be ≤ 2
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
    // Truncation error should be positive (we truncated)
    assert!(err > 0.0, "expected positive truncation error");
    // Should still be canonicalized
    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Canonicalized { center: 1 }
    );
}

#[test]
fn test_truncate_preserves_state_approximately() {
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 1);
    let dense_before = mps_to_dense(&mps);
    let norm_before = mps::norm(&mps);

    let params = mps::TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    };
    mps::truncate(&mut mps, &params);
    let dense_after = mps_to_dense(&mps);
    let norm_after = mps::norm(&mps);

    // Normalize and compute overlap between original and truncated
    let mut overlap = 0.0;
    for i in 0..dense_before.len() {
        overlap += (dense_before.data()[i] / norm_before) * (dense_after.data()[i] / norm_after);
    }
    // Overlap should be close to 1 (truncation removes small components)
    assert!(overlap > 0.9, "overlap too low: {overlap}");
}

#[test]
fn test_truncate_with_cutoff() {
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 0);

    let params = mps::TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(1e-14),
    };
    let err = mps::truncate(&mut mps, &params);

    // With very tight cutoff, truncation error should be very small
    assert!(err < 1e-10);
    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Canonicalized { center: 0 }
    );
}

#[test]
fn test_truncate_single_site() {
    let storages = vec![TensorStorage::Dense(DenseTensor::from_data_with_order(
        vec![3.0, 4.0],
        vec![1, 2, 1],
        MemoryOrder::RowMajor,
    ))];
    let mut mps = Mps::from_storages(storages);
    mps::orthogonalize(&mut mps, 0);

    let params = mps::TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let err = mps::truncate(&mut mps, &params);

    assert_abs_diff_eq!(err, 0.0, epsilon = 1e-12);
    assert_eq!(mps.len(), 1);
}

#[test]
fn test_truncate_canonical_form_after() {
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 3);

    let params = mps::TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    };
    mps::truncate(&mut mps, &params);

    // Center should be preserved
    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Canonicalized { center: 3 }
    );

    // Left sites should be left-canonical
    let tol = 1e-10;
    for j in 0..3 {
        assert!(
            is_left_canonical(mps.storage(j), tol),
            "site {j} not left-canonical after truncate"
        );
    }
}
