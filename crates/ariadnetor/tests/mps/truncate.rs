//! Truncation operation tests.

use approx::assert_abs_diff_eq;
use arnet::mps::{
    self, CanonicalForm, Mps, SvdAbsorb, TensorChain, TruncSvdParams, TruncateParams,
};
use arnet_tensor::{DenseTensor, MemoryOrder, TensorStorage};

use super::helpers::{is_left_canonical, is_right_canonical, make_4site_mps, mps_to_dense};

#[test]
fn test_truncate_no_change_within_tolerance() {
    // Build a small MPS, orthogonalize, then truncate with large chi_max
    // Bond dims should stay the same since no truncation is needed
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 2);

    let bond_dims_before = mps.bond_dims();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(100),
        target_trunc_err: None,
    });
    let result = mps::truncate(&mut mps, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
    // Error should be zero (no truncation)
    assert_abs_diff_eq!(result.error, 0.0, epsilon = 1e-12);
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

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let result = mps::truncate(&mut mps, &params);

    // Bond dims should all be ≤ 2
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
    // Truncation error should be positive (we truncated)
    assert!(result.error > 0.0, "expected positive truncation error");
    // Should still be canonicalized
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 1 });
}

#[test]
fn test_truncate_preserves_state_approximately() {
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 1);
    let dense_before = mps_to_dense(&mps);
    let norm_before = mps::norm(&mps);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
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

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(1e-14),
    });
    let result = mps::truncate(&mut mps, &params);

    // With very tight cutoff, truncation error should be very small
    assert!(result.error < 1e-10);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
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

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = mps::truncate(&mut mps, &params);

    assert_abs_diff_eq!(result.error, 0.0, epsilon = 1e-12);
    assert_eq!(mps.len(), 1);
}

#[test]
fn test_truncate_canonical_form_after() {
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 3);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    mps::truncate(&mut mps, &params);

    // Center should be preserved
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 3 });

    // Left sites should be left-canonical
    let tol = 1e-10;
    for j in 0..3 {
        assert!(
            is_left_canonical(mps.storage(j), tol),
            "site {j} not left-canonical after truncate"
        );
    }
}

// ============================================================================
// SvdAbsorb mode tests
// ============================================================================

#[test]
fn test_truncate_absorb_left() {
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 1);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Left,
        center: None,
    };
    let result = mps::truncate(&mut mps, &params);

    assert!(result.error >= 0.0);
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
    // SvdAbsorb::Left still produces Mixed canonical form
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 1 });

    // Right sites should be right-canonical (absorb left means Vt is right-canonical)
    let tol = 1e-10;
    for j in 2..4 {
        assert!(
            is_right_canonical(mps.storage(j), tol),
            "site {j} not right-canonical with SvdAbsorb::Left"
        );
    }
}

#[test]
fn test_truncate_absorb_both() {
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 1);
    let dense_before = mps_to_dense(&mps);
    let norm_before = mps::norm(&mps);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Both,
        center: None,
    };
    let result = mps::truncate(&mut mps, &params);

    assert!(result.error >= 0.0);
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
    // SvdAbsorb::Both does not produce Mixed canonical form
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);

    // State should still be approximately preserved
    let dense_after = mps_to_dense(&mps);
    let norm_after = mps::norm(&mps);
    let mut overlap = 0.0;
    for i in 0..dense_before.len() {
        overlap += (dense_before.data()[i] / norm_before) * (dense_after.data()[i] / norm_after);
    }
    assert!(overlap > 0.9, "overlap too low: {overlap}");
}

// ============================================================================
// Auto-orthogonalize tests
// ============================================================================

#[test]
fn test_truncate_unknown_auto_orthogonalizes() {
    // Truncating an Unknown MPS should auto-orthogonalize, not panic
    let mut mps = make_4site_mps();
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Right,
        center: Some(2),
    };
    let result = mps::truncate(&mut mps, &params);

    assert!(result.error >= 0.0);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

#[test]
fn test_truncate_unknown_default_center() {
    // Without specifying center, auto-orthogonalize defaults to site 0
    let mut mps = make_4site_mps();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    mps::truncate(&mut mps, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn test_truncate_left_canonical_auto() {
    // Left canonical: all sites left-isometric, center should be last site
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 3);
    mps.set_canonical_form(CanonicalForm::Left);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let result = mps::truncate(&mut mps, &params);

    assert!(result.error >= 0.0);
    // Left → center = N-1 = 3
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 3 });
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

#[test]
fn test_truncate_right_canonical_auto() {
    // Right canonical: all sites right-isometric, center should be site 0
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 0);
    mps.set_canonical_form(CanonicalForm::Right);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let result = mps::truncate(&mut mps, &params);

    assert!(result.error >= 0.0);
    // Right → center = 0
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

#[test]
fn test_truncate_error_accumulates_correctly() {
    // Verify truncation error is positive and consistent across sweeps.
    // Truncating to chi_max=1 forces maximal truncation, so error must be
    // strictly positive and the squared-error accumulation (err*err) matters.
    let mut mps = make_4site_mps();
    mps::orthogonalize(&mut mps, 1);
    let norm_before = mps::norm(&mps);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = mps::truncate(&mut mps, &params);

    // With chi_max=1, truncation error must be strictly positive
    assert!(
        result.error > 0.0,
        "expected positive truncation error with chi_max=1"
    );
    // Error should be less than the original norm (we didn't discard everything)
    assert!(
        result.error < norm_before,
        "truncation error {err} exceeds norm {norm}",
        err = result.error,
        norm = norm_before
    );
}
