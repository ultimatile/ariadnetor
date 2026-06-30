//! Truncation operation tests.

use approx::assert_abs_diff_eq;
use ariadnetor_mps::{CanonicalForm, Mps, SvdAbsorb, TensorChain, TruncSvdParams, TruncateParams};
use ariadnetor_native::NativeBackend;

use super::helpers::{
    cm_dense_tensor, is_left_canonical, is_right_canonical, make_4site_mps, mps_to_dense,
};

#[test]
fn test_truncate_no_change_within_tolerance() {
    // Build a small MPS, canonicalize, then truncate with large chi_max
    // Bond dims should stay the same since no truncation is needed
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 2);

    let bond_dims_before = mps.bond_dims();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(100),
        target_trunc_err: None,
    });
    let result = mps.truncate(&backend, &params);

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
    // Build MPS with large bond dims, canonicalize, then truncate to chi_max=2
    let backend = NativeBackend::new();
    let storages = vec![
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 4]),
        cm_dense_tensor((1..=32).map(|i| i as f64 * 0.1).collect(), vec![4, 2, 4]),
        cm_dense_tensor((1..=32).map(|i| i as f64 * 0.01).collect(), vec![4, 2, 4]),
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.1).collect(), vec![4, 2, 1]),
    ];
    let mut mps = Mps::from_sites(storages);
    mps.canonicalize(&backend, 1);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let result = mps.truncate(&backend, &params);

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
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 1);
    let dense_before = mps_to_dense(&mps);
    let norm_before = mps.norm(&backend);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    mps.truncate(&backend, &params);
    let dense_after = mps_to_dense(&mps);
    let norm_after = mps.norm(&backend);

    // Normalize and compute overlap between original and truncated
    let mut overlap = 0.0;
    for i in 0..dense_before.len() {
        overlap += (dense_before.data_slice()[i] / norm_before)
            * (dense_after.data_slice()[i] / norm_after);
    }
    // Overlap should be close to 1 (truncation removes small components)
    assert!(overlap > 0.9, "overlap too low: {overlap}");
}

#[test]
fn test_truncate_with_cutoff() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 0);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(1e-14),
    });
    let result = mps.truncate(&backend, &params);

    // With very tight cutoff, truncation error should be very small
    assert!(result.error < 1e-10);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn test_truncate_single_site() {
    let backend = NativeBackend::new();
    let storages = vec![cm_dense_tensor(vec![3.0, 4.0], vec![1, 2, 1])];
    let mut mps = Mps::from_sites(storages);
    mps.canonicalize(&backend, 0);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = mps.truncate(&backend, &params);

    assert_abs_diff_eq!(result.error, 0.0, epsilon = 1e-12);
    assert_eq!(mps.len(), 1);
}

#[test]
fn test_truncate_canonical_form_after() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 3);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    mps.truncate(&backend, &params);

    // Center should be preserved
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 3 });

    // Left sites should be left-canonical
    let tol = 1e-10;
    for j in 0..3 {
        assert!(
            is_left_canonical(mps.site(j), tol),
            "site {j} not left-canonical after truncate"
        );
    }
}

// ============================================================================
// SvdAbsorb mode tests
// ============================================================================

#[test]
fn test_truncate_absorb_left() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 1);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Left,
        center: None,
    };
    let result = mps.truncate(&backend, &params);

    assert!(result.error >= 0.0);
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
    // SvdAbsorb::Left still produces mixed-canonical form
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 1 });

    let tol = 1e-10;
    assert!(
        is_left_canonical(mps.site(0), tol),
        "site 0 not left-canonical with SvdAbsorb::Left"
    );
    for j in 2..4 {
        assert!(
            is_right_canonical(mps.site(j), tol),
            "site {j} not right-canonical with SvdAbsorb::Left"
        );
    }
}

#[test]
fn test_truncate_absorb_both() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 1);
    let dense_before = mps_to_dense(&mps);
    let norm_before = mps.norm(&backend);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Both,
        center: None,
    };
    let result = mps.truncate(&backend, &params);

    assert!(result.error >= 0.0);
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
    // SvdAbsorb::Both does not produce Mixed canonical form
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);

    // State should still be approximately preserved
    let dense_after = mps_to_dense(&mps);
    let norm_after = mps.norm(&backend);
    let mut overlap = 0.0;
    for i in 0..dense_before.len() {
        overlap += (dense_before.data_slice()[i] / norm_before)
            * (dense_after.data_slice()[i] / norm_after);
    }
    assert!(overlap > 0.9, "overlap too low: {overlap}");
}

// ============================================================================
// Auto-canonicalize tests
// ============================================================================

#[test]
fn test_truncate_unknown_auto_canonicalizes() {
    // Truncating an Unknown MPS should auto-canonicalize, not panic
    let backend = NativeBackend::new();
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
    let result = mps.truncate(&backend, &params);

    assert!(result.error >= 0.0);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

#[test]
fn test_truncate_unknown_default_center() {
    // Without specifying center, auto-canonicalize defaults to site 0
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    mps.truncate(&backend, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn test_truncate_left_canonical_auto() {
    // Left canonical: all sites left-isometric, center should be last site
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 3);
    mps.set_canonical_form(CanonicalForm::Left);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let result = mps.truncate(&backend, &params);

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
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 0);
    mps.set_canonical_form(CanonicalForm::Right);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let result = mps.truncate(&backend, &params);

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
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 1);
    let norm_before = mps.norm(&backend);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = mps.truncate(&backend, &params);

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

#[test]
fn test_absorb_left_differs_from_right() {
    let backend = NativeBackend::new();
    let mut mps_l = make_4site_mps();
    mps_l.canonicalize(&backend, 1);
    let mut mps_r = mps_l.clone();

    let params_l = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Left,
        center: None,
    };
    let params_r = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    mps_l.truncate(&backend, &params_l);
    mps_r.truncate(&backend, &params_r);

    // Center tensors should differ between Left and Right
    let center_l = mps_l.site(1);
    let center_r = mps_r.site(1);
    let max_diff = center_l
        .data_slice()
        .iter()
        .zip(center_r.data_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 1e-10,
        "Left and Right should produce different center tensors"
    );
}
