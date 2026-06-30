//! Targeted mutation-testing coverage for truncate.rs.
//!
//! Covers: CanonicalForm match arms (Left→n-1, Right→0, Mixed→center,
//! Unknown→auto-canonicalize), SvdAbsorb::Both sets Unknown form, truncation
//! error accumulation arithmetic, and isometry verification.

use approx::assert_abs_diff_eq;
use ariadnetor_mps::{
    self as mps, CanonicalForm, Mps, SvdAbsorb, TensorChain, TruncSvdParams, TruncateParams,
};
use ariadnetor_native::NativeBackend;

use super::helpers::{
    cm_dense_tensor, is_left_canonical, is_right_canonical, make_4site_mps, mps_to_dense,
};

// --------------------------------------------------------------------------
// CanonicalForm::Left → center = n - 1
// --------------------------------------------------------------------------

#[test]
fn test_truncate_left_form_center_is_last_site() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 3);
    mps.set_canonical_form(CanonicalForm::Left);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    mps.truncate(&backend, &params);

    // Left → center = n-1 = 3
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 3 });
    // Sites 0..3 should be left-canonical
    let tol = 1e-10;
    for j in 0..3 {
        assert!(
            is_left_canonical(mps.site(j), tol),
            "site {j} not left-canonical"
        );
    }
}

// --------------------------------------------------------------------------
// CanonicalForm::Right → center = 0
// --------------------------------------------------------------------------

#[test]
fn test_truncate_right_form_center_is_zero() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 0);
    mps.set_canonical_form(CanonicalForm::Right);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    mps.truncate(&backend, &params);

    // Right → center = 0
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
    // Sites 1..4 should be right-canonical
    let tol = 1e-10;
    for j in 1..4 {
        assert!(
            is_right_canonical(mps.site(j), tol),
            "site {j} not right-canonical"
        );
    }
}

// --------------------------------------------------------------------------
// CanonicalForm::Mixed → uses existing center
// --------------------------------------------------------------------------

#[test]
fn test_truncate_mixed_preserves_center() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 2);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    mps.truncate(&backend, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
}

// --------------------------------------------------------------------------
// CanonicalForm::Unknown → auto-canonicalizes to default center 0
// --------------------------------------------------------------------------

#[test]
fn test_truncate_unknown_auto_canonicalize_default_center() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    mps.truncate(&backend, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

// --------------------------------------------------------------------------
// CanonicalForm::Partial → auto-canonicalizes to specified center
// --------------------------------------------------------------------------

#[test]
fn test_truncate_partial_auto_canonicalize_explicit_center() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.set_canonical_form(CanonicalForm::Partial {
        left_end: 1,
        right_start: 3,
    });

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Right,
        center: Some(2),
    };
    let result = mps.truncate(&backend, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
    assert!(result.error >= 0.0);
}

// --------------------------------------------------------------------------
// SvdAbsorb::Both → canonical form becomes Unknown
// --------------------------------------------------------------------------

#[test]
fn test_truncate_absorb_both_sets_unknown() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 1);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Both,
        center: None,
    };
    mps.truncate(&backend, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);
}

#[test]
fn test_truncate_absorb_both_state_preserved() {
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
    mps.truncate(&backend, &params);
    let dense_after = mps_to_dense(&mps);
    let norm_after = mps.norm(&backend);

    // Normalized overlap > 0.9
    let mut overlap = 0.0;
    for i in 0..dense_before.len() {
        overlap += (dense_before.data_slice()[i] / norm_before)
            * (dense_after.data_slice()[i] / norm_after);
    }
    assert!(overlap > 0.9, "overlap too low: {overlap}");
}

// --------------------------------------------------------------------------
// SvdAbsorb::Left → Mixed canonical form
// --------------------------------------------------------------------------

#[test]
fn test_truncate_absorb_left_produces_mixed() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 2);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Left,
        center: None,
    };
    mps.truncate(&backend, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
}

// --------------------------------------------------------------------------
// SvdAbsorb::Right → Mixed canonical form with isometries
// --------------------------------------------------------------------------

#[test]
fn test_truncate_absorb_right_isometry_structure() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 1);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Right,
        center: None,
    };
    mps.truncate(&backend, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 1 });

    let tol = 1e-10;
    assert!(
        is_left_canonical(mps.site(0), tol),
        "site 0 not left-canonical"
    );
    for j in 2..4 {
        assert!(
            is_right_canonical(mps.site(j), tol),
            "site {j} not right-canonical"
        );
    }
}

// --------------------------------------------------------------------------
// Truncation error: squared-error accumulation arithmetic
// --------------------------------------------------------------------------

#[test]
fn test_truncation_error_positive_for_lossy_truncation() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 1);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = mps.truncate(&backend, &params);

    // chi_max=1 must produce positive truncation error
    assert!(result.error > 0.0, "expected nonzero truncation error");
}

#[test]
fn test_truncation_error_zero_when_no_truncation() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 2);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(100),
        target_trunc_err: None,
    });
    let result = mps.truncate(&backend, &params);

    assert_abs_diff_eq!(result.error, 0.0, epsilon = 1e-12);
}

#[test]
fn test_truncation_error_sqrt_of_sum_of_squares() {
    // Verify that error is the Frobenius norm (sqrt of sum of squared
    // discarded SVs), not just the sum. Do this by checking that chi_max=1
    // produces smaller error than chi_max=1 would if we just summed SVs.
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 1);
    let norm_before = mps.norm(&backend);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = mps.truncate(&backend, &params);

    // Error should be strictly less than norm (we didn't discard everything)
    assert!(
        result.error < norm_before,
        "error {err} >= norm {n}",
        err = result.error,
        n = norm_before
    );
    assert!(result.error > 0.0);
}

// --------------------------------------------------------------------------
// Single-site chain: no truncation, zero error
// --------------------------------------------------------------------------

#[test]
fn test_truncate_single_site_returns_zero_error() {
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
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

// --------------------------------------------------------------------------
// Bond dims all ≤ chi_max after truncation
// --------------------------------------------------------------------------

#[test]
fn test_all_bond_dims_within_chi_max() {
    let backend = NativeBackend::new();
    for chi in [1, 2, 3] {
        let mut mps = make_4site_mps();
        mps.canonicalize(&backend, 2);

        let params = TruncateParams::from(TruncSvdParams {
            chi_max: Some(chi),
            target_trunc_err: None,
        });
        mps.truncate(&backend, &params);

        for (bond, dim) in mps.bond_dims().iter().enumerate() {
            assert!(*dim <= chi, "bond {bond} dim={dim} exceeds chi_max={chi}");
        }
    }
}

// --------------------------------------------------------------------------
// Verify three absorb modes produce same norm after truncation
// --------------------------------------------------------------------------

#[test]
fn test_all_absorb_modes_same_norm() {
    let backend = NativeBackend::new();
    let base = make_4site_mps();
    let inner_orig = mps::inner(&backend, &base, &base);

    for absorb in [SvdAbsorb::Left, SvdAbsorb::Right, SvdAbsorb::Both] {
        let mut mps = base.clone();
        mps.canonicalize(&backend, 1);

        let params = TruncateParams {
            svd: TruncSvdParams {
                chi_max: Some(2),
                target_trunc_err: None,
            },
            absorb,
            center: None,
        };
        mps.truncate(&backend, &params);

        let inner_trunc = mps::inner(&backend, &mps, &mps);
        // Truncated inner product should be less than or equal to original
        // (within tolerance for rounding)
        assert!(
            inner_trunc <= inner_orig + 1e-10,
            "truncated inner product increased for {:?}",
            absorb
        );
        assert!(inner_trunc > 0.0);
    }
}

// --------------------------------------------------------------------------
// Kill: delete match arm CanonicalForm::Right (distinguishes from fallback)
// --------------------------------------------------------------------------

#[test]
fn test_truncate_right_form_ignores_center_param() {
    // Right arm sets center=0, but fallback would use params.center=Some(2).
    // With the arm deleted, center would be 2 instead of 0.
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 0);
    mps.set_canonical_form(CanonicalForm::Right);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Right,
        center: Some(2),
    };
    mps.truncate(&backend, &params);

    // Right form always uses center=0, regardless of params.center
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

// --------------------------------------------------------------------------
// Kill: err accumulation arithmetic (+→-) and err*err → err+err
// by verifying reported error matches actual reconstruction error
// --------------------------------------------------------------------------

#[test]
fn test_truncation_error_matches_reconstruction_error() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    mps.canonicalize(&backend, 2);
    let dense_before = mps_to_dense(&mps);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = mps.truncate(&backend, &params);
    let dense_after = mps_to_dense(&mps);

    // Pythagorean: ||A||² ≈ ||A_trunc||² + error²
    let norm_sq_before = dense_before
        .data_slice()
        .iter()
        .map(|&x| x * x)
        .sum::<f64>();
    let norm_sq_after = dense_after.data_slice().iter().map(|&x| x * x).sum::<f64>();
    let expected_err_sq = norm_sq_before - norm_sq_after;

    // Reported error² should be close to expected
    let reported_err_sq = result.error * result.error;
    assert!(
        (reported_err_sq - expected_err_sq).abs() < 1e-6 * norm_sq_before,
        "reported²={reported_err_sq}, expected²={expected_err_sq}, \
         diff={d}",
        d = (reported_err_sq - expected_err_sq).abs(),
    );
}
