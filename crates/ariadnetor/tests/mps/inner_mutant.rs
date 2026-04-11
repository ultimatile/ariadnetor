//! Targeted mutation-testing coverage for inner.rs norm function.
//!
//! Covers the norm match arms: Left|Right returning 1.0, and
//! Mixed{center} returning the center tensor's Frobenius norm.

use approx::assert_abs_diff_eq;
use arnet::mps::{self, CanonicalForm, Mps, TensorChain};
use arnet_tensor::{Dense, MemoryOrder};

use super::helpers::make_4site_mps;

// --------------------------------------------------------------------------
// norm: Left canonical form → returns exactly 1.0
// --------------------------------------------------------------------------

#[test]
fn test_norm_left_returns_one_not_computed() {
    let mut mps = make_4site_mps();
    mps::canonicalize(&mut mps, 3);
    mps.set_canonical_form(CanonicalForm::Left);

    let n = mps::norm(&mps);
    assert_abs_diff_eq!(n, 1.0, epsilon = 1e-15);
}

// --------------------------------------------------------------------------
// norm: Right canonical form → returns exactly 1.0
// --------------------------------------------------------------------------

#[test]
fn test_norm_right_returns_one_not_computed() {
    let mut mps = make_4site_mps();
    mps::canonicalize(&mut mps, 0);
    mps.set_canonical_form(CanonicalForm::Right);

    let n = mps::norm(&mps);
    assert_abs_diff_eq!(n, 1.0, epsilon = 1e-15);
}

// --------------------------------------------------------------------------
// norm: Mixed{center=0} — should use storage(0).norm()
// --------------------------------------------------------------------------

#[test]
fn test_norm_mixed_center_0() {
    let mut mps = make_4site_mps();
    mps::canonicalize(&mut mps, 0);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });

    let n = mps::norm(&mps);
    let expected = mps.storage(0).norm();
    assert_abs_diff_eq!(n, expected, epsilon = 1e-12);
    // Must be positive
    assert!(n > 0.0);
}

// --------------------------------------------------------------------------
// norm: Mixed{center=N-1} — should use storage(N-1).norm()
// --------------------------------------------------------------------------

#[test]
fn test_norm_mixed_center_last() {
    let mut mps = make_4site_mps();
    mps::canonicalize(&mut mps, 3);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 3 });

    let n = mps::norm(&mps);
    let expected = mps.storage(3).norm();
    assert_abs_diff_eq!(n, expected, epsilon = 1e-12);
}

// --------------------------------------------------------------------------
// norm: Mixed{center=1} — middle site
// --------------------------------------------------------------------------

#[test]
fn test_norm_mixed_center_1() {
    let mut mps = make_4site_mps();
    mps::canonicalize(&mut mps, 1);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 1 });

    let n_mixed = mps::norm(&mps);
    let center_norm = mps.storage(1).norm();
    assert_abs_diff_eq!(n_mixed, center_norm, epsilon = 1e-12);

    // Also verify consistency with full contraction
    mps.set_canonical_form(CanonicalForm::Unknown);
    let n_full = mps::norm(&mps);
    assert_abs_diff_eq!(n_mixed, n_full, epsilon = 1e-10);
}

// --------------------------------------------------------------------------
// norm: Unknown/Partial → uses full contraction
// --------------------------------------------------------------------------

#[test]
fn test_norm_unknown_uses_full_contraction() {
    let mps = make_4site_mps();
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);

    let n = mps::norm(&mps);
    // Should be nonzero and positive
    assert!(n > 0.0);
    // Verify it equals sqrt(inner(psi,psi))
    let overlap = mps::inner(&mps, &mps);
    assert_abs_diff_eq!(n, overlap.sqrt(), epsilon = 1e-12);
}

// --------------------------------------------------------------------------
// norm: Product state |+⟩^N — exact value verification
// --------------------------------------------------------------------------

#[test]
fn test_norm_plus_state_exact() {
    // |+⟩ = (|0⟩ + |1⟩)/sqrt(2), but we store unnormalized [1,1]
    // for 2 sites: norm = sqrt(sum of all products) = 2.0
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let storages = vec![
        Dense::from_data_with_order(
            vec![inv_sqrt2, inv_sqrt2],
            vec![1, 2, 1],
            MemoryOrder::RowMajor,
        ),
        Dense::from_data_with_order(
            vec![inv_sqrt2, inv_sqrt2],
            vec![1, 2, 1],
            MemoryOrder::RowMajor,
        ),
    ];
    let psi = Mps::from_storages(storages);
    let n = mps::norm(&psi);
    // |+⟩|+⟩ is normalized: ⟨++|++⟩ = 1
    assert_abs_diff_eq!(n, 1.0, epsilon = 1e-12);
}
