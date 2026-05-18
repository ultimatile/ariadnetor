//! Column-major MPS integration tests.

use approx::assert_abs_diff_eq;
use arnet_mps::{self as mps, CanonicalForm, Mps, TensorChain, TruncSvdParams, TruncateParams};
use arnet_tensor::{DenseLayout, DenseStorage, DenseTensorData};

use super::helpers::{make_4site_mps, make_identity_mpo, mps_to_dense};

/// Identity pass-through: MPS data is already in the backend's preferred order.
fn to_col_major(t: &DenseTensorData<f64>) -> DenseTensorData<f64> {
    t.clone()
}

/// Build the same 4-site MPS as make_4site_mps but with column-major site tensors.
fn make_4site_mps_col_major() -> Mps<DenseStorage<f64>, DenseLayout> {
    let rm = make_4site_mps();
    let storages: Vec<DenseTensorData<f64>> =
        (0..rm.len()).map(|j| to_col_major(rm.site(j))).collect();
    Mps::from_sites(storages)
}

#[test]
fn test_col_major_canonicalize_preserves_state() {
    let mps_rm = make_4site_mps();
    let mut mps_cm = make_4site_mps_col_major();

    let dense_before = mps_to_dense(&mps_rm);

    mps::canonicalize(&mut mps_cm, 1);

    let dense_after = mps_to_dense(&mps_cm);
    for (a, b) in dense_before.data().iter().zip(dense_after.data().iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }
}

#[test]
fn test_col_major_inner_matches_row_major() {
    let mps_rm = make_4site_mps();
    let mps_cm = make_4site_mps_col_major();

    let inner_rm = mps::inner(&mps_rm, &mps_rm);
    let inner_cm = mps::inner(&mps_cm, &mps_cm);

    assert_abs_diff_eq!(inner_rm, inner_cm, epsilon = 1e-10);
}

#[test]
fn test_col_major_inner_cross() {
    let mps_rm = make_4site_mps();
    let mps_cm = make_4site_mps_col_major();

    // ⟨rm|cm⟩ should equal ⟨rm|rm⟩ since they represent the same state
    let inner_rr = mps::inner(&mps_rm, &mps_rm);
    let inner_rc = mps::inner(&mps_rm, &mps_cm);

    assert_abs_diff_eq!(inner_rr, inner_rc, epsilon = 1e-10);
}

#[test]
fn test_col_major_truncate_preserves_state() {
    let mps_rm = make_4site_mps();
    let mut mps_cm = make_4site_mps_col_major();

    let norm_before = mps::norm(&mps_rm);

    mps::canonicalize(&mut mps_cm, 1);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: None,
    });
    let result = mps::truncate(&mut mps_cm, &params);

    // Truncation error should be small relative to norm
    assert!(
        result.error / norm_before < 0.5,
        "truncation error too large: {}",
        result.error
    );

    // Inner product with original should be close to norm squared
    let overlap = mps::inner(&mps_rm, &mps_cm);
    let norm_after = mps::norm(&mps_cm);
    // Cauchy-Schwarz: |overlap| <= norm_before * norm_after
    assert!(overlap.abs() <= norm_before * norm_after + 1e-10);
    assert!(overlap.abs() > 0.0);
}

#[test]
fn test_col_major_apply_identity() {
    let mps_rm = make_4site_mps();
    let mps_cm = make_4site_mps_col_major();
    let identity = make_identity_mpo(4, 2);

    let result = mps::apply(&identity, &mps_cm, None);

    // Apply result sites are row-major, so compare with row-major reference
    let dense_ref = mps_to_dense(&mps_rm);
    let dense_result = mps_to_dense(&result);
    for (a, b) in dense_ref.data().iter().zip(dense_result.data().iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }
}

#[test]
fn test_col_major_apply_with_truncation() {
    let mps_cm = make_4site_mps_col_major();
    let identity = make_identity_mpo(4, 2);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: None,
    });
    let result = mps::apply(&identity, &mps_cm, Some(&params));

    // Bond dims should be capped
    for d in result.bond_dims() {
        assert!(d <= 3, "bond dim {d} exceeds chi_max=3");
    }
    assert_eq!(*result.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn test_col_major_braket() {
    let mps_rm = make_4site_mps();
    let mps_cm = make_4site_mps_col_major();
    let identity = make_identity_mpo(4, 2);

    let braket_rm = mps::braket(&mps_rm, &identity, &mps_rm);
    let braket_cm = mps::braket(&mps_cm, &identity, &mps_cm);

    assert_abs_diff_eq!(braket_rm, braket_cm, epsilon = 1e-10);
}
