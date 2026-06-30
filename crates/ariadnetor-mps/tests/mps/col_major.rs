//! Column-major MPS integration tests.
//!
//! `NativeBackend`'s preferred order is ColumnMajor, and the dense site
//! constructors materialize in the host substrate's preferred order, so
//! all dense MPS sites here are column-major. The tests below pass a
//! `NativeBackend` at the call site and cover the end-to-end
//! canonicalize / inner / truncate / apply / braket paths on that
//! column-major chain.

use approx::assert_abs_diff_eq;
use ariadnetor_mps::{self as mps, CanonicalForm, TensorChain, TruncSvdParams, TruncateParams};
use ariadnetor_native::NativeBackend;

use super::helpers::{make_4site_mps, make_identity_mpo, mps_to_dense};

#[test]
fn test_col_major_canonicalize_preserves_state() {
    let backend = NativeBackend::new();
    let mps_ref = make_4site_mps();
    let mut mps_cm = make_4site_mps();

    let dense_before = mps_to_dense(&mps_ref);

    mps_cm.canonicalize(&backend, 1);

    let dense_after = mps_to_dense(&mps_cm);
    for (a, b) in dense_before
        .data_slice()
        .iter()
        .zip(dense_after.data_slice().iter())
    {
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }
}

#[test]
fn test_col_major_inner_self_consistent() {
    let backend = NativeBackend::new();
    let mps_ref = make_4site_mps();
    let mps_cm = make_4site_mps();

    let inner_ref = mps::inner(&backend, &mps_ref, &mps_ref);
    let inner_cm = mps::inner(&backend, &mps_cm, &mps_cm);

    assert_abs_diff_eq!(inner_ref, inner_cm, epsilon = 1e-10);
}

#[test]
fn test_col_major_inner_cross() {
    let backend = NativeBackend::new();
    let mps_ref = make_4site_mps();
    let mps_cm = make_4site_mps();

    let inner_rr = mps::inner(&backend, &mps_ref, &mps_ref);
    let inner_rc = mps::inner(&backend, &mps_ref, &mps_cm);

    assert_abs_diff_eq!(inner_rr, inner_rc, epsilon = 1e-10);
}

#[test]
fn test_col_major_truncate_preserves_state() {
    let backend = NativeBackend::new();
    let mps_ref = make_4site_mps();
    let mut mps_cm = make_4site_mps();

    let norm_before = mps_ref.norm(&backend);

    mps_cm.canonicalize(&backend, 1);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: None,
    });
    let result = mps_cm.truncate(&backend, &params);

    assert!(
        result.error / norm_before < 0.5,
        "truncation error too large: {}",
        result.error,
    );

    let overlap = mps::inner(&backend, &mps_ref, &mps_cm);
    let norm_after = mps_cm.norm(&backend);
    assert!(overlap.abs() <= norm_before * norm_after + 1e-10);
    assert!(overlap.abs() > 0.0);
}

#[test]
fn test_col_major_apply_identity() {
    let backend = NativeBackend::new();
    let mps_ref = make_4site_mps();
    let mps_cm = make_4site_mps();
    let identity = make_identity_mpo(4, 2);

    let result = mps::apply(&backend, &identity, &mps_cm, None);

    let dense_ref = mps_to_dense(&mps_ref);
    let dense_result = mps_to_dense(&result);
    for (a, b) in dense_ref
        .data_slice()
        .iter()
        .zip(dense_result.data_slice().iter())
    {
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }
}

#[test]
fn test_col_major_apply_with_truncation() {
    let backend = NativeBackend::new();
    let mps_cm = make_4site_mps();
    let identity = make_identity_mpo(4, 2);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: None,
    });
    let result = mps::apply(&backend, &identity, &mps_cm, Some(&params));

    for d in result.bond_dims() {
        assert!(d <= 3, "bond dim {d} exceeds chi_max=3");
    }
    assert_eq!(*result.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn test_col_major_braket() {
    let backend = NativeBackend::new();
    let mps_ref = make_4site_mps();
    let mps_cm = make_4site_mps();
    let identity = make_identity_mpo(4, 2);

    let braket_ref = mps::braket(&backend, &mps_ref, &identity, &mps_ref);
    let braket_cm = mps::braket(&backend, &mps_cm, &identity, &mps_cm);

    assert_abs_diff_eq!(braket_ref, braket_cm, epsilon = 1e-10);
}
