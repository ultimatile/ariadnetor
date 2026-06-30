//! Canonicalization tests.

use ariadnetor_mps::{CanonicalForm, Mps, TensorChain};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{ComputeBackendTensorExt, DenseLayout, DenseStorage, Host};

use super::helpers::{is_left_canonical, is_right_canonical, make_4site_mps, mps_to_dense};

#[test]
fn test_canonicalize_center_0() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    let dense_before = mps_to_dense(&mps);

    mps.canonicalize(&backend, 0);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });

    let tol = 1e-10;
    for j in 1..4 {
        assert!(
            is_right_canonical(mps.site(j), tol),
            "site {j} not right-canonical"
        );
    }

    let dense_after = mps_to_dense(&mps);
    let norm_before: f64 = dense_before
        .data_slice()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    let norm_after: f64 = dense_after
        .data_slice()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    for i in 0..dense_before.len() {
        let a = dense_before.data_slice()[i] / norm_before;
        let b = dense_after.data_slice()[i] / norm_after;
        assert!(
            (a - b).abs() < 1e-10,
            "state vector changed at index {i}: {a} vs {b}",
        );
    }
}

#[test]
fn test_canonicalize_center_middle() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();
    let dense_before = mps_to_dense(&mps);

    mps.canonicalize(&backend, 2);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });

    let tol = 1e-10;
    assert!(is_left_canonical(mps.site(0), tol));
    assert!(is_left_canonical(mps.site(1), tol));
    assert!(is_right_canonical(mps.site(3), tol));

    let dense_after = mps_to_dense(&mps);
    let norm_before: f64 = dense_before
        .data_slice()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    let norm_after: f64 = dense_after
        .data_slice()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    for i in 0..dense_before.len() {
        let a = dense_before.data_slice()[i] / norm_before;
        let b = dense_after.data_slice()[i] / norm_after;
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_canonicalize_center_last() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();

    mps.canonicalize(&backend, 3);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 3 });

    let tol = 1e-10;
    for j in 0..3 {
        assert!(
            is_left_canonical(mps.site(j), tol),
            "site {j} not left-canonical"
        );
    }
}

#[test]
fn test_canonicalize_single_site() {
    let backend = NativeBackend::new();
    let site = Host::shared().dense(vec![1.0, 2.0], vec![1, 2, 1]);
    let mut mps: Mps<DenseStorage<f64>, DenseLayout> = Mps::from_sites(vec![site]);

    mps.canonicalize(&backend, 0);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn test_canonicalize_preserves_physical_dims() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_mps();

    let phys_dims: Vec<usize> = (0..4).map(|j| mps.site(j).shape()[1]).collect();

    mps.canonicalize(&backend, 1);

    for (j, &expected) in phys_dims.iter().enumerate() {
        assert_eq!(
            mps.site(j).shape()[1],
            expected,
            "physical dim changed at site {j}",
        );
    }
}
