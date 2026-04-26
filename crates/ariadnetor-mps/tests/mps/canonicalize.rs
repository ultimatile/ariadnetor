//! Canonicalization tests.

use arnet_mps::{self as mps, CanonicalForm, Mps, TensorChain};
use arnet_tensor::{Dense, MemoryOrder};

use super::helpers::{is_left_canonical, is_right_canonical, make_4site_mps, mps_to_dense};

#[test]
fn test_canonicalize_center_0() {
    let mut mps = make_4site_mps();
    let dense_before = mps_to_dense(&mps);

    mps::canonicalize(&mut mps, 0);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });

    // Sites 1..3 should be right-canonical
    let tol = 1e-10;
    for j in 1..4 {
        assert!(
            is_right_canonical(mps.storage(j), tol),
            "site {j} not right-canonical"
        );
    }

    // State vector should be preserved (up to normalization)
    let dense_after = mps_to_dense(&mps);
    let norm_before: f64 = dense_before
        .data()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    let norm_after: f64 = dense_after.data().iter().map(|x| x * x).sum::<f64>().sqrt();
    for i in 0..dense_before.len() {
        let a = dense_before.data()[i] / norm_before;
        let b = dense_after.data()[i] / norm_after;
        assert!(
            (a - b).abs() < 1e-10,
            "state vector changed at index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_canonicalize_center_middle() {
    let mut mps = make_4site_mps();
    let dense_before = mps_to_dense(&mps);

    mps::canonicalize(&mut mps, 2);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });

    let tol = 1e-10;
    // Sites 0, 1 should be left-canonical
    assert!(is_left_canonical(mps.storage(0), tol));
    assert!(is_left_canonical(mps.storage(1), tol));
    // Site 3 should be right-canonical
    assert!(is_right_canonical(mps.storage(3), tol));

    // State vector preserved
    let dense_after = mps_to_dense(&mps);
    let norm_before: f64 = dense_before
        .data()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    let norm_after: f64 = dense_after.data().iter().map(|x| x * x).sum::<f64>().sqrt();
    for i in 0..dense_before.len() {
        let a = dense_before.data()[i] / norm_before;
        let b = dense_after.data()[i] / norm_after;
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_canonicalize_center_last() {
    let mut mps = make_4site_mps();

    mps::canonicalize(&mut mps, 3);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 3 });

    let tol = 1e-10;
    for j in 0..3 {
        assert!(
            is_left_canonical(mps.storage(j), tol),
            "site {j} not left-canonical"
        );
    }
}

#[test]
fn test_canonicalize_single_site() {
    let storages = vec![Dense::new(vec![1.0, 2.0], vec![1, 2, 1])];
    let mut mps = Mps::from_storages(storages);

    mps::canonicalize(&mut mps, 0);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn test_canonicalize_preserves_physical_dims() {
    let mut mps = make_4site_mps();

    let phys_dims: Vec<usize> = (0..4).map(|j| mps.storage(j).shape()[1]).collect();

    mps::canonicalize(&mut mps, 1);

    for (j, &expected) in phys_dims.iter().enumerate() {
        assert_eq!(
            mps.storage(j).shape()[1],
            expected,
            "physical dim changed at site {j}"
        );
    }
}
