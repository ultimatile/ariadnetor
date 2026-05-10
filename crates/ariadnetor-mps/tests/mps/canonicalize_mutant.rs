//! Targeted mutation-testing coverage for canonicalize.rs.
//!
//! Tests arithmetic correctness in left_qr_step, right_lq_step,
//! absorb_from_left, and absorb_from_right by verifying isometry
//! properties and state preservation with exact assertions.

use approx::assert_abs_diff_eq;
use arnet_mps::{self as mps, CanonicalForm, Mps, TensorChain};
use arnet_tensor::{Dense, MemoryOrder};

use super::helpers::{is_left_canonical, is_right_canonical, make_4site_mps, mps_to_dense};

// --------------------------------------------------------------------------
// left_qr_step: verify each site becomes left-isometric after sweep
// --------------------------------------------------------------------------

#[test]
fn test_left_sweep_all_sites_left_canonical() {
    let mut mps = make_4site_mps();
    mps::canonicalize(&mut mps, 3);

    let tol = 1e-10;
    // Sites 0, 1, 2 must be left-canonical (Q^T Q = I)
    for j in 0..3 {
        assert!(
            is_left_canonical(mps.storage(j), tol),
            "site {j} not left-canonical after center=3"
        );
    }
}

// --------------------------------------------------------------------------
// right_lq_step: verify each site becomes right-isometric after sweep
// --------------------------------------------------------------------------

#[test]
fn test_right_sweep_all_sites_right_canonical() {
    let mut mps = make_4site_mps();
    mps::canonicalize(&mut mps, 0);

    let tol = 1e-10;
    // Sites 1, 2, 3 must be right-canonical (Q Q^T = I)
    for j in 1..4 {
        assert!(
            is_right_canonical(mps.storage(j), tol),
            "site {j} not right-canonical after center=0"
        );
    }
}

// --------------------------------------------------------------------------
// Mixed canonical form: left and right regions for center=1
// --------------------------------------------------------------------------

#[test]
fn test_mixed_canonical_center_1() {
    let mut mps = make_4site_mps();
    mps::canonicalize(&mut mps, 1);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 1 });

    let tol = 1e-10;
    // Site 0 should be left-canonical
    assert!(
        is_left_canonical(mps.storage(0), tol),
        "site 0 not left-canonical"
    );
    // Sites 2, 3 should be right-canonical
    for j in 2..4 {
        assert!(
            is_right_canonical(mps.storage(j), tol),
            "site {j} not right-canonical"
        );
    }
}

// --------------------------------------------------------------------------
// State preservation: inner product unchanged after canonicalize
// --------------------------------------------------------------------------

#[test]
fn test_canonicalize_preserves_inner_product_center_0() {
    let mps = make_4site_mps();
    let inner_before = mps::inner(&mps, &mps);

    let mut mps_canon = mps.clone();
    mps::canonicalize(&mut mps_canon, 0);
    let inner_after = mps::inner(&mps_canon, &mps_canon);

    assert_abs_diff_eq!(inner_before, inner_after, epsilon = 1e-10);
}

#[test]
fn test_canonicalize_preserves_inner_product_center_3() {
    let mps = make_4site_mps();
    let inner_before = mps::inner(&mps, &mps);

    let mut mps_canon = mps.clone();
    mps::canonicalize(&mut mps_canon, 3);
    let inner_after = mps::inner(&mps_canon, &mps_canon);

    assert_abs_diff_eq!(inner_before, inner_after, epsilon = 1e-10);
}

// --------------------------------------------------------------------------
// State vector equivalence (absorb_from_left / absorb_from_right)
// --------------------------------------------------------------------------

#[test]
fn test_state_vector_preserved_center_2() {
    let mut mps = make_4site_mps();
    let dense_before = mps_to_dense(&mps);

    mps::canonicalize(&mut mps, 2);
    let dense_after = mps_to_dense(&mps);

    // Normalize both and compare
    let norm_b: f64 = dense_before
        .data()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    let norm_a: f64 = dense_after.data().iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm_b > 0.0);
    assert!(norm_a > 0.0);

    for i in 0..dense_before.len() {
        let a = dense_before.data()[i] / norm_b;
        let b = dense_after.data()[i] / norm_a;
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }
}

// --------------------------------------------------------------------------
// Shape preservation: physical dims unchanged
// --------------------------------------------------------------------------

#[test]
fn test_physical_dims_preserved_all_centers() {
    for center in 0..4 {
        let mut mps = make_4site_mps();
        let phys: Vec<_> = (0..4).map(|j| mps.storage(j).shape()[1]).collect();

        mps::canonicalize(&mut mps, center);

        for (j, &expected) in phys.iter().enumerate() {
            assert_eq!(
                mps.storage(j).shape()[1],
                expected,
                "physical dim changed at site {j} with center={center}"
            );
        }
    }
}

// --------------------------------------------------------------------------
// Rank preservation: all tensors remain rank-3
// --------------------------------------------------------------------------

#[test]
fn test_tensors_remain_rank_3() {
    let mut mps = make_4site_mps();
    mps::canonicalize(&mut mps, 2);

    for j in 0..4 {
        assert_eq!(
            mps.storage(j).rank(),
            3,
            "site {j} rank changed after canonicalize"
        );
    }
}

// --------------------------------------------------------------------------
// Bond compatibility: right-bond of site j == left-bond of site j+1
// --------------------------------------------------------------------------

#[test]
fn test_bond_dim_compatibility_after_canonicalize() {
    for center in 0..4 {
        let mut mps = make_4site_mps();
        mps::canonicalize(&mut mps, center);

        for j in 0..3 {
            let right_bond = *mps.storage(j).shape().last().unwrap();
            let left_bond = mps.storage(j + 1).shape()[0];
            assert_eq!(
                right_bond,
                left_bond,
                "bond mismatch between sites {j} and {} for center={center}",
                j + 1
            );
        }
    }
}

// --------------------------------------------------------------------------
// Two-site MPS: minimal case for absorb_from_left / absorb_from_right
// --------------------------------------------------------------------------

#[test]
fn test_two_site_canonicalize_center_0() {
    let storages = vec![
        Dense::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 2, 2],
            MemoryOrder::ColumnMajor,
        ),
        Dense::new(
            vec![1.0, 0.5, 0.3, 0.1],
            vec![2, 2, 1],
            MemoryOrder::ColumnMajor,
        ),
    ];
    let mut mps = Mps::from_storages(storages);
    let dense_before = mps_to_dense(&mps);
    let norm_before: f64 = dense_before
        .data()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();

    mps::canonicalize(&mut mps, 0);

    // Site 1 should be right-canonical
    assert!(is_right_canonical(mps.storage(1), 1e-10));

    // State preserved
    let dense_after = mps_to_dense(&mps);
    let norm_after: f64 = dense_after.data().iter().map(|x| x * x).sum::<f64>().sqrt();
    for i in 0..dense_before.len() {
        assert_abs_diff_eq!(
            dense_before.data()[i] / norm_before,
            dense_after.data()[i] / norm_after,
            epsilon = 1e-10
        );
    }
}

#[test]
fn test_two_site_canonicalize_center_1() {
    let storages = vec![
        Dense::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 2, 2],
            MemoryOrder::ColumnMajor,
        ),
        Dense::new(
            vec![1.0, 0.5, 0.3, 0.1],
            vec![2, 2, 1],
            MemoryOrder::ColumnMajor,
        ),
    ];
    let mut mps = Mps::from_storages(storages);
    let dense_before = mps_to_dense(&mps);
    let norm_before: f64 = dense_before
        .data()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();

    mps::canonicalize(&mut mps, 1);

    // Site 0 should be left-canonical
    assert!(is_left_canonical(mps.storage(0), 1e-10));

    // State preserved
    let dense_after = mps_to_dense(&mps);
    let norm_after: f64 = dense_after.data().iter().map(|x| x * x).sum::<f64>().sqrt();
    for i in 0..dense_before.len() {
        assert_abs_diff_eq!(
            dense_before.data()[i] / norm_before,
            dense_after.data()[i] / norm_after,
            epsilon = 1e-10
        );
    }
}
