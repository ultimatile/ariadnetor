//! Inner product, norm, and expectation value tests.

use approx::assert_abs_diff_eq;
use arnet_mps::{self as mps, CanonicalForm, Mpo, Mps, TensorChain};
use arnet_tensor::{Dense, MemoryOrder};

use super::helpers::make_4site_mps;

#[test]
fn test_inner_self_equals_norm_squared() {
    let mps = make_4site_mps();

    let overlap = mps::inner(&mps, &mps);
    let n = mps::norm(&mps);

    assert_abs_diff_eq!(overlap, n * n, epsilon = 1e-10);
}

#[test]
fn test_inner_product_state() {
    // |0000⟩: each site has tensor [1, 0] reshaped to (1, 2, 1)
    let storages_0 = vec![
        Dense::new(vec![1.0, 0.0], vec![1, 2, 1]),
        Dense::new(vec![1.0, 0.0], vec![1, 2, 1]),
    ];
    let psi = Mps::from_storages(storages_0);

    // |00⟩ with itself → 1.0
    let overlap = mps::inner(&psi, &psi);
    assert_abs_diff_eq!(overlap, 1.0, epsilon = 1e-12);

    // |11⟩
    let storages_1 = vec![
        Dense::new(vec![0.0, 1.0], vec![1, 2, 1]),
        Dense::new(vec![0.0, 1.0], vec![1, 2, 1]),
    ];
    let phi = Mps::from_storages(storages_1);

    // ⟨00|11⟩ = 0
    let overlap = mps::inner(&psi, &phi);
    assert_abs_diff_eq!(overlap, 0.0, epsilon = 1e-12);
}

#[test]
fn test_norm_canonicalized_is_fast() {
    let mut mps = make_4site_mps();

    // Compute norm before canonicalization (full contraction)
    let norm_full = mps::norm(&mps);

    // Canonicalize and compute norm (O(1) from center tensor)
    mps::canonicalize(&mut mps, 2);
    let norm_canonical = mps::norm(&mps);

    assert_abs_diff_eq!(norm_full, norm_canonical, epsilon = 1e-10);
}

#[test]
fn test_norm_product_state() {
    let storages = vec![
        Dense::new(vec![1.0, 0.0], vec![1, 2, 1]),
        Dense::new(vec![1.0, 0.0], vec![1, 2, 1]),
        Dense::new(vec![1.0, 0.0], vec![1, 2, 1]),
    ];
    let psi = Mps::from_storages(storages);

    assert_abs_diff_eq!(mps::norm(&psi), 1.0, epsilon = 1e-12);
}

#[test]
fn test_norm_left_canonical_returns_one() {
    let mut mps = make_4site_mps();
    let norm_full = mps::norm(&mps);

    // Canonicalize to make all sites left-isometric, then mark as Left
    mps::canonicalize(&mut mps, 3);
    mps.set_canonical_form(CanonicalForm::Left);

    let norm_left = mps::norm(&mps);
    // Left canonical means normalized → norm should be 1.0
    assert_abs_diff_eq!(norm_left, 1.0, epsilon = 1e-12);
    // This should differ from the full norm (which is not 1.0 for make_4site_mps)
    assert!(
        (norm_full - 1.0).abs() > 0.01,
        "test setup: full norm should not be 1.0"
    );
}

#[test]
fn test_norm_right_canonical_returns_one() {
    let mut mps = make_4site_mps();

    mps::canonicalize(&mut mps, 0);
    mps.set_canonical_form(CanonicalForm::Right);

    let norm_right = mps::norm(&mps);
    assert_abs_diff_eq!(norm_right, 1.0, epsilon = 1e-12);
}

#[test]
fn test_norm_mixed_uses_center_tensor() {
    let mut mps = make_4site_mps();
    let norm_full = mps::norm(&mps);

    mps::canonicalize(&mut mps, 2);
    // canonical_form is Mixed { center: 2 } after canonicalize
    let norm_mixed = mps::norm(&mps);

    // Both should agree
    assert_abs_diff_eq!(norm_full, norm_mixed, epsilon = 1e-10);
    // And the result should equal the Frobenius norm of the center tensor
    let center_norm = mps.storage(2).norm();
    assert_abs_diff_eq!(norm_mixed, center_norm, epsilon = 1e-12);
}

#[test]
fn test_inner_preserved_by_canonicalize() {
    let mps_a = make_4site_mps();
    let mut mps_b = make_4site_mps();

    let overlap_before = mps::inner(&mps_a, &mps_b);

    mps::canonicalize(&mut mps_b, 1);

    let overlap_after = mps::inner(&mps_a, &mps_b);

    assert_abs_diff_eq!(overlap_before, overlap_after, epsilon = 1e-10);
}

#[test]
fn test_expect_identity_mpo() {
    // Identity MPO: each site is a 1×2×2×1 tensor = identity matrix reshaped
    let id_storages = vec![
        Dense::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2, 1]),
        Dense::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2, 1]),
        Dense::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2, 1]),
    ];
    let identity = Mpo::from_storages(id_storages);

    let storages = vec![
        Dense::new(vec![1.0, 0.0], vec![1, 2, 1]),
        Dense::new(vec![1.0, 0.0], vec![1, 2, 1]),
        Dense::new(vec![1.0, 0.0], vec![1, 2, 1]),
    ];
    let psi = Mps::from_storages(storages);

    // ⟨ψ|I|ψ⟩ = ⟨ψ|ψ⟩ = 1.0
    let result = mps::braket(&psi, &identity, &psi);
    assert_abs_diff_eq!(result, 1.0, epsilon = 1e-12);
}

#[test]
fn test_expect_sz_product_state() {
    // Sz operator as MPO on single site: diag(0.5, -0.5)
    // MPO shape: (1, d_ket=2, d_bra=2, 1)
    // Sz[0,0,0,0]=0.5, Sz[0,1,1,0]=-0.5 (diagonal elements)
    let sz_data = vec![0.5, 0.0, 0.0, -0.5]; // row-major (1,2,2,1)
    let sz_mpo = Mpo::from_storages(vec![Dense::new(sz_data, vec![1, 2, 2, 1])]);

    // |0⟩ (spin up): ⟨0|Sz|0⟩ = 0.5
    let up = Mps::from_storages(vec![Dense::new(vec![1.0, 0.0], vec![1, 2, 1])]);
    assert_abs_diff_eq!(mps::braket(&up, &sz_mpo, &up), 0.5, epsilon = 1e-12);

    // |1⟩ (spin down): ⟨1|Sz|1⟩ = -0.5
    let dn = Mps::from_storages(vec![Dense::new(vec![0.0, 1.0], vec![1, 2, 1])]);
    assert_abs_diff_eq!(mps::braket(&dn, &sz_mpo, &dn), -0.5, epsilon = 1e-12);
}

#[test]
fn test_expect_identity_equals_inner() {
    let mps = make_4site_mps();

    let id_storages: Vec<_> = (0..4)
        .map(|_| Dense::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2, 1]))
        .collect();
    let identity = Mpo::from_storages(id_storages);

    let inner_val = mps::inner(&mps, &mps);
    let expect_val = mps::braket(&mps, &identity, &mps);

    assert_abs_diff_eq!(inner_val, expect_val, epsilon = 1e-10);
}
