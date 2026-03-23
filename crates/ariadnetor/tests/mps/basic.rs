//! MPS/MPO construction, accessors, canonical form, and edge case tests.

use arnet::mps::{CanonicalForm, Mpo, Mps, TensorChain};
use arnet_tensor::TensorStorage;

/// Build a simple 3-site MPS with shapes (1,2,4), (4,2,4), (4,2,1).
fn make_3site_mps() -> Mps<f64> {
    let storages = vec![
        TensorStorage::ones(vec![1, 2, 4]), // site 0
        TensorStorage::ones(vec![4, 2, 4]), // site 1
        TensorStorage::ones(vec![4, 2, 1]), // site 2
    ];
    Mps::from_storages(storages)
}

#[test]
fn test_mps_from_storages() {
    let mps = make_3site_mps();

    assert_eq!(mps.len(), 3);
    assert!(!mps.is_empty());
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);
}

#[test]
fn test_mps_storage_access() {
    let mps = make_3site_mps();

    assert_eq!(mps.storage(0).shape(), &[1, 2, 4]);
    assert_eq!(mps.storage(1).shape(), &[4, 2, 4]);
    assert_eq!(mps.storage(2).shape(), &[4, 2, 1]);
    assert_eq!(mps.storages().len(), 3);
}

#[test]
fn test_mps_bond_dim() {
    let mps = make_3site_mps();

    // bond 0: between site 0 and 1, χ_R of site 0 = 4
    assert_eq!(mps.bond_dim(0), 4);
    // bond 1: between site 1 and 2, χ_R of site 1 = 4
    assert_eq!(mps.bond_dim(1), 4);
}

#[test]
fn test_mps_bond_dims() {
    let mps = make_3site_mps();

    assert_eq!(mps.bond_dims(), vec![4, 4]);
    assert_eq!(mps.max_bond_dim(), 4);
}

#[test]
fn test_mps_varying_bond_dims() {
    let storages = vec![
        TensorStorage::<f64>::ones(vec![1, 2, 3]),
        TensorStorage::ones(vec![3, 2, 5]),
        TensorStorage::ones(vec![5, 2, 2]),
        TensorStorage::ones(vec![2, 2, 1]),
    ];
    let mps = Mps::from_storages(storages);

    assert_eq!(mps.bond_dims(), vec![3, 5, 2]);
    assert_eq!(mps.max_bond_dim(), 5);
}

// ============================================================================
// Canonical form tracking
// ============================================================================

#[test]
fn test_canonical_form_initial_unknown() {
    let mps = make_3site_mps();
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);
}

#[test]
fn test_canonical_form_set_and_get() {
    let mut mps = make_3site_mps();

    mps.set_canonical_form(CanonicalForm::Canonicalized { center: 1 });
    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Canonicalized { center: 1 }
    );

    mps.set_canonical_form(CanonicalForm::PartiallyCanonicalized { llim: 2, rlim: 4 });
    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::PartiallyCanonicalized { llim: 2, rlim: 4 }
    );
}

#[test]
fn test_storage_mut_resets_canonical_form() {
    let mut mps = make_3site_mps();

    mps.set_canonical_form(CanonicalForm::Canonicalized { center: 1 });
    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Canonicalized { center: 1 }
    );

    // Accessing storage_mut should reset to Unknown
    let _ = mps.storage_mut(0);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);
}

// ============================================================================
// MPO construction and accessors
// ============================================================================

#[test]
fn test_mpo_from_storages() {
    let storages = vec![
        TensorStorage::<f64>::ones(vec![1, 2, 2, 3]), // site 0: (1, d_ket, d_bra, 3)
        TensorStorage::ones(vec![3, 2, 2, 3]),        // site 1
        TensorStorage::ones(vec![3, 2, 2, 1]),        // site 2
    ];
    let mpo = Mpo::from_storages(storages);

    assert_eq!(mpo.len(), 3);
    assert_eq!(mpo.storage(0).shape(), &[1, 2, 2, 3]);
    assert_eq!(mpo.bond_dims(), vec![3, 3]);
    assert_eq!(mpo.max_bond_dim(), 3);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_single_site_mps() {
    let storages = vec![TensorStorage::<f64>::ones(vec![1, 2, 1])];
    let mps = Mps::from_storages(storages);

    assert_eq!(mps.len(), 1);
    assert!(mps.bond_dims().is_empty());
    assert_eq!(mps.max_bond_dim(), 0);
}

#[test]
fn test_empty_mps() {
    let mps = Mps::<f64>::from_storages(vec![]);

    assert_eq!(mps.len(), 0);
    assert!(mps.is_empty());
    assert!(mps.bond_dims().is_empty());
    assert_eq!(mps.max_bond_dim(), 0);
}

#[test]
fn test_mps_clone() {
    let mps = make_3site_mps();
    let mps2 = mps.clone();

    assert_eq!(mps.len(), mps2.len());
    assert_eq!(mps.bond_dims(), mps2.bond_dims());
}
