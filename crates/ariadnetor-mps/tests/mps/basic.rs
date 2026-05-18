//! MPS/MPO construction, accessors, canonical form, and edge case tests.

use arnet_mps::{CanonicalForm, Mpo, Mps, TensorChain};
use arnet_tensor::{DenseLayout, DenseStorage, DenseTensorData};

/// Build a simple 3-site MPS with shapes (1,2,4), (4,2,4), (4,2,1).
fn make_3site_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    let sites = vec![
        DenseTensorData::ones(vec![1, 2, 4]), // site 0
        DenseTensorData::ones(vec![4, 2, 4]), // site 1
        DenseTensorData::ones(vec![4, 2, 1]), // site 2
    ];
    Mps::from_sites(sites)
}

#[test]
fn test_mps_from_sites() {
    let mps = make_3site_mps();

    assert_eq!(mps.len(), 3);
    assert!(!mps.is_empty());
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);
}

#[test]
fn test_mps_site_access() {
    let mps = make_3site_mps();

    assert_eq!(mps.site(0).shape(), &[1, 2, 4]);
    assert_eq!(mps.site(1).shape(), &[4, 2, 4]);
    assert_eq!(mps.site(2).shape(), &[4, 2, 1]);
    assert_eq!(mps.sites().len(), 3);
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
    let sites = vec![
        DenseTensorData::<f64>::ones(vec![1, 2, 3]),
        DenseTensorData::ones(vec![3, 2, 5]),
        DenseTensorData::ones(vec![5, 2, 2]),
        DenseTensorData::ones(vec![2, 2, 1]),
    ];
    let mps = Mps::from_sites(sites);

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

    mps.set_canonical_form(CanonicalForm::Mixed { center: 1 });
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 1 });

    mps.set_canonical_form(CanonicalForm::Partial {
        left_end: 2,
        right_start: 4,
    });
    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Partial {
            left_end: 2,
            right_start: 4
        }
    );
}

#[test]
fn test_site_mut_resets_canonical_form() {
    let mut mps = make_3site_mps();

    mps.set_canonical_form(CanonicalForm::Mixed { center: 1 });
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 1 });

    // Accessing site_mut should reset to Unknown
    let _ = mps.site_mut(0);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);
}

// ============================================================================
// MPO construction and accessors
// ============================================================================

#[test]
fn test_mpo_from_sites() {
    let sites = vec![
        DenseTensorData::<f64>::ones(vec![1, 2, 2, 3]), // site 0: (1, d_ket, d_bra, 3)
        DenseTensorData::ones(vec![3, 2, 2, 3]),        // site 1
        DenseTensorData::ones(vec![3, 2, 2, 1]),        // site 2
    ];
    let mpo = Mpo::from_sites(sites);

    assert_eq!(mpo.len(), 3);
    assert_eq!(mpo.site(0).shape(), &[1, 2, 2, 3]);
    assert_eq!(mpo.bond_dims(), vec![3, 3]);
    assert_eq!(mpo.max_bond_dim(), 3);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_single_site_mps() {
    let sites = vec![DenseTensorData::<f64>::ones(vec![1, 2, 1])];
    let mps = Mps::from_sites(sites);

    assert_eq!(mps.len(), 1);
    assert!(mps.bond_dims().is_empty());
    assert_eq!(mps.max_bond_dim(), 0);
}

#[test]
fn test_empty_mps() {
    let mps = Mps::<DenseStorage<f64>, DenseLayout>::from_sites(vec![]);

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
