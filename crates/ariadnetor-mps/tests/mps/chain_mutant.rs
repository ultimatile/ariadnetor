//! Targeted mutation-testing coverage for chain.rs accessors.
//!
//! Focuses on exact value assertions for bond_dim, bond_dims,
//! max_bond_dim, and is_empty to catch arithmetic/comparison mutants.

use arnet_mps::{Mps, TensorChain};
use arnet_tensor::{DenseLayout, DenseStorage, DenseTensorData};

/// Two-site MPS with asymmetric bond dimensions to distinguish index choices.
fn make_2site_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    // site 0: (1, 2, 3), site 1: (3, 2, 1)
    let storages = vec![
        DenseTensorData::ones(vec![1, 2, 3]),
        DenseTensorData::ones(vec![3, 2, 1]),
    ];
    Mps::from_sites(storages)
}

// --------------------------------------------------------------------------
// is_empty
// --------------------------------------------------------------------------

#[test]
fn test_is_empty_true_for_zero_sites() {
    let mps = Mps::<DenseStorage<f64>, DenseLayout>::from_sites(vec![]);
    assert!(mps.is_empty());
}

#[test]
fn test_is_empty_false_for_one_site() {
    let mps = Mps::from_sites(vec![DenseTensorData::<f64>::ones(vec![1, 2, 1])]);
    assert!(!mps.is_empty());
}

#[test]
fn test_is_empty_false_for_two_sites() {
    let mps = make_2site_mps();
    assert!(!mps.is_empty());
}

// --------------------------------------------------------------------------
// bond_dim: exact values on asymmetric shapes
// --------------------------------------------------------------------------

#[test]
fn test_bond_dim_returns_last_mode_of_site() {
    let mps = make_2site_mps();
    // bond 0: last dim of site 0 = 3
    assert_eq!(mps.bond_dim(0), 3);
}

#[test]
fn test_bond_dim_asymmetric_three_sites() {
    // site shapes: (1,2,5), (5,3,7), (7,2,1)
    let storages = vec![
        DenseTensorData::<f64>::ones(vec![1, 2, 5]),
        DenseTensorData::ones(vec![5, 3, 7]),
        DenseTensorData::ones(vec![7, 2, 1]),
    ];
    let mps = Mps::from_sites(storages);
    assert_eq!(mps.bond_dim(0), 5);
    assert_eq!(mps.bond_dim(1), 7);
}

// --------------------------------------------------------------------------
// bond_dims: length and content
// --------------------------------------------------------------------------

#[test]
fn test_bond_dims_single_site_empty() {
    let mps = Mps::from_sites(vec![DenseTensorData::<f64>::ones(vec![1, 2, 1])]);
    let dims = mps.bond_dims();
    assert!(dims.is_empty());
    assert_eq!(dims.len(), 0);
}

#[test]
fn test_bond_dims_two_sites_one_bond() {
    let mps = make_2site_mps();
    let dims = mps.bond_dims();
    assert_eq!(dims.len(), 1);
    assert_eq!(dims, vec![3]);
}

#[test]
fn test_bond_dims_four_distinct_bonds() {
    // Ensure each bond has a unique dimension to catch off-by-one / wrong-index mutants.
    let storages = vec![
        DenseTensorData::<f64>::ones(vec![1, 2, 3]),
        DenseTensorData::ones(vec![3, 2, 5]),
        DenseTensorData::ones(vec![5, 2, 7]),
        DenseTensorData::ones(vec![7, 2, 11]),
        DenseTensorData::ones(vec![11, 2, 1]),
    ];
    let mps = Mps::from_sites(storages);
    assert_eq!(mps.bond_dims(), vec![3, 5, 7, 11]);
}

// --------------------------------------------------------------------------
// max_bond_dim
// --------------------------------------------------------------------------

#[test]
fn test_max_bond_dim_zero_sites() {
    let mps = Mps::<DenseStorage<f64>, DenseLayout>::from_sites(vec![]);
    assert_eq!(mps.max_bond_dim(), 0);
}

#[test]
fn test_max_bond_dim_single_site() {
    let mps = Mps::from_sites(vec![DenseTensorData::<f64>::ones(vec![1, 2, 1])]);
    assert_eq!(mps.max_bond_dim(), 0);
}

#[test]
fn test_max_bond_dim_finds_largest() {
    // bonds: 3, 5, 7, 11 → max = 11
    let storages = vec![
        DenseTensorData::<f64>::ones(vec![1, 2, 3]),
        DenseTensorData::ones(vec![3, 2, 5]),
        DenseTensorData::ones(vec![5, 2, 7]),
        DenseTensorData::ones(vec![7, 2, 11]),
        DenseTensorData::ones(vec![11, 2, 1]),
    ];
    let mps = Mps::from_sites(storages);
    assert_eq!(mps.max_bond_dim(), 11);
}

#[test]
fn test_max_bond_dim_uniform() {
    // All bonds equal 4 → max = 4
    let storages = vec![
        DenseTensorData::<f64>::ones(vec![1, 2, 4]),
        DenseTensorData::ones(vec![4, 2, 4]),
        DenseTensorData::ones(vec![4, 2, 1]),
    ];
    let mps = Mps::from_sites(storages);
    assert_eq!(mps.max_bond_dim(), 4);
}

// --------------------------------------------------------------------------
// len
// --------------------------------------------------------------------------

#[test]
fn test_len_matches_site_count() {
    assert_eq!(
        Mps::<DenseStorage<f64>, DenseLayout>::from_sites(vec![]).len(),
        0
    );
    assert_eq!(
        Mps::from_sites(vec![DenseTensorData::<f64>::ones(vec![1, 2, 1])]).len(),
        1
    );
    assert_eq!(make_2site_mps().len(), 2);
}
