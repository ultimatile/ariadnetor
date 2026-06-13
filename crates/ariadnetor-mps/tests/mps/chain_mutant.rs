//! Targeted mutation-testing coverage for chain.rs accessors.

use arnet_mps::{Mps, TensorChain};
use arnet_native::NativeBackend;
use arnet_tensor::{DenseLayout, DenseStorage, DenseTensor};

fn empty_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    Mps::<DenseStorage<f64>, DenseLayout, NativeBackend>::empty(NativeBackend::shared())
}

/// Two-site MPS with asymmetric bond dimensions to distinguish index choices.
fn make_2site_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    let sites = vec![
        DenseTensor::<f64>::ones(vec![1, 2, 3]),
        DenseTensor::<f64>::ones(vec![3, 2, 1]),
    ];
    Mps::from_sites(sites)
}

#[test]
fn test_is_empty_true_for_zero_sites() {
    let mps = empty_mps();
    assert!(mps.is_empty());
}

#[test]
fn test_is_empty_false_for_one_site() {
    let mps = Mps::from_sites(vec![DenseTensor::<f64>::ones(vec![1, 2, 1])]);
    assert!(!mps.is_empty());
}

#[test]
fn test_is_empty_false_for_two_sites() {
    let mps = make_2site_mps();
    assert!(!mps.is_empty());
}

#[test]
fn test_bond_dim_returns_last_mode_of_site() {
    let mps = make_2site_mps();
    assert_eq!(mps.bond_dim(0), 3);
}

#[test]
fn test_bond_dim_asymmetric_three_sites() {
    let sites = vec![
        DenseTensor::<f64>::ones(vec![1, 2, 5]),
        DenseTensor::<f64>::ones(vec![5, 3, 7]),
        DenseTensor::<f64>::ones(vec![7, 2, 1]),
    ];
    let mps = Mps::from_sites(sites);
    assert_eq!(mps.bond_dim(0), 5);
    assert_eq!(mps.bond_dim(1), 7);
}

#[test]
fn test_bond_dims_single_site_empty() {
    let mps = Mps::from_sites(vec![DenseTensor::<f64>::ones(vec![1, 2, 1])]);
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
    let sites = vec![
        DenseTensor::<f64>::ones(vec![1, 2, 3]),
        DenseTensor::<f64>::ones(vec![3, 2, 5]),
        DenseTensor::<f64>::ones(vec![5, 2, 7]),
        DenseTensor::<f64>::ones(vec![7, 2, 11]),
        DenseTensor::<f64>::ones(vec![11, 2, 1]),
    ];
    let mps = Mps::from_sites(sites);
    assert_eq!(mps.bond_dims(), vec![3, 5, 7, 11]);
}

#[test]
fn test_max_bond_dim_zero_sites() {
    let mps = empty_mps();
    assert_eq!(mps.max_bond_dim(), 0);
}

#[test]
fn test_max_bond_dim_single_site() {
    let mps = Mps::from_sites(vec![DenseTensor::<f64>::ones(vec![1, 2, 1])]);
    assert_eq!(mps.max_bond_dim(), 0);
}

#[test]
fn test_max_bond_dim_finds_largest() {
    let sites = vec![
        DenseTensor::<f64>::ones(vec![1, 2, 3]),
        DenseTensor::<f64>::ones(vec![3, 2, 5]),
        DenseTensor::<f64>::ones(vec![5, 2, 7]),
        DenseTensor::<f64>::ones(vec![7, 2, 11]),
        DenseTensor::<f64>::ones(vec![11, 2, 1]),
    ];
    let mps = Mps::from_sites(sites);
    assert_eq!(mps.max_bond_dim(), 11);
}

#[test]
fn test_max_bond_dim_uniform() {
    let sites = vec![
        DenseTensor::<f64>::ones(vec![1, 2, 4]),
        DenseTensor::<f64>::ones(vec![4, 2, 4]),
        DenseTensor::<f64>::ones(vec![4, 2, 1]),
    ];
    let mps = Mps::from_sites(sites);
    assert_eq!(mps.max_bond_dim(), 4);
}

#[test]
fn test_len_matches_site_count() {
    assert_eq!(empty_mps().len(), 0);
    assert_eq!(
        Mps::from_sites(vec![DenseTensor::<f64>::ones(vec![1, 2, 1])]).len(),
        1,
    );
    assert_eq!(make_2site_mps().len(), 2);
}
