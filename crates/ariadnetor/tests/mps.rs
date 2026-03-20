//! Tests for MPS/MPO data structures and TensorChain trait

use arnet::mps::{self, CanonicalForm, Mpo, Mps, TensorChain};
use arnet_tensor::{DenseTensor, TensorStorage};

// ============================================================================
// MPS construction and accessors
// ============================================================================

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

// ============================================================================
// Orthogonalize tests
// ============================================================================

/// Build a random-ish 4-site MPS from deterministic data.
fn make_4site_mps() -> Mps<f64> {
    let storages = vec![
        TensorStorage::from_data(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![1, 2, 4]),
        TensorStorage::from_data((1..=32).map(|i| i as f64 * 0.1).collect(), vec![4, 2, 4]),
        TensorStorage::from_data((1..=24).map(|i| i as f64 * 0.1).collect(), vec![4, 2, 3]),
        TensorStorage::from_data((1..=6).map(|i| i as f64 * 0.1).collect(), vec![3, 2, 1]),
    ];
    Mps::from_storages(storages)
}

/// Check that site j is left-canonical: Q^H Q ≈ I (columns are orthonormal).
/// Reshape to (m, k) where m = product(shape[..rank-1]), k = shape[rank-1].
fn is_left_canonical(storage: &TensorStorage<f64>, tol: f64) -> bool {
    let dense = match storage {
        TensorStorage::Dense(d) => d,
    };
    let shape = dense.shape();
    let rank = shape.len();
    let k = shape[rank - 1];
    let m: usize = shape[..rank - 1].iter().product();
    let mat = dense.reshape(vec![m, k]);

    // Compute Q^T Q (should be k×k identity)
    let backend = arnet_native::NativeBackend::new();
    let qtq = arnet_linalg::contract(&backend, &mat, &mat, "ab,ac->bc").unwrap();

    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (qtq.get(&[i, j]) - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Check that site j is right-canonical: Q Q^H ≈ I (rows are orthonormal).
/// Reshape to (k, n) where k = shape[0], n = product(shape[1..]).
fn is_right_canonical(storage: &TensorStorage<f64>, tol: f64) -> bool {
    let dense = match storage {
        TensorStorage::Dense(d) => d,
    };
    let shape = dense.shape();
    let k = shape[0];
    let n: usize = shape[1..].iter().product();
    let mat = dense.reshape(vec![k, n]);

    // Compute Q Q^T (should be k×k identity)
    let backend = arnet_native::NativeBackend::new();
    let qqt = arnet_linalg::contract(&backend, &mat, &mat, "ab,cb->ac").unwrap();

    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (qqt.get(&[i, j]) - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Compute the full state vector from an MPS by contracting all sites.
fn mps_to_dense(mps: &Mps<f64>) -> DenseTensor<f64> {
    let backend = arnet_native::NativeBackend::new();
    let n = mps.len();

    let first = match mps.storage(0) {
        TensorStorage::Dense(d) => d.clone(),
    };
    let mut result = first;

    for j in 1..n {
        let site = match mps.storage(j) {
            TensorStorage::Dense(d) => d,
        };
        // Contract last index of result with first index of site
        let r_rank = result.rank();
        let r_last: usize = *result.shape().last().unwrap();
        let r_rest: usize = result.shape()[..r_rank - 1].iter().product();
        let result_2d = result.reshape(vec![r_rest, r_last]);

        let s_first = site.shape()[0];
        let s_rest: usize = site.shape()[1..].iter().product();
        let site_2d = site.reshape(vec![s_first, s_rest]);

        let contracted =
            arnet_linalg::contract(&backend, &result_2d, &site_2d, "ab,bc->ac").unwrap();

        let mut new_shape: Vec<usize> = result.shape()[..r_rank - 1].to_vec();
        new_shape.extend_from_slice(&site.shape()[1..]);
        result = contracted.reshape(new_shape);
    }

    result
}

#[test]
fn test_orthogonalize_center_0() {
    let mut mps = make_4site_mps();
    let dense_before = mps_to_dense(&mps);

    mps::orthogonalize(&mut mps, 0);

    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Canonicalized { center: 0 }
    );

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
fn test_orthogonalize_center_middle() {
    let mut mps = make_4site_mps();
    let dense_before = mps_to_dense(&mps);

    mps::orthogonalize(&mut mps, 2);

    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Canonicalized { center: 2 }
    );

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
fn test_orthogonalize_center_last() {
    let mut mps = make_4site_mps();

    mps::orthogonalize(&mut mps, 3);

    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Canonicalized { center: 3 }
    );

    let tol = 1e-10;
    for j in 0..3 {
        assert!(
            is_left_canonical(mps.storage(j), tol),
            "site {j} not left-canonical"
        );
    }
}

#[test]
fn test_orthogonalize_single_site() {
    let storages = vec![TensorStorage::from_data(vec![1.0, 2.0], vec![1, 2, 1])];
    let mut mps = Mps::from_storages(storages);

    mps::orthogonalize(&mut mps, 0);

    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Canonicalized { center: 0 }
    );
}

#[test]
fn test_orthogonalize_preserves_physical_dims() {
    let mut mps = make_4site_mps();

    let phys_dims: Vec<usize> = (0..4).map(|j| mps.storage(j).shape()[1]).collect();

    mps::orthogonalize(&mut mps, 1);

    for j in 0..4 {
        assert_eq!(
            mps.storage(j).shape()[1],
            phys_dims[j],
            "physical dim changed at site {j}"
        );
    }
}
