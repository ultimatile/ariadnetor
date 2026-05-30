//! MPS/MPO construction, accessors, canonical form, and edge case tests.

use arnet::{DenseLayout, DenseStorage, DenseTensor, MemoryOrder, NativeBackend};
use arnet_mps::{CanonicalForm, Mpo, Mps, TensorChain};
use std::sync::Arc;

/// Build a simple 3-site MPS with shapes (1,2,4), (4,2,4), (4,2,1).
fn make_3site_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    let sites = vec![
        DenseTensor::<f64>::ones(vec![1, 2, 4]), // site 0
        DenseTensor::<f64>::ones(vec![4, 2, 4]), // site 1
        DenseTensor::<f64>::ones(vec![4, 2, 1]), // site 2
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

    assert_eq!(mps.bond_dim(0), 4);
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
        DenseTensor::<f64>::ones(vec![1, 2, 3]),
        DenseTensor::<f64>::ones(vec![3, 2, 5]),
        DenseTensor::<f64>::ones(vec![5, 2, 2]),
        DenseTensor::<f64>::ones(vec![2, 2, 1]),
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

    // Accessing site_mut should reset to Unknown.
    let _ = mps.site_mut(0);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);
}

// ============================================================================
// MPO construction and accessors
// ============================================================================

#[test]
fn test_mpo_from_sites() {
    let sites = vec![
        DenseTensor::<f64>::ones(vec![1, 2, 2, 3]),
        DenseTensor::<f64>::ones(vec![3, 2, 2, 3]),
        DenseTensor::<f64>::ones(vec![3, 2, 2, 1]),
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
    let sites = vec![DenseTensor::<f64>::ones(vec![1, 2, 1])];
    let mps = Mps::from_sites(sites);

    assert_eq!(mps.len(), 1);
    assert!(mps.bond_dims().is_empty());
    assert_eq!(mps.max_bond_dim(), 0);
}

#[test]
fn test_empty_mps() {
    let mps = Mps::<DenseStorage<f64>, DenseLayout, NativeBackend>::empty(NativeBackend::shared());

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
// Tier 1 rejection tests — every chain constructor must enforce the order
// invariant. These pin the rejection so a future "papers over instead of
// rejects" defensive align cannot silently regress the invariant.
// ============================================================================

#[test]
#[should_panic(expected = "from_sites")]
fn test_mps_from_sites_empty_rejected() {
    let _ = Mps::<DenseStorage<f64>, DenseLayout, NativeBackend>::from_sites(Vec::new());
}

#[test]
#[should_panic(expected = "from_sites")]
fn test_mpo_from_sites_empty_rejected() {
    let _ = Mpo::<DenseStorage<f64>, DenseLayout, NativeBackend>::from_sites(Vec::new());
}

/// Build a single rank-3 site whose layout order disagrees with
/// NativeBackend's preferred order (NativeBackend is ColumnMajor; this
/// site is RowMajor).
fn rm_site() -> DenseTensor<f64> {
    DenseTensor::from_raw_parts(vec![1.0; 4], vec![1, 2, 2], NativeBackend::shared())
        .reordered(MemoryOrder::RowMajor)
}

/// Same as `rm_site` but rank-4 (for MPO).
fn rm_mpo_site() -> DenseTensor<f64> {
    DenseTensor::from_raw_parts(vec![1.0; 8], vec![1, 2, 2, 2], NativeBackend::shared())
        .reordered(MemoryOrder::RowMajor)
}

#[test]
#[should_panic(expected = "order")]
fn test_mps_from_sites_rejects_mismatched_order() {
    let _ = Mps::from_sites(vec![rm_site()]);
}

#[test]
#[should_panic(expected = "order")]
fn test_mpo_from_sites_rejects_mismatched_order() {
    let _ = Mpo::from_sites(vec![rm_mpo_site()]);
}

#[test]
#[should_panic(expected = "order")]
fn test_mps_with_backend_rejects_mismatched_order() {
    let _ = Mps::with_backend(vec![rm_site()], NativeBackend::shared());
}

#[test]
#[should_panic(expected = "order")]
fn test_mpo_with_backend_rejects_mismatched_order() {
    let _ = Mpo::with_backend(vec![rm_mpo_site()], NativeBackend::shared());
}

#[test]
fn test_mps_with_backend_accepts_distinct_arc_same_preferred_order() {
    // The plan deliberately uses a per-site `order == backend.preferred_order()`
    // check rather than `Arc::ptr_eq`, so distinct backend instances with
    // matching preferred order must be accepted.
    let site_backend: Arc<NativeBackend> = Arc::new(NativeBackend::new());
    let chain_backend: Arc<NativeBackend> = Arc::new(NativeBackend::new());
    assert!(!Arc::ptr_eq(&site_backend, &chain_backend));

    let site = DenseTensor::from_raw_parts(vec![1.0; 4], vec![1, 2, 2], site_backend);

    let mps = Mps::with_backend(vec![site], chain_backend);
    assert_eq!(mps.len(), 1);
}
