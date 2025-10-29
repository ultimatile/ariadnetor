//! Future FatTensor integration tests
//!
//! These tests demonstrate the intended API and will fail until implementation is complete.
//! This is expected behavior (Red-First TDD).

use arnet_tensor::{RawTensor, FatTensor, Index, IndexSet};

#[test]
fn test_fat_tensor_creation() {
    // TODO: Implement Index::new()
    let raw = RawTensor::<f64>::zeros(vec![10, 20]);
    let indices = IndexSet {
        indices: vec![Index::new("i"), Index::new("j")],
        rowrank: 1,
    };
    let tensor = FatTensor::new(raw, indices);

    assert_eq!(tensor.shape(), &[10, 20]);
    assert_eq!(tensor.rank(), 2);
}

#[test]
fn test_fat_tensor_contraction() {
    // TODO: Implement Index matching for contraction
    let raw_a = RawTensor::<f64>::zeros(vec![10, 20]);
    let indices_a = IndexSet {
        indices: vec![Index::new("i"), Index::new("j")],
        rowrank: 1,
    };
    let _tensor_a = FatTensor::new(raw_a, indices_a);

    let raw_b = RawTensor::<f64>::ones(vec![20, 30]);
    let indices_b = IndexSet {
        indices: vec![Index::new("j"), Index::new("k")],
        rowrank: 1,
    };
    let _tensor_b = FatTensor::new(raw_b, indices_b);

    // TODO: Implement contraction that automatically matches "j" index
    // let result = tensor_a.contract(&tensor_b);
    // assert_eq!(result.shape(), &[10, 30]);
}

#[test]
fn test_fat_tensor_permutation() {
    // TODO: Implement permutation
    let raw = RawTensor::<f64>::zeros(vec![10, 20, 30]);
    let indices = IndexSet {
        indices: vec![Index::new("i"), Index::new("j"), Index::new("k")],
        rowrank: 2,
    };
    let _tensor = FatTensor::new(raw, indices);

    // TODO: Implement automatic permutation based on index reordering
    // Should permute axes to match new index order
}

#[test]
fn test_fat_tensor_trace() {
    // TODO: Implement trace
    let raw = RawTensor::<f64>::zeros(vec![10, 10, 20]);
    let indices = IndexSet {
        indices: vec![Index::new("i"), Index::new("i'"), Index::new("j")],
        rowrank: 2,
    };
    let _tensor = FatTensor::new(raw, indices);

    // TODO: Implement trace over repeated indices
    // Should contract over "i" and "i'" automatically
}

#[test]
fn test_index_prime_levels() {
    // TODO: Implement Index prime level
    let _idx1 = Index::new("i");  // prime_level = 0
    // let idx2 = idx1.prime();       // prime_level = 1
    // let idx3 = idx2.prime();       // prime_level = 2

    // assert_eq!(idx1.prime_level, 0);
    // assert_eq!(idx2.prime_level, 1);
    // assert_eq!(idx3.prime_level, 2);
}

#[test]
fn test_index_tags() {
    // TODO: Implement Index tags
    // let mut idx = Index::new("i");
    // idx.add_tag("Site");
    // idx.add_tag("Up");
    // assert!(idx.has_tag("Site"));
    // assert!(idx.has_tag("Up"));
    // assert!(!idx.has_tag("Down"));
}
