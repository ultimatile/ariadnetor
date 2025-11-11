//! Future FatTensor integration tests
//!
//! These tests demonstrate the intended API and will fail until implementation is complete.
//! This is expected behavior (Red-First TDD).

use arnet_tensor::{FatTensor, LabelId, RawTensor};

#[test]
fn test_fat_tensor_creation() {
    let raw = RawTensor::<f64>::zeros(vec![10, 20]);
    let tensor = FatTensor::from_raw(raw, &["i", "j"]);

    assert_eq!(tensor.shape(), &[10, 20]);
    assert_eq!(tensor.rank(), 2);
}

#[test]
fn test_fat_tensor_contraction() {
    // TODO: Implement label matching for contraction
    let raw_a = RawTensor::<f64>::zeros(vec![10, 20]);
    let _tensor_a = FatTensor::from_raw(raw_a, &["i", "j"]);

    let raw_b = RawTensor::<f64>::ones(vec![20, 30]);
    let _tensor_b = FatTensor::from_raw(raw_b, &["j", "k"]);

    // TODO: Implement contraction that automatically matches "j" label
    // let result = tensor_a.contract(&tensor_b);
    // assert_eq!(result.shape(), &[10, 30]);
}

#[test]
fn test_fat_tensor_permutation() {
    // TODO: Implement permutation
    let raw = RawTensor::<f64>::zeros(vec![10, 20, 30]);
    let _tensor = FatTensor::from_raw(raw, &["i", "j", "k"]);

    // TODO: Implement automatic permutation based on label reordering
    // Should permute axes to match new label order
}

#[test]
fn test_fat_tensor_trace() {
    // TODO: Implement trace
    let raw = RawTensor::<f64>::zeros(vec![10, 10, 20]);
    let _tensor = FatTensor::from_raw(raw, &["i", "i'", "j"]);

    // TODO: Implement trace over repeated labels
    // Should contract over "i" and "i'" automatically
}

#[test]
fn test_label_prime_levels() {
    let idx1 = LabelId::intern("i"); // prime_level = 0
    let idx2 = idx1.prime(); // prime_level = 1
    let idx3 = idx2.prime(); // prime_level = 2

    assert_eq!(idx1.prime_level(), 0);
    assert_eq!(idx2.prime_level(), 1);
    assert_eq!(idx3.prime_level(), 2);
    assert_eq!(idx1.name(), "i");
    assert_eq!(idx2.name(), "i'");
    assert_eq!(idx3.name(), "i''");
}

#[test]
fn test_label_tags() {
    // TODO: Implement label tags (if needed in future)
    // Tags functionality may not be needed with the LabelId approach
    // as labels are simple identifiers without additional metadata
    // If tags are required, they should be stored in a separate structure
}
