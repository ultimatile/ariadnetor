//! Tests based on design documentation examples
//!
//! Validates that usage examples from two_layer_tensor_architecture.md actually work.

use arnet_tensor::{DenseTensor, RawTensor};

#[test]
fn test_design_doc_example_dense_tensor() {
    // From two_layer_tensor_architecture.md Section 5
    // Low-level API usage (メタデータ不要な場合)
    let raw_a = RawTensor::Dense(DenseTensor::zeros(vec![10, 20]));
    let raw_b = RawTensor::Dense(DenseTensor::ones(vec![20, 30]));

    assert_eq!(raw_a.shape(), &[10, 20]);
    assert_eq!(raw_b.shape(), &[20, 30]);

    // Manual shape management is required at this level
    assert_eq!(raw_a.rank(), 2);
    assert_eq!(raw_b.rank(), 2);
}

#[test]
fn test_design_doc_arc_cow() {
    // From tensor_storage_design.md Section 2.1
    // Arc + Copy-on-Write example
    let mut tensor = DenseTensor::zeros(vec![10, 20]);
    let cloned = tensor.clone(); // O(1) - only increments reference count

    // Modification triggers CoW
    tensor.set(&[0, 0], 42.0);

    assert_eq!(tensor.get(&[0, 0]), 42.0);
    assert_eq!(cloned.get(&[0, 0]), 0.0); // Original unchanged
}

#[test]
fn test_design_doc_dense_storage() {
    // From two_layer_tensor_architecture.md Section 2.1
    // DenseTensor basic usage
    let tensor = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    );

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.get(&[0, 0]), 1.0);
    assert_eq!(tensor.get(&[0, 1]), 2.0);
    assert_eq!(tensor.get(&[0, 2]), 3.0);
    assert_eq!(tensor.get(&[1, 0]), 4.0);
    assert_eq!(tensor.get(&[1, 1]), 5.0);
    assert_eq!(tensor.get(&[1, 2]), 6.0);
}

#[test]
fn test_design_doc_row_major_layout() {
    // From dense.rs documentation
    // Row-major layout verification
    // [[a, b, c],
    //  [d, e, f]]
    // → [a, b, c, d, e, f]
    let tensor = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    );

    // Verify row-major ordering
    let data = tensor.data();
    assert_eq!(data[0], 1.0); // [0,0]
    assert_eq!(data[1], 2.0); // [0,1]
    assert_eq!(data[2], 3.0); // [0,2]
    assert_eq!(data[3], 4.0); // [1,0]
    assert_eq!(data[4], 5.0); // [1,1]
    assert_eq!(data[5], 6.0); // [1,2]
}

#[test]
fn test_design_doc_constructors() {
    // Various constructor methods from design doc
    let zeros = DenseTensor::zeros(vec![3, 4]);
    assert_eq!(zeros.len(), 12);
    for &val in zeros.data() {
        assert_eq!(val, 0.0);
    }

    let ones = DenseTensor::ones(vec![2, 3]);
    assert_eq!(ones.len(), 6);
    for &val in ones.data() {
        assert_eq!(val, 1.0);
    }

    let constant = DenseTensor::constant(vec![2, 2], 3.14);
    assert_eq!(constant.len(), 4);
    for &val in constant.data() {
        assert_eq!(val, 3.14);
    }
}

#[test]
fn test_design_doc_phase_0_vec_usage() {
    // From tensor_storage_design.md Phase 0
    // Current implementation uses Vec everywhere
    let tensor = DenseTensor::zeros(vec![10, 20, 30, 40]);

    // Shape is stored as Vec (not SmallVec in Phase 0)
    assert_eq!(tensor.rank(), 4);
    assert_eq!(tensor.shape(), &[10, 20, 30, 40]);
    assert_eq!(tensor.len(), 10 * 20 * 30 * 40);
}
