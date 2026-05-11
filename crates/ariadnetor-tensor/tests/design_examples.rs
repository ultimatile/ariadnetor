//! Tests based on design documentation examples
//!
//! Validates that usage examples from two_layer_tensor_architecture.md actually work.

use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_design_doc_example_dense_tensor() {
    // Low-level API usage
    let raw_a = Dense::<f64>::zeros(vec![10, 20]);
    let raw_b = Dense::<f64>::ones(vec![20, 30]);

    assert_eq!(raw_a.shape(), &[10, 20]);
    assert_eq!(raw_b.shape(), &[20, 30]);

    // Manual shape management is required at this level
    assert_eq!(raw_a.rank(), 2);
    assert_eq!(raw_b.rank(), 2);
}

#[test]
fn test_design_doc_arc_cow() {
    // Arc + Copy-on-Write example
    let mut tensor = Dense::<f64>::zeros(vec![10, 20]);
    let cloned = tensor.clone(); // O(1) - only increments reference count

    // Modification triggers CoW
    tensor.set(&[0, 0], 42.0);

    assert_eq!(tensor.get(&[0, 0]), 42.0);
    assert_eq!(cloned.get(&[0, 0]), 0.0); // Original unchanged
}

#[test]
fn test_design_doc_dense_storage() {
    // Dense basic usage. Data laid out row-major
    // (`[[1, 2, 3], [4, 5, 6]]` → flat `[1, 2, 3, 4, 5, 6]`); the
    // `Dense::get` assertions below use row-major flat-index
    // semantics, so the tag must match.
    let tensor = Dense::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
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
    // Row-major layout verification
    // [[a, b, c],
    //  [d, e, f]]
    // → [a, b, c, d, e, f]
    let tensor = Dense::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
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
    let zeros = Dense::<f64>::zeros(vec![3, 4]);
    assert_eq!(zeros.len(), 12);
    for &val in zeros.data() {
        assert_eq!(val, 0.0);
    }

    let ones = Dense::<f64>::ones(vec![2, 3]);
    assert_eq!(ones.len(), 6);
    for &val in ones.data() {
        assert_eq!(val, 1.0);
    }

    let constant = Dense::constant(vec![2, 2], 3.15);
    assert_eq!(constant.len(), 4);
    for &val in constant.data() {
        assert_eq!(val, 3.15);
    }
}

#[test]
fn test_design_doc_phase_0_vec_usage() {
    // Current implementation uses Vec everywhere
    let tensor = Dense::<f64>::zeros(vec![10, 20, 30, 40]);

    // Shape is stored as Vec (not SmallVec in Phase 0)
    assert_eq!(tensor.rank(), 4);
    assert_eq!(tensor.shape(), &[10, 20, 30, 40]);
    assert_eq!(tensor.len(), 10 * 20 * 30 * 40);
}
