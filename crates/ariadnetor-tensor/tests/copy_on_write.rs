//! Copy-on-Write behavior tests
//!
//! Tests that Arc-based shared ownership and CoW work correctly.

use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_clone_is_cheap() {
    let tensor1 =
        Dense::from_data_with_order(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let tensor2 = tensor1.clone();

    // Both should have the same values
    assert_eq!(tensor1.get(&[0, 0]), 1.0);
    assert_eq!(tensor2.get(&[0, 0]), 1.0);
    assert_eq!(tensor1.get(&[1, 1]), 4.0);
    assert_eq!(tensor2.get(&[1, 1]), 4.0);
}

#[test]
fn test_copy_on_write_tensor_storage() {
    let tensor1 =
        Dense::from_data_with_order(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let mut tensor2 = tensor1.clone(); // Share data (O(1) clone)

    // Modify tensor2 - should trigger CoW
    tensor2.set(&[0, 0], 999.0);

    // tensor1 should be unchanged
    assert_eq!(tensor1.get(&[0, 0]), 1.0);
    assert_eq!(tensor2.get(&[0, 0]), 999.0);

    // Other values should remain the same
    assert_eq!(tensor1.get(&[1, 1]), 4.0);
    assert_eq!(tensor2.get(&[1, 1]), 4.0);
}

#[test]
fn test_copy_on_write_dense_tensor() {
    let tensor1 = Dense::from_data_with_order(
        vec![10.0, 20.0, 30.0, 40.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let mut tensor2 = tensor1.clone();

    // Modify tensor2
    tensor2.set(&[1, 0], 777.0);

    // tensor1 unchanged
    assert_eq!(tensor1.get(&[1, 0]), 30.0);
    assert_eq!(tensor2.get(&[1, 0]), 777.0);
}

#[test]
fn test_fill_triggers_cow() {
    let tensor1 = Dense::<f64>::ones(vec![5, 5]);
    let mut tensor2 = tensor1.clone();

    // Fill should trigger CoW
    tensor2.fill(0.0);

    // tensor1 should still have ones
    assert_eq!(tensor1.get(&[0, 0]), 1.0);
    assert_eq!(tensor1.get(&[4, 4]), 1.0);

    // tensor2 should have zeros
    assert_eq!(tensor2.get(&[0, 0]), 0.0);
    assert_eq!(tensor2.get(&[4, 4]), 0.0);
}

#[test]
fn test_data_mut_triggers_cow() {
    let tensor1 =
        Dense::from_data_with_order(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let mut tensor2 = tensor1.clone();

    // Get mutable reference - should trigger CoW
    {
        let data = tensor2.data_mut();
        data[0] = 100.0;
    }

    // tensor1 unchanged
    assert_eq!(tensor1.data()[0], 1.0);
    assert_eq!(tensor2.data()[0], 100.0);
}

#[test]
fn test_multiple_clones() {
    let original =
        Dense::from_data_with_order(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let mut clone1 = original.clone();
    let mut clone2 = original.clone();
    let mut clone3 = original.clone();

    // Each modification should be independent
    clone1.set(&[0, 0], 10.0);
    clone2.set(&[0, 0], 20.0);
    clone3.set(&[0, 0], 30.0);

    assert_eq!(original.get(&[0, 0]), 1.0);
    assert_eq!(clone1.get(&[0, 0]), 10.0);
    assert_eq!(clone2.get(&[0, 0]), 20.0);
    assert_eq!(clone3.get(&[0, 0]), 30.0);
}
