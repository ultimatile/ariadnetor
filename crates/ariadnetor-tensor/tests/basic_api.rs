//! Basic API integration tests
//!
//! Tests the public API usage patterns from design documentation.

use arnet_tensor::{DenseTensor, MemoryOrder, TensorStorage};

#[test]
fn test_tensor_storage_creation() {
    // Create tensors using TensorStorage constructors
    let zeros = TensorStorage::<f64>::zeros(vec![10, 20]);
    assert_eq!(zeros.shape(), &[10, 20]);
    assert_eq!(zeros.len(), 200);

    let ones = TensorStorage::<f64>::ones(vec![5, 5]);
    assert_eq!(ones.shape(), &[5, 5]);
    if let Some(data) = ones.data() {
        assert_eq!(data[0], 1.0);
        assert_eq!(data[24], 1.0);
    }

    let from_data = TensorStorage::Dense(DenseTensor::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    ));
    assert_eq!(from_data.shape(), &[2, 2]);
    assert_eq!(from_data.get(&[0, 0]), 1.0);
    assert_eq!(from_data.get(&[1, 1]), 4.0);
}

#[test]
fn test_dense_tensor_creation() {
    // Create tensors using DenseTensor directly
    let zeros = DenseTensor::<f64>::zeros(vec![3, 4]);
    assert_eq!(zeros.shape(), &[3, 4]);
    assert_eq!(zeros.len(), 12);

    let ones = DenseTensor::<f64>::ones(vec![2, 3]);
    assert_eq!(ones.data()[0], 1.0);

    let constant = DenseTensor::constant(vec![2, 2], 3.14);
    assert_eq!(constant.get(&[0, 0]), 3.14);
    assert_eq!(constant.get(&[1, 1]), 3.14);
}

#[test]
fn test_tensor_indexing() {
    let mut tensor = TensorStorage::<f64>::zeros(vec![3, 4]);

    // Set values
    tensor.set(&[0, 0], 1.0);
    tensor.set(&[1, 2], 42.0);
    tensor.set(&[2, 3], 99.0);

    // Get values
    assert_eq!(tensor.get(&[0, 0]), 1.0);
    assert_eq!(tensor.get(&[1, 2]), 42.0);
    assert_eq!(tensor.get(&[2, 3]), 99.0);
}

#[test]
fn test_tensor_fill() {
    let mut tensor = TensorStorage::<f64>::zeros(vec![10, 10]);
    tensor.fill(2.718);

    if let Some(data) = tensor.data() {
        for &val in data {
            assert_eq!(val, 2.718);
        }
    }
}
