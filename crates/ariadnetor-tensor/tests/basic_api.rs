//! Basic API integration tests
//!
//! Tests the public API usage patterns from design documentation.

use arnet_tensor::Dense;

#[test]
fn test_tensor_storage_creation() {
    // Create tensors using Dense constructors
    let zeros = Dense::<f64>::zeros(vec![10, 20]);
    assert_eq!(zeros.shape(), &[10, 20]);
    assert_eq!(zeros.len(), 200);

    let ones = Dense::<f64>::ones(vec![5, 5]);
    assert_eq!(ones.shape(), &[5, 5]);
    {
        let data = ones.data();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[24], 1.0);
    }

    let from_data = Dense::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(from_data.shape(), &[2, 2]);
    assert_eq!(from_data.data()[0], 1.0);
    assert_eq!(from_data.data()[3], 4.0);
}

#[test]
fn test_dense_tensor_creation() {
    // Create tensors using Dense directly
    let zeros = Dense::<f64>::zeros(vec![3, 4]);
    assert_eq!(zeros.shape(), &[3, 4]);
    assert_eq!(zeros.len(), 12);

    let ones = Dense::<f64>::ones(vec![2, 3]);
    assert_eq!(ones.data()[0], 1.0);

    let constant = Dense::constant(vec![2, 2], 3.15);
    // All elements should be 3.15
    for &v in constant.data() {
        assert_eq!(v, 3.15);
    }
}

#[test]
fn test_tensor_data_mut() {
    let mut tensor = Dense::<f64>::zeros(vec![3, 4]);

    // Set values via data_mut
    tensor.data_mut()[0] = 1.0;
    tensor.data_mut()[6] = 42.0;
    tensor.data_mut()[11] = 99.0;

    // Verify via data
    assert_eq!(tensor.data()[0], 1.0);
    assert_eq!(tensor.data()[6], 42.0);
    assert_eq!(tensor.data()[11], 99.0);
}

#[test]
fn test_tensor_fill() {
    let mut tensor = Dense::<f64>::zeros(vec![10, 10]);
    tensor.fill(2.72);

    {
        let data = tensor.data();
        for &val in data {
            assert_eq!(val, 2.72);
        }
    }
}
