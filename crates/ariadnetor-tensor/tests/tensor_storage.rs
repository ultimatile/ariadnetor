use arnet_tensor::{DenseTensor, MemoryOrder, TensorStorage};

#[test]
fn test_tensor_storage_zeros() {
    let tensor = TensorStorage::<f64>::zeros(vec![3, 4]);
    assert_eq!(tensor.shape(), &[3, 4]);
    assert_eq!(tensor.len(), 12);
}

#[test]
fn test_tensor_storage_ones() {
    let tensor = TensorStorage::<f64>::ones(vec![2, 3]);
    if let Some(data) = tensor.data() {
        for &val in data {
            assert_eq!(val, 1.0);
        }
    }
}

#[test]
fn test_tensor_storage_from_dense() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let dense = DenseTensor::from_data_with_order(data.clone(), vec![2, 2], MemoryOrder::RowMajor);
    let tensor = TensorStorage::Dense(dense);
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.data().unwrap(), &data[..]);
}

#[test]
fn test_tensor_storage_indexing() {
    let mut tensor = TensorStorage::<f64>::zeros(vec![3, 4]);
    tensor.set(&[1, 2], 42.0);
    assert_eq!(tensor.get(&[1, 2]), 42.0);
}
