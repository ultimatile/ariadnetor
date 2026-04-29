use arnet_tensor::Dense;

/// Compute row-major flat index for (i, j) in shape [rows, cols]
#[allow(dead_code)]
fn rm(i: usize, j: usize, cols: usize) -> usize {
    i * cols + j
}

#[test]
fn test_tensor_storage_zeros() {
    let tensor = Dense::<f64>::zeros(vec![3, 4]);
    assert_eq!(tensor.shape(), &[3, 4]);
    assert_eq!(tensor.len(), 12);
}

#[test]
fn test_tensor_storage_ones() {
    let tensor = Dense::<f64>::ones(vec![2, 3]);
    {
        let data = tensor.data();
        for &val in data {
            assert_eq!(val, 1.0);
        }
    }
}

#[test]
fn test_tensor_storage_from_dense() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let dense = Dense::new(data.clone(), vec![2, 2]);
    let tensor = dense;
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.data(), &data[..]);
}

#[test]
fn test_tensor_storage_indexing() {
    let mut tensor = Dense::<f64>::zeros(vec![3, 4]);
    tensor.set(&[1, 2], 42.0);
    assert_eq!(tensor.get(&[1, 2]), 42.0);
}
