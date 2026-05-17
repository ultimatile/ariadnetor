use arnet_linalg::transpose_dense as transpose;
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, MemoryOrder};

/// Create Dense from row-major data, converted to column-major for NativeBackend.
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> Dense<T> {
    let rm = Dense::new(data, shape, MemoryOrder::RowMajor);
    arnet_tensor::reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)
}

/// Convert column-major Dense back to row-major so `.get()` returns correct values.
fn to_rm<T: Clone>(tensor: &Dense<T>) -> Dense<T> {
    arnet_tensor::reorder(tensor, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor)
}

#[test]
fn test_transpose_f64_2d() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let result = to_rm(&transpose(&backend, &tensor, &[1, 0]).unwrap());

    assert_eq!(result.shape(), &[3, 2]);
    // Transposed: [[1,4],[2,5],[3,6]]
    assert_eq!(result.get(&[0, 0]), 1.0);
    assert_eq!(result.get(&[0, 1]), 4.0);
    assert_eq!(result.get(&[1, 0]), 2.0);
    assert_eq!(result.get(&[1, 1]), 5.0);
    assert_eq!(result.get(&[2, 0]), 3.0);
    assert_eq!(result.get(&[2, 1]), 6.0);
}

#[test]
fn test_transpose_f64_3d() {
    let backend = NativeBackend::new();
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let tensor = cm(data, vec![2, 3, 4]);

    let result = to_rm(&transpose(&backend, &tensor, &[2, 0, 1]).unwrap());
    let tensor_rm = to_rm(&tensor);

    assert_eq!(result.shape(), &[4, 2, 3]);
    assert_eq!(result.len(), 24);
    // input[0][0][0] = 0 -> output[0][0][0]
    assert_eq!(result.get(&[0, 0, 0]), tensor_rm.get(&[0, 0, 0]));
    // input[0][0][1] = 1 -> output[1][0][0]
    assert_eq!(result.get(&[1, 0, 0]), tensor_rm.get(&[0, 0, 1]));
}

#[test]
fn test_transpose_f32_2d() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let result = to_rm(&transpose(&backend, &tensor, &[1, 0]).unwrap());

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.get(&[0, 0]), 1.0f32);
    assert_eq!(result.get(&[0, 1]), 4.0f32);
    assert_eq!(result.get(&[1, 0]), 2.0f32);
}

#[test]
fn test_transpose_complex_f64_2d() {
    use num_complex::Complex;

    let backend = NativeBackend::new();
    let input = vec![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
        Complex::new(9.0, 10.0),
        Complex::new(11.0, 12.0),
    ];
    let tensor = cm(input, vec![2, 3]);

    let result = to_rm(&transpose(&backend, &tensor, &[1, 0]).unwrap());

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.get(&[0, 0]), Complex::new(1.0, 2.0));
    assert_eq!(result.get(&[0, 1]), Complex::new(7.0, 8.0));
    assert_eq!(result.get(&[1, 0]), Complex::new(3.0, 4.0));
    assert_eq!(result.get(&[1, 1]), Complex::new(9.0, 10.0));
}

#[test]
fn test_transpose_empty_tensor() {
    let backend = NativeBackend::new();
    let tensor = cm(Vec::<f64>::new(), vec![0, 3]);

    let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

    assert_eq!(result.shape(), &[3, 0]);
    assert_eq!(result.len(), 0);
}
