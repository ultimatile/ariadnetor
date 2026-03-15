use arnet_cpu::CpuBackend;
use arnet_linalg::transpose;
use arnet_tensor::DenseTensor;

#[test]
fn test_transpose_f64_2d() {
    let backend = CpuBackend::new();
    let tensor =
        DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_f64_3d() {
    let backend = CpuBackend::new();
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

    let result = transpose(&backend, &tensor, &[2, 0, 1]).unwrap();

    assert_eq!(result.shape(), &[4, 2, 3]);
    assert_eq!(result.len(), 24);
    // input[0][0][0] = 0 → output[0][0][0]
    assert_eq!(result.get(&[0, 0, 0]), 0.0);
    // input[0][0][1] = 1 → output[1][0][0]
    assert_eq!(result.get(&[1, 0, 0]), 1.0);
}

#[test]
fn test_transpose_f32_2d() {
    let backend = CpuBackend::new();
    let tensor =
        DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_complex_f64_2d() {
    use num_complex::Complex;

    let backend = CpuBackend::new();
    let input = vec![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
        Complex::new(9.0, 10.0),
        Complex::new(11.0, 12.0),
    ];
    let tensor = DenseTensor::from_data(input, vec![2, 3]);

    let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.get(&[0, 0]), Complex::new(1.0, 2.0));
    assert_eq!(result.get(&[0, 1]), Complex::new(7.0, 8.0));
    assert_eq!(result.get(&[1, 0]), Complex::new(3.0, 4.0));
    assert_eq!(result.get(&[1, 1]), Complex::new(9.0, 10.0));
}

#[test]
fn test_transpose_empty_tensor() {
    let backend = CpuBackend::new();
    let tensor = DenseTensor::<f64>::from_data(vec![], vec![0, 3]);

    let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

    assert_eq!(result.shape(), &[3, 0]);
    assert_eq!(result.len(), 0);
}
