use arnet_tensor::Dense;
use num_complex::Complex;

/// Compute row-major flat index for (i, j) in shape [rows, cols]
#[allow(dead_code)]
fn rm(i: usize, j: usize, cols: usize) -> usize {
    i * cols + j
}

#[test]
fn test_scale_basic() {
    let mut tensor = Dense::<f64>::ones(vec![2, 2]);
    tensor.scale(3.0);
    assert_eq!(tensor.get(&[0, 0]), 3.0);
    assert_eq!(tensor.get(&[1, 1]), 3.0);
}

#[test]
fn test_scaled_immutable() {
    let tensor = Dense::<f64>::constant(vec![2, 2], 2.0);
    let scaled = tensor.scaled(5.0);
    assert_eq!(tensor.get(&[0, 0]), 2.0);
    assert_eq!(scaled.get(&[0, 0]), 10.0);
}

#[test]
fn test_scale_complex() {
    let mut tensor = Dense::<Complex<f64>>::ones(vec![2, 2]);
    tensor.scale(Complex::new(2.0, 3.0));
    // (1 + 0i) * (2 + 3i) = (2 + 3i)
    assert_eq!(tensor.get(&[0, 0]), Complex::new(2.0, 3.0));
}

#[test]
fn test_linear_combine_basic() {
    let a = Dense::<f64>::constant(vec![2, 2], 1.0);
    let b = Dense::<f64>::constant(vec![2, 2], 2.0);
    let result = Dense::linear_combine(&[&a, &b], &[3.0, 4.0]).unwrap();
    // 3*1 + 4*2 = 11
    assert_eq!(result.get(&[0, 0]), 11.0);
}

#[test]
fn test_add_all_basic() {
    let a = Dense::<f64>::constant(vec![2, 2], 1.0);
    let b = Dense::<f64>::constant(vec![2, 2], 2.0);
    let c = Dense::<f64>::constant(vec![2, 2], 3.0);
    let result = Dense::add_all(&[&a, &b, &c]).unwrap();
    assert_eq!(result.get(&[0, 0]), 6.0);
}
