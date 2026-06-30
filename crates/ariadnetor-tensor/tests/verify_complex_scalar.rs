//! Verification test for Complex<T> with Scalar trait

use ariadnetor_tensor::{DenseTensorData, MemoryOrder, Scalar};
use num_complex::Complex;

#[test]
fn test_complex_f64_scalar_trait() {
    // Verify Complex<f64> implements Scalar
    let z = Complex::new(3.0, 4.0);

    // Test abs (should be |z| = 5.0)
    assert_eq!(z.abs(), 5.0);

    // Test scale_real
    let scaled = z.scale_real(2.0);
    assert_eq!(scaled, Complex::new(6.0, 8.0));

    // Test conj
    let conjugate = z.conj();
    assert_eq!(conjugate, Complex::new(3.0, -4.0));
}

#[test]
fn test_complex_f64_norm() {
    // Test norm computation for complex tensor
    // [3+4i, 0+0i]
    // |3+4i|² = 9 + 16 = 25
    // |0+0i|² = 0
    // sum = 25, norm = 5.0

    let data = vec![Complex::new(3.0, 4.0), Complex::new(0.0, 0.0)];
    let tensor = DenseTensorData::from_raw_parts(data, vec![2], MemoryOrder::ColumnMajor);

    let norm = tensor.norm();
    assert_eq!(norm, 5.0);
    assert!(std::any::type_name_of_val(&norm).contains("f64"));
}

#[test]
fn test_complex_f64_normalize() {
    // Test normalization for complex tensor
    // [1+0i, 0+1i]
    // |1+0i|² = 1, |0+1i|² = 1
    // sum = 2, norm = √2
    // After normalization: [1/√2+0i, 0+1/√2·i]

    let data = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
    let mut tensor = DenseTensorData::from_raw_parts(data, vec![2], MemoryOrder::ColumnMajor);

    let norm: f64 = tensor.normalize();
    let expected_norm: f64 = 2.0f64.sqrt();
    assert!((norm - expected_norm).abs() < 1e-10);

    // Check normalized values
    {
        let data = tensor.data();
        let expected = 1.0 / expected_norm;
        assert!((data[0].re - expected).abs() < 1e-10);
        assert!(data[0].im.abs() < 1e-10);
        assert!(data[1].re.abs() < 1e-10);
        assert!((data[1].im - expected).abs() < 1e-10);
    }

    // Verify new norm is 1.0
    let new_norm = tensor.norm();
    assert!((new_norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_complex_f32_norm() {
    // Test with f32 complex
    let data = vec![Complex::new(1.0f32, 1.0f32), Complex::new(1.0f32, -1.0f32)];
    let tensor = DenseTensorData::from_raw_parts(data, vec![2], MemoryOrder::ColumnMajor);

    // |1+i|² = 2, |1-i|² = 2
    // sum = 4, norm = 2.0
    let norm = tensor.norm();
    assert!((norm - 2.0f32).abs() < 1e-6);
    assert!(std::any::type_name_of_val(&norm).contains("f32"));
}

#[test]
fn test_complex_scale_real_in_normalize() {
    // Verify that scale_real is correctly used in normalization
    let z = Complex::new(2.0, 2.0);
    let inv_norm = 0.5;

    let scaled = z.scale_real(inv_norm);
    assert_eq!(scaled, Complex::new(1.0, 1.0));
}

#[test]
fn test_norm_returns_real_type() {
    // Verify that norm returns T::Real, not T
    let complex_data = vec![Complex::new(3.0, 4.0)];
    let complex_tensor =
        DenseTensorData::from_raw_parts(complex_data, vec![1], MemoryOrder::ColumnMajor);

    let norm: f64 = complex_tensor.norm(); // Should be f64, not Complex<f64>
    assert_eq!(norm, 5.0);
}

#[test]
fn test_generic_function_with_scalar() {
    // Verify that generic functions work with both real and complex
    fn compute_norm<T: Scalar>(tensor: &DenseTensorData<T>) -> T::Real {
        tensor.norm()
    }

    let real_tensor = DenseTensorData::<f64>::ones_in_order(vec![2, 2], MemoryOrder::ColumnMajor);
    let real_norm: f64 = compute_norm(&real_tensor);
    assert_eq!(real_norm, 2.0);

    let complex_data = vec![Complex::new(3.0, 4.0)];
    let complex_tensor =
        DenseTensorData::from_raw_parts(complex_data, vec![1], MemoryOrder::ColumnMajor);
    let complex_norm: f64 = compute_norm(&complex_tensor);
    assert_eq!(complex_norm, 5.0);
}
