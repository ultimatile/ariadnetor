//! Verification test for Complex<T> with Scalar trait, exercised
//! through `DenseTensorData<T>`.

use arnet_tensor::{DenseTensorData, MemoryOrder, Scalar};
use num_complex::Complex;

#[test]
fn complex_f64_scalar_trait() {
    // Verify Complex<f64> implements Scalar at the trait level.
    let z = Complex::new(3.0, 4.0);

    // |z| = 5.0
    assert_eq!(z.abs(), 5.0);

    // scale_real
    let scaled = z.scale_real(2.0);
    assert_eq!(scaled, Complex::new(6.0, 8.0));

    // conj
    let conjugate = z.conj();
    assert_eq!(conjugate, Complex::new(3.0, -4.0));
}

#[test]
fn complex_f64_norm() {
    // [3+4i, 0+0i]: |3+4i|² = 25, |0|² = 0, sum = 25, norm = 5.0.
    let data = vec![Complex::new(3.0, 4.0), Complex::new(0.0, 0.0)];
    let td = DenseTensorData::from_raw_parts(data, vec![2], MemoryOrder::ColumnMajor);

    let norm = td.norm();
    assert_eq!(norm, 5.0);
    assert!(std::any::type_name_of_val(&norm).contains("f64"));
}

#[test]
fn complex_f32_norm() {
    let data = vec![Complex::new(1.0f32, 1.0f32), Complex::new(1.0f32, -1.0f32)];
    let td = DenseTensorData::from_raw_parts(data, vec![2], MemoryOrder::ColumnMajor);

    // |1+i|² = 2, |1-i|² = 2, sum = 4, norm = 2.0
    let norm = td.norm();
    assert!((norm - 2.0f32).abs() < 1e-6);
    assert!(std::any::type_name_of_val(&norm).contains("f32"));
}

#[test]
fn complex_scale_real_in_normalize() {
    // Pin scale_real semantics independently of the tensor surface.
    let z = Complex::new(2.0, 2.0);
    let inv_norm = 0.5;
    let scaled = z.scale_real(inv_norm);
    assert_eq!(scaled, Complex::new(1.0, 1.0));
}

#[test]
fn norm_returns_real_type() {
    // norm() must return T::Real, not T (otherwise complex tensors would
    // expose the imaginary part of a sum-of-squares accumulator).
    let complex_data = vec![Complex::new(3.0, 4.0)];
    let td = DenseTensorData::from_raw_parts(complex_data, vec![1], MemoryOrder::ColumnMajor);

    let norm: f64 = td.norm();
    assert_eq!(norm, 5.0);
}

#[test]
fn generic_function_with_scalar() {
    // Verify a Scalar-generic function works with both real and complex
    // element types on `DenseTensorData`.
    fn compute_norm<T: Scalar>(t: &DenseTensorData<T>) -> T::Real {
        t.norm()
    }

    let real_td = DenseTensorData::<f64>::ones(vec![2, 2]);
    let real_norm: f64 = compute_norm(&real_td);
    assert_eq!(real_norm, 2.0);

    let complex_data = vec![Complex::new(3.0, 4.0)];
    let complex_td =
        DenseTensorData::from_raw_parts(complex_data, vec![1], MemoryOrder::ColumnMajor);
    let complex_norm: f64 = compute_norm(&complex_td);
    assert_eq!(complex_norm, 5.0);
}
