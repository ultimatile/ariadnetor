//! Arithmetic operations tests for Dense and Tensor
//!
//! Tests for TCI-spec arithmetic operations:
//! - scale: scalar multiplication
//! - linear_combine: linear combination of tensors

use arnet_tensor::Dense;
use num_complex::Complex;

// ============================================================================
// Dense::scale tests
// ============================================================================

#[test]
fn test_tensor_storage_scale_f64() {
    let mut tensor = Dense::<f64>::ones(vec![2, 3]);
    tensor.scale(2.5);

    {
        let data = tensor.data();
        for &val in data {
            assert_eq!(val, 2.5);
        }
    }
}

#[test]
fn test_tensor_storage_scale_f32() {
    let mut tensor = Dense::<f32>::ones(vec![3, 2]);
    tensor.scale(3.0f32);

    {
        let data = tensor.data();
        for &val in data {
            assert_eq!(val, 3.0f32);
        }
    }
}

#[test]
fn test_tensor_storage_scale_complex_f64() {
    let mut tensor = Dense::<Complex<f64>>::ones(vec![2, 2]);
    let factor = Complex::new(2.0, 1.0);
    tensor.scale(factor);

    {
        let data = tensor.data();
        for &val in data {
            // (1 + 0i) * (2 + 1i) = (2 + 1i)
            assert_eq!(val, Complex::new(2.0, 1.0));
        }
    }
}

#[test]
fn test_tensor_storage_scaled_out_of_place() {
    let tensor = Dense::<f64>::constant(vec![2, 2], 3.0);
    let scaled = tensor.scaled(2.0);

    // Original unchanged
    {
        let data = tensor.data();
        assert_eq!(data[0], 3.0);
    }

    // Scaled version
    {
        let data = scaled.data();
        for &val in data {
            assert_eq!(val, 6.0);
        }
    }
}

// ============================================================================
// Dense::linear_combine tests
// ============================================================================

#[test]
fn test_linear_combine_simple() {
    let a = Dense::<f64>::constant(vec![2, 2], 1.0);
    let b = Dense::<f64>::constant(vec![2, 2], 2.0);
    let c = Dense::<f64>::constant(vec![2, 2], 3.0);

    // 1.0*a + 2.0*b + 3.0*c = 1.0*1 + 2.0*2 + 3.0*3 = 14.0
    let result = Dense::linear_combine(&[&a, &b, &c], &[1.0, 2.0, 3.0]).unwrap();

    {
        let data = result.data();
        for &val in data {
            assert_eq!(val, 14.0);
        }
    }
}

#[test]
fn test_linear_combine_f32() {
    let a = Dense::<f32>::constant(vec![3, 3], 1.0);
    let b = Dense::<f32>::constant(vec![3, 3], 4.0);

    // 2.0*a + 0.5*b = 2.0*1 + 0.5*4 = 4.0
    let result = Dense::linear_combine(&[&a, &b], &[2.0f32, 0.5f32]).unwrap();

    {
        let data = result.data();
        for &val in data {
            assert_eq!(val, 4.0f32);
        }
    }
}

#[test]
fn test_linear_combine_complex() {
    let a = Dense::<Complex<f64>>::constant(vec![2, 2], Complex::new(1.0, 0.0));
    let b = Dense::<Complex<f64>>::constant(vec![2, 2], Complex::new(0.0, 1.0));

    // (2+0i)*(1+0i) + (0+1i)*(0+1i) = (2+0i) + (i^2) = (2+0i) + (-1+0i) = (1+0i)
    let result =
        Dense::linear_combine(&[&a, &b], &[Complex::new(2.0, 0.0), Complex::new(0.0, 1.0)])
            .unwrap();

    {
        let data = result.data();
        for &val in data {
            assert_eq!(val.re, 1.0);
            assert_eq!(val.im, 0.0);
        }
    }
}

#[test]
fn test_linear_combine_empty_error() {
    let result = Dense::<f64>::linear_combine(&[], &[]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("empty"));
}

#[test]
fn test_linear_combine_mismatched_lengths() {
    let a = Dense::<f64>::ones(vec![2, 2]);
    let result = Dense::linear_combine(&[&a], &[1.0, 2.0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Mismatched lengths"));
}

#[test]
fn test_linear_combine_shape_mismatch() {
    let a = Dense::<f64>::ones(vec![2, 2]);
    let b = Dense::<f64>::ones(vec![3, 3]);
    let result = Dense::linear_combine(&[&a, &b], &[1.0, 1.0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("same shape"));
}

// ============================================================================
// Dense::add_all tests
// ============================================================================

#[test]
fn test_add_all_simple() {
    let a = Dense::<f64>::constant(vec![2, 2], 1.0);
    let b = Dense::<f64>::constant(vec![2, 2], 2.0);
    let c = Dense::<f64>::constant(vec![2, 2], 3.0);

    // a + b + c = 1 + 2 + 3 = 6
    let result = Dense::add_all(&[&a, &b, &c]).unwrap();

    {
        let data = result.data();
        for &val in data {
            assert_eq!(val, 6.0);
        }
    }
}

#[test]
fn test_add_all_complex() {
    let a = Dense::<Complex<f64>>::constant(vec![2, 2], Complex::new(1.0, 2.0));
    let b = Dense::<Complex<f64>>::constant(vec![2, 2], Complex::new(3.0, 4.0));

    let result = Dense::add_all(&[&a, &b]).unwrap();

    {
        let data = result.data();
        for &val in data {
            assert_eq!(val, Complex::new(4.0, 6.0));
        }
    }
}

// Tensor-level tests (scale, linear_combine, etc.) are in ariadnetor/tests/
// since Tensor is now defined in the ariadnetor crate.
