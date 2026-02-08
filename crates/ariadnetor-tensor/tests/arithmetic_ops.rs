//! Arithmetic operations tests for TensorStorage and Tensor
//!
//! Tests for TCI-spec arithmetic operations:
//! - scale: scalar multiplication
//! - linear_combine: linear combination of tensors

use arnet_tensor::{Tensor, TensorStorage};
use num_complex::Complex;

// ============================================================================
// TensorStorage::scale tests
// ============================================================================

#[test]
fn test_tensor_storage_scale_f64() {
    let mut tensor = TensorStorage::<f64>::ones(vec![2, 3]);
    tensor.scale(2.5);

    if let Some(data) = tensor.data() {
        for &val in data {
            assert_eq!(val, 2.5);
        }
    }
}

#[test]
fn test_tensor_storage_scale_f32() {
    let mut tensor = TensorStorage::<f32>::ones(vec![3, 2]);
    tensor.scale(3.0f32);

    if let Some(data) = tensor.data() {
        for &val in data {
            assert_eq!(val, 3.0f32);
        }
    }
}

#[test]
fn test_tensor_storage_scale_complex_f64() {
    let mut tensor = TensorStorage::<Complex<f64>>::ones(vec![2, 2]);
    let factor = Complex::new(2.0, 1.0);
    tensor.scale(factor);

    if let Some(data) = tensor.data() {
        for &val in data {
            // (1 + 0i) * (2 + 1i) = (2 + 1i)
            assert_eq!(val, Complex::new(2.0, 1.0));
        }
    }
}

#[test]
fn test_tensor_storage_scaled_out_of_place() {
    let tensor = TensorStorage::<f64>::constant(vec![2, 2], 3.0);
    let scaled = tensor.scaled(2.0);

    // Original unchanged
    if let Some(data) = tensor.data() {
        assert_eq!(data[0], 3.0);
    }

    // Scaled version
    if let Some(data) = scaled.data() {
        for &val in data {
            assert_eq!(val, 6.0);
        }
    }
}

// ============================================================================
// TensorStorage::linear_combine tests
// ============================================================================

#[test]
fn test_linear_combine_simple() {
    let a = TensorStorage::<f64>::constant(vec![2, 2], 1.0);
    let b = TensorStorage::<f64>::constant(vec![2, 2], 2.0);
    let c = TensorStorage::<f64>::constant(vec![2, 2], 3.0);

    // 1.0*a + 2.0*b + 3.0*c = 1.0*1 + 2.0*2 + 3.0*3 = 14.0
    let result = TensorStorage::linear_combine(&[&a, &b, &c], &[1.0, 2.0, 3.0]).unwrap();

    if let Some(data) = result.data() {
        for &val in data {
            assert_eq!(val, 14.0);
        }
    }
}

#[test]
fn test_linear_combine_f32() {
    let a = TensorStorage::<f32>::constant(vec![3, 3], 1.0);
    let b = TensorStorage::<f32>::constant(vec![3, 3], 4.0);

    // 2.0*a + 0.5*b = 2.0*1 + 0.5*4 = 4.0
    let result = TensorStorage::linear_combine(&[&a, &b], &[2.0f32, 0.5f32]).unwrap();

    if let Some(data) = result.data() {
        for &val in data {
            assert_eq!(val, 4.0f32);
        }
    }
}

#[test]
fn test_linear_combine_complex() {
    let a = TensorStorage::<Complex<f64>>::constant(vec![2, 2], Complex::new(1.0, 0.0));
    let b = TensorStorage::<Complex<f64>>::constant(vec![2, 2], Complex::new(0.0, 1.0));

    // (2+0i)*(1+0i) + (0+1i)*(0+1i) = (2+0i) + (i^2) = (2+0i) + (-1+0i) = (1+0i)
    let result = TensorStorage::linear_combine(
        &[&a, &b],
        &[Complex::new(2.0, 0.0), Complex::new(0.0, 1.0)],
    )
    .unwrap();

    if let Some(data) = result.data() {
        for &val in data {
            assert_eq!(val.re, 1.0);
            assert_eq!(val.im, 0.0);
        }
    }
}

#[test]
fn test_linear_combine_empty_error() {
    let result = TensorStorage::<f64>::linear_combine(&[], &[]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("empty"));
}

#[test]
fn test_linear_combine_mismatched_lengths() {
    let a = TensorStorage::<f64>::ones(vec![2, 2]);
    let result = TensorStorage::linear_combine(&[&a], &[1.0, 2.0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Mismatched lengths"));
}

#[test]
fn test_linear_combine_shape_mismatch() {
    let a = TensorStorage::<f64>::ones(vec![2, 2]);
    let b = TensorStorage::<f64>::ones(vec![3, 3]);
    let result = TensorStorage::linear_combine(&[&a, &b], &[1.0, 1.0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("same shape"));
}

// ============================================================================
// TensorStorage::add_all tests
// ============================================================================

#[test]
fn test_add_all_simple() {
    let a = TensorStorage::<f64>::constant(vec![2, 2], 1.0);
    let b = TensorStorage::<f64>::constant(vec![2, 2], 2.0);
    let c = TensorStorage::<f64>::constant(vec![2, 2], 3.0);

    // a + b + c = 1 + 2 + 3 = 6
    let result = TensorStorage::add_all(&[&a, &b, &c]).unwrap();

    if let Some(data) = result.data() {
        for &val in data {
            assert_eq!(val, 6.0);
        }
    }
}

#[test]
fn test_add_all_complex() {
    let a = TensorStorage::<Complex<f64>>::constant(vec![2, 2], Complex::new(1.0, 2.0));
    let b = TensorStorage::<Complex<f64>>::constant(vec![2, 2], Complex::new(3.0, 4.0));

    let result = TensorStorage::add_all(&[&a, &b]).unwrap();

    if let Some(data) = result.data() {
        for &val in data {
            assert_eq!(val, Complex::new(4.0, 6.0));
        }
    }
}

// ============================================================================
// Tensor::scale tests
// ============================================================================

#[test]
fn test_tensor_scale() {
    let mut t = Tensor::<f64>::ones(vec![2, 3]);
    t.scale(3.0);

    if let Some(data) = t.storage.data() {
        for &val in data {
            assert_eq!(val, 3.0);
        }
    }
}

#[test]
fn test_tensor_scaled() {
    let t = Tensor::<f64>::constant(vec![2, 2], 2.0);
    let scaled = t.scaled(5.0);

    // Original unchanged
    if let Some(data) = t.storage.data() {
        assert_eq!(data[0], 2.0);
    }

    // Scaled version
    if let Some(data) = scaled.storage.data() {
        for &val in data {
            assert_eq!(val, 10.0);
        }
    }
}

// ============================================================================
// Tensor::linear_combine tests
// ============================================================================

#[test]
fn test_tensor_linear_combine() {
    let a = Tensor::<f64>::constant(vec![2, 2], 1.0);
    let b = Tensor::<f64>::constant(vec![2, 2], 2.0);

    // 3*a + 2*b = 3*1 + 2*2 = 7
    let result = Tensor::linear_combine(&[&a, &b], &[3.0, 2.0]).unwrap();

    if let Some(data) = result.storage.data() {
        for &val in data {
            assert_eq!(val, 7.0);
        }
    }
}

#[test]
fn test_tensor_add_all() {
    let a = Tensor::<f64>::constant(vec![3], 1.0);
    let b = Tensor::<f64>::constant(vec![3], 2.0);
    let c = Tensor::<f64>::constant(vec![3], 4.0);

    let result = Tensor::add_all(&[&a, &b, &c]).unwrap();

    if let Some(data) = result.storage.data() {
        for &val in data {
            assert_eq!(val, 7.0);
        }
    }
}
