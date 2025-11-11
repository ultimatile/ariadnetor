//! Arithmetic operations tests for RawTensor and FatTensor
//!
//! Tests for TCI-spec arithmetic operations:
//! - scale: scalar multiplication
//! - linear_combine: linear combination of tensors

use arnet_tensor::{FatTensor, LabelId, RawTensor};
use num_complex::Complex;

// ============================================================================
// RawTensor::scale tests
// ============================================================================

#[test]
fn test_raw_tensor_scale_f64() {
    let mut tensor = RawTensor::<f64>::ones(vec![2, 3]);
    tensor.scale(2.5);

    if let Some(data) = tensor.data() {
        for &val in data {
            assert_eq!(val, 2.5);
        }
    }
}

#[test]
fn test_raw_tensor_scale_f32() {
    let mut tensor = RawTensor::<f32>::ones(vec![3, 2]);
    tensor.scale(3.0f32);

    if let Some(data) = tensor.data() {
        for &val in data {
            assert_eq!(val, 3.0f32);
        }
    }
}

#[test]
fn test_raw_tensor_scale_complex_f64() {
    let mut tensor = RawTensor::<Complex<f64>>::ones(vec![2, 2]);
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
fn test_raw_tensor_scaled_out_of_place() {
    let tensor = RawTensor::<f64>::constant(vec![2, 2], 3.0);
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
// RawTensor::linear_combine tests
// ============================================================================

#[test]
fn test_linear_combine_simple() {
    let a = RawTensor::<f64>::constant(vec![2, 2], 1.0);
    let b = RawTensor::<f64>::constant(vec![2, 2], 2.0);
    let c = RawTensor::<f64>::constant(vec![2, 2], 3.0);

    // 1.0*a + 2.0*b + 3.0*c = 1.0*1 + 2.0*2 + 3.0*3 = 14.0
    let result = RawTensor::linear_combine(&[&a, &b, &c], &[1.0, 2.0, 3.0]).unwrap();

    if let Some(data) = result.data() {
        for &val in data {
            assert_eq!(val, 14.0);
        }
    }
}

#[test]
fn test_linear_combine_f32() {
    let a = RawTensor::<f32>::constant(vec![3, 3], 1.0);
    let b = RawTensor::<f32>::constant(vec![3, 3], 4.0);

    // 2.0*a + 0.5*b = 2.0*1 + 0.5*4 = 4.0
    let result = RawTensor::linear_combine(&[&a, &b], &[2.0f32, 0.5f32]).unwrap();

    if let Some(data) = result.data() {
        for &val in data {
            assert_eq!(val, 4.0f32);
        }
    }
}

#[test]
fn test_linear_combine_complex() {
    let a = RawTensor::<Complex<f64>>::constant(vec![2, 2], Complex::new(1.0, 0.0));
    let b = RawTensor::<Complex<f64>>::constant(vec![2, 2], Complex::new(0.0, 1.0));

    // (2+0i)*(1+0i) + (0+1i)*(0+1i) = (2+0i) + (i^2) = (2+0i) + (-1+0i) = (1+0i)
    let result =
        RawTensor::linear_combine(&[&a, &b], &[Complex::new(2.0, 0.0), Complex::new(0.0, 1.0)])
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
    let result = RawTensor::<f64>::linear_combine(&[], &[]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("empty"));
}

#[test]
fn test_linear_combine_mismatched_lengths() {
    let a = RawTensor::<f64>::ones(vec![2, 2]);
    let result = RawTensor::linear_combine(&[&a], &[1.0, 2.0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Mismatched lengths"));
}

#[test]
fn test_linear_combine_shape_mismatch() {
    let a = RawTensor::<f64>::ones(vec![2, 2]);
    let b = RawTensor::<f64>::ones(vec![3, 3]);
    let result = RawTensor::linear_combine(&[&a, &b], &[1.0, 1.0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("same shape"));
}

// ============================================================================
// RawTensor::add_all tests
// ============================================================================

#[test]
fn test_add_all_simple() {
    let a = RawTensor::<f64>::constant(vec![2, 2], 1.0);
    let b = RawTensor::<f64>::constant(vec![2, 2], 2.0);
    let c = RawTensor::<f64>::constant(vec![2, 2], 3.0);

    // a + b + c = 1 + 2 + 3 = 6
    let result = RawTensor::add_all(&[&a, &b, &c]).unwrap();

    if let Some(data) = result.data() {
        for &val in data {
            assert_eq!(val, 6.0);
        }
    }
}

#[test]
fn test_add_all_complex() {
    let a = RawTensor::<Complex<f64>>::constant(vec![2, 2], Complex::new(1.0, 2.0));
    let b = RawTensor::<Complex<f64>>::constant(vec![2, 2], Complex::new(3.0, 4.0));

    let result = RawTensor::add_all(&[&a, &b]).unwrap();

    if let Some(data) = result.data() {
        for &val in data {
            assert_eq!(val, Complex::new(4.0, 6.0));
        }
    }
}

// ============================================================================
// FatTensor::scale tests
// ============================================================================

#[test]
fn test_fat_tensor_scale() {
    let raw = RawTensor::<f64>::ones(vec![2, 3]);
    let labels = vec![LabelId::intern("i"), LabelId::intern("j")];
    let mut fat = FatTensor::new(raw, labels.clone());

    fat.scale(3.0);

    if let Some(data) = fat.tensor.data() {
        for &val in data {
            assert_eq!(val, 3.0);
        }
    }

    // Labels should be preserved
    assert_eq!(fat.labels, labels);
}

#[test]
fn test_fat_tensor_scaled() {
    let raw = RawTensor::<f64>::constant(vec![2, 2], 2.0);
    let labels = vec![LabelId::intern("a"), LabelId::intern("b")];
    let fat = FatTensor::new(raw, labels.clone());

    let scaled = fat.scaled(5.0);

    // Original unchanged
    if let Some(data) = fat.tensor.data() {
        assert_eq!(data[0], 2.0);
    }

    // Scaled version
    if let Some(data) = scaled.tensor.data() {
        for &val in data {
            assert_eq!(val, 10.0);
        }
    }

    // Labels preserved in both
    assert_eq!(scaled.labels, labels);
}

// ============================================================================
// FatTensor::linear_combine tests
// ============================================================================

#[test]
fn test_fat_tensor_linear_combine() {
    let labels = vec![LabelId::intern("i"), LabelId::intern("j")];

    let a = FatTensor::new(RawTensor::<f64>::constant(vec![2, 2], 1.0), labels.clone());
    let b = FatTensor::new(RawTensor::<f64>::constant(vec![2, 2], 2.0), labels.clone());

    // 3*a + 2*b = 3*1 + 2*2 = 7
    let result = FatTensor::linear_combine(&[&a, &b], &[3.0, 2.0]).unwrap();

    if let Some(data) = result.tensor.data() {
        for &val in data {
            assert_eq!(val, 7.0);
        }
    }

    assert_eq!(result.labels, labels);
}

#[test]
fn test_fat_tensor_linear_combine_index_mismatch() {
    let labels1 = vec![LabelId::intern("i"), LabelId::intern("j")];
    let labels2 = vec![LabelId::intern("k"), LabelId::intern("l")];

    let a = FatTensor::new(RawTensor::<f64>::ones(vec![2, 2]), labels1);
    let b = FatTensor::new(RawTensor::<f64>::ones(vec![2, 2]), labels2);

    let result = FatTensor::linear_combine(&[&a, &b], &[1.0, 1.0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("matching labels"));
}

#[test]
fn test_fat_tensor_add_all() {
    let labels = vec![LabelId::intern("x")];

    let a = FatTensor::new(RawTensor::<f64>::constant(vec![3], 1.0), labels.clone());
    let b = FatTensor::new(RawTensor::<f64>::constant(vec![3], 2.0), labels.clone());
    let c = FatTensor::new(RawTensor::<f64>::constant(vec![3], 4.0), labels.clone());

    let result = FatTensor::add_all(&[&a, &b, &c]).unwrap();

    if let Some(data) = result.tensor.data() {
        for &val in data {
            assert_eq!(val, 7.0);
        }
    }
}
