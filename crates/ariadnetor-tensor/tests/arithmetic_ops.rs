//! Arithmetic operations tests for DenseTensorData and Tensor
//!
//! Tests for TCI-spec arithmetic operations:
//! - scale: scalar multiplication
//! - linear_combine: linear combination of tensors

use arnet_tensor::{DenseTensorData, MemoryOrder};
use num_complex::Complex;

// ============================================================================
// DenseTensorData::scale tests
// ============================================================================

#[test]
fn test_tensor_storage_scale_f64() {
    let mut tensor = DenseTensorData::<f64>::ones_in_order(vec![2, 3], MemoryOrder::ColumnMajor);
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
    let mut tensor = DenseTensorData::<f32>::ones_in_order(vec![3, 2], MemoryOrder::ColumnMajor);
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
    let mut tensor =
        DenseTensorData::<Complex<f64>>::ones_in_order(vec![2, 2], MemoryOrder::ColumnMajor);
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
    let tensor = DenseTensorData::<f64>::filled_in_order(vec![2, 2], 3.0, MemoryOrder::ColumnMajor);
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
// DenseTensorData::linear_combine tests
// ============================================================================

#[test]
fn test_linear_combine_simple() {
    let a = DenseTensorData::<f64>::filled_in_order(vec![2, 2], 1.0, MemoryOrder::ColumnMajor);
    let b = DenseTensorData::<f64>::filled_in_order(vec![2, 2], 2.0, MemoryOrder::ColumnMajor);
    let c = DenseTensorData::<f64>::filled_in_order(vec![2, 2], 3.0, MemoryOrder::ColumnMajor);

    // 1.0*a + 2.0*b + 3.0*c = 1.0*1 + 2.0*2 + 3.0*3 = 14.0
    let result = DenseTensorData::linear_combine(&[&a, &b, &c], &[1.0, 2.0, 3.0]).unwrap();

    {
        let data = result.data();
        for &val in data {
            assert_eq!(val, 14.0);
        }
    }
}

#[test]
fn test_linear_combine_f32() {
    let a = DenseTensorData::<f32>::filled_in_order(vec![3, 3], 1.0, MemoryOrder::ColumnMajor);
    let b = DenseTensorData::<f32>::filled_in_order(vec![3, 3], 4.0, MemoryOrder::ColumnMajor);

    // 2.0*a + 0.5*b = 2.0*1 + 0.5*4 = 4.0
    let result = DenseTensorData::linear_combine(&[&a, &b], &[2.0f32, 0.5f32]).unwrap();

    {
        let data = result.data();
        for &val in data {
            assert_eq!(val, 4.0f32);
        }
    }
}

#[test]
fn test_linear_combine_complex() {
    let a = DenseTensorData::<Complex<f64>>::filled_in_order(
        vec![2, 2],
        Complex::new(1.0, 0.0),
        MemoryOrder::ColumnMajor,
    );
    let b = DenseTensorData::<Complex<f64>>::filled_in_order(
        vec![2, 2],
        Complex::new(0.0, 1.0),
        MemoryOrder::ColumnMajor,
    );

    // (2+0i)*(1+0i) + (0+1i)*(0+1i) = (2+0i) + (i^2) = (2+0i) + (-1+0i) = (1+0i)
    let result = DenseTensorData::linear_combine(
        &[&a, &b],
        &[Complex::new(2.0, 0.0), Complex::new(0.0, 1.0)],
    )
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
    let result = DenseTensorData::<f64>::linear_combine(&[], &[]);
    let Err(msg) = result else {
        panic!("expected Err for empty input");
    };
    assert!(msg.to_string().contains("empty"));
}

#[test]
fn test_linear_combine_mismatched_lengths() {
    let a = DenseTensorData::<f64>::ones_in_order(vec![2, 2], MemoryOrder::ColumnMajor);
    let result = DenseTensorData::linear_combine(&[&a], &[1.0, 2.0]);
    let Err(msg) = result else {
        panic!("expected Err for length mismatch");
    };
    assert!(msg.to_string().contains("Mismatched lengths"));
}

#[test]
fn test_linear_combine_shape_mismatch() {
    let a = DenseTensorData::<f64>::ones_in_order(vec![2, 2], MemoryOrder::ColumnMajor);
    let b = DenseTensorData::<f64>::ones_in_order(vec![3, 3], MemoryOrder::ColumnMajor);
    let result = DenseTensorData::linear_combine(&[&a, &b], &[1.0, 1.0]);
    let Err(msg) = result else {
        panic!("expected Err for shape mismatch");
    };
    assert!(msg.to_string().contains("same shape"));
}

// ============================================================================
// DenseTensorData::add_all tests
// ============================================================================

#[test]
fn test_add_all_simple() {
    let a = DenseTensorData::<f64>::filled_in_order(vec![2, 2], 1.0, MemoryOrder::ColumnMajor);
    let b = DenseTensorData::<f64>::filled_in_order(vec![2, 2], 2.0, MemoryOrder::ColumnMajor);
    let c = DenseTensorData::<f64>::filled_in_order(vec![2, 2], 3.0, MemoryOrder::ColumnMajor);

    // a + b + c = 1 + 2 + 3 = 6
    let result = DenseTensorData::add_all(&[&a, &b, &c]).unwrap();

    {
        let data = result.data();
        for &val in data {
            assert_eq!(val, 6.0);
        }
    }
}

#[test]
fn test_add_all_complex() {
    let a = DenseTensorData::<Complex<f64>>::filled_in_order(
        vec![2, 2],
        Complex::new(1.0, 2.0),
        MemoryOrder::ColumnMajor,
    );
    let b = DenseTensorData::<Complex<f64>>::filled_in_order(
        vec![2, 2],
        Complex::new(3.0, 4.0),
        MemoryOrder::ColumnMajor,
    );

    let result = DenseTensorData::add_all(&[&a, &b]).unwrap();

    {
        let data = result.data();
        for &val in data {
            assert_eq!(val, Complex::new(4.0, 6.0));
        }
    }
}

// Tensor-level tests (scale, linear_combine, etc.) are in ariadnetor/tests/
// since Tensor is now defined in the ariadnetor crate.
