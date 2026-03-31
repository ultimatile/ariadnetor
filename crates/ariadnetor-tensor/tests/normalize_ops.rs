//! Normalization operations tests
//!
//! Tests for TCI-spec normalization operations:
//! - norm: Frobenius norm (sqrt(sum|element|^2))
//! - normalize: Divide by norm and return the norm value

use arnet_tensor::{Dense, MemoryOrder};
use num_complex::Complex;

const EPSILON: f64 = 1e-10;

// ============================================================================
// Dense::norm tests
// ============================================================================

#[test]
fn test_norm_f64_simple() {
    // Identity matrix 3x3: norm = sqrt(3)
    let mut tensor = Dense::<f64>::zeros(vec![3, 3]);
    tensor.set(&[0, 0], 1.0);
    tensor.set(&[1, 1], 1.0);
    tensor.set(&[2, 2], 1.0);

    let norm = tensor.norm();
    assert!((norm - 3.0f64.sqrt()).abs() < EPSILON);
}

#[test]
fn test_norm_f64_all_ones() {
    // 2x3 tensor of ones: norm = sqrt(6)
    let tensor = Dense::<f64>::ones(vec![2, 3]);
    let norm = tensor.norm();
    assert!((norm - 6.0f64.sqrt()).abs() < EPSILON);
}

#[test]
fn test_norm_f32() {
    let tensor = Dense::<f32>::ones(vec![2, 2]);
    let norm = tensor.norm();
    assert!((norm - 4.0f32.sqrt()).abs() < 1e-6);
}

#[test]
fn test_norm_complex_f64() {
    // [1+0i, 0+1i, 2+0i, 0+2i]
    // |1+0i|^2 = 1, |0+1i|^2 = 1, |2+0i|^2 = 4, |0+2i|^2 = 4
    // sum = 10, norm = sqrt(10)
    let data: Vec<Complex<f64>> = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(2.0, 0.0),
        Complex::new(0.0, 2.0),
    ];
    let tensor = Dense::from_data_with_order(data, vec![2, 2], MemoryOrder::RowMajor);
    let norm = tensor.norm();
    assert!((norm - 10.0f64.sqrt()).abs() < EPSILON);
}

#[test]
fn test_norm_zero_tensor() {
    let tensor = Dense::<f64>::zeros(vec![3, 3]);
    let norm = tensor.norm();
    assert!(norm.abs() < EPSILON);
}

// ============================================================================
// Dense::normalize tests
// ============================================================================

#[test]
fn test_normalize_f64_inplace() {
    let mut tensor = Dense::<f64>::ones(vec![2, 2]);
    // Initial norm = sqrt(4) = 2
    let norm = tensor.normalize();
    assert!((norm - 2.0).abs() < EPSILON);

    // After normalization, each element should be 1/2
    {
        let data = tensor.data();
        for &val in data {
            assert!((val - 0.5).abs() < EPSILON);
        }
    }

    // New norm should be 1
    let new_norm = tensor.norm();
    assert!((new_norm - 1.0).abs() < EPSILON);
}

#[test]
fn test_normalize_f64_out_of_place() {
    let tensor = Dense::<f64>::constant(vec![2, 2], 3.0);
    // Initial norm = sqrt(4*9) = 6
    let (normalized, norm) = tensor.normalized();
    assert!((norm - 6.0).abs() < EPSILON);

    // Original unchanged
    {
        let data = tensor.data();
        assert!((data[0] - 3.0).abs() < EPSILON);
    }

    // Normalized: each element should be 0.5
    {
        let data = normalized.data();
        for &val in data {
            assert!((val - 0.5).abs() < EPSILON);
        }
    }
}

#[test]
fn test_normalize_f32() {
    let mut tensor = Dense::<f32>::ones(vec![3, 3]);
    let norm = tensor.normalize();
    assert!((norm - 3.0f32).abs() < 1e-6);

    {
        let data = tensor.data();
        for &val in data {
            assert!((val - 1.0 / 3.0).abs() < 1e-6);
        }
    }
}

#[test]
fn test_normalize_complex_f64() {
    // [2+0i, 0+2i]: |2+0i|^2 = 4, |0+2i|^2 = 4, sum = 8, norm = sqrt(8) = 2*sqrt(2)
    let data: Vec<Complex<f64>> = vec![Complex::new(2.0, 0.0), Complex::new(0.0, 2.0)];
    let mut tensor = Dense::from_data_with_order(data, vec![2], MemoryOrder::RowMajor);

    let norm = tensor.normalize();
    assert!((norm - 8.0f64.sqrt()).abs() < EPSILON);

    // Each element should be divided by 2*sqrt(2)
    {
        let data = tensor.data();
        // 2/(2*sqrt(2)) = 1/sqrt(2)
        let expected = 1.0 / 2.0f64.sqrt();
        assert!((data[0].re - expected).abs() < EPSILON);
        assert!(data[0].im.abs() < EPSILON);
        assert!(data[1].re.abs() < EPSILON);
        assert!((data[1].im - expected).abs() < EPSILON);
    }

    // New norm should be 1
    let new_norm = tensor.norm();
    assert!((new_norm - 1.0).abs() < EPSILON);
}

#[test]
#[should_panic(expected = "Cannot normalize zero tensor")]
fn test_normalize_zero_tensor_panic() {
    let mut tensor = Dense::<f64>::zeros(vec![2, 2]);
    tensor.normalize();
}

// Tensor-level norm/normalize tests are in ariadnetor/tests/
// since Tensor is now defined in the ariadnetor crate.
