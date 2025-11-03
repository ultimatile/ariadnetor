//! Normalization operations tests
//!
//! Tests for TCI-spec normalization operations:
//! - norm: Frobenius norm (√(Σ|element|²))
//! - normalize: Divide by norm and return the norm value

use arnet_tensor::{FatTensor, Index, IndexSet, RawTensor};
use num_complex::Complex;

const EPSILON: f64 = 1e-10;

// ============================================================================
// RawTensor::norm tests
// ============================================================================

#[test]
fn test_norm_f64_simple() {
    // Identity matrix 3x3: norm = √3
    let mut tensor = RawTensor::<f64>::zeros(vec![3, 3]);
    tensor.set(&[0, 0], 1.0);
    tensor.set(&[1, 1], 1.0);
    tensor.set(&[2, 2], 1.0);

    let norm = tensor.norm();
    assert!((norm - 3.0f64.sqrt()).abs() < EPSILON);
}

#[test]
fn test_norm_f64_all_ones() {
    // 2x3 tensor of ones: norm = √6
    let tensor = RawTensor::<f64>::ones(vec![2, 3]);
    let norm = tensor.norm();
    assert!((norm - 6.0f64.sqrt()).abs() < EPSILON);
}

#[test]
fn test_norm_f32() {
    let tensor = RawTensor::<f32>::ones(vec![2, 2]);
    let norm = tensor.norm();
    assert!((norm - 4.0f32.sqrt()).abs() < 1e-6);
}

#[test]
fn test_norm_complex_f64() {
    // [1+0i, 0+1i, 2+0i, 0+2i]
    // |1+0i|² = 1, |0+1i|² = 1, |2+0i|² = 4, |0+2i|² = 4
    // sum = 10, norm = √10
    let data: Vec<Complex<f64>> = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(2.0, 0.0),
        Complex::new(0.0, 2.0),
    ];
    let tensor = RawTensor::from_data(data, vec![2, 2]);
    let norm = tensor.norm();
    assert!((norm - 10.0f64.sqrt()).abs() < EPSILON);
}

#[test]
fn test_norm_zero_tensor() {
    let tensor = RawTensor::<f64>::zeros(vec![3, 3]);
    let norm = tensor.norm();
    assert!(norm.abs() < EPSILON);
}

// ============================================================================
// RawTensor::normalize tests
// ============================================================================

#[test]
fn test_normalize_f64_inplace() {
    let mut tensor = RawTensor::<f64>::ones(vec![2, 2]);
    // Initial norm = √4 = 2
    let norm = tensor.normalize();
    assert!((norm - 2.0).abs() < EPSILON);

    // After normalization, each element should be 1/2
    if let Some(data) = tensor.data() {
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
    let tensor = RawTensor::<f64>::constant(vec![2, 2], 3.0);
    // Initial norm = √(4*9) = 6
    let (normalized, norm) = tensor.normalized();
    assert!((norm - 6.0).abs() < EPSILON);

    // Original unchanged
    if let Some(data) = tensor.data() {
        assert!((data[0] - 3.0).abs() < EPSILON);
    }

    // Normalized: each element should be 0.5
    if let Some(data) = normalized.data() {
        for &val in data {
            assert!((val - 0.5).abs() < EPSILON);
        }
    }
}

#[test]
fn test_normalize_f32() {
    let mut tensor = RawTensor::<f32>::ones(vec![3, 3]);
    let norm = tensor.normalize();
    assert!((norm - 3.0f32).abs() < 1e-6);

    if let Some(data) = tensor.data() {
        for &val in data {
            assert!((val - 1.0 / 3.0).abs() < 1e-6);
        }
    }
}

#[test]
fn test_normalize_complex_f64() {
    // [2+0i, 0+2i]: |2+0i|² = 4, |0+2i|² = 4, sum = 8, norm = √8 = 2√2
    let data: Vec<Complex<f64>> = vec![Complex::new(2.0, 0.0), Complex::new(0.0, 2.0)];
    let mut tensor = RawTensor::from_data(data, vec![2]);

    let norm = tensor.normalize();
    assert!((norm - 8.0f64.sqrt()).abs() < EPSILON);

    // Each element should be divided by 2√2
    if let Some(data) = tensor.data() {
        // 2/(2√2) = 1/√2
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
    let mut tensor = RawTensor::<f64>::zeros(vec![2, 2]);
    tensor.normalize();
}

// ============================================================================
// FatTensor::norm tests
// ============================================================================

#[test]
fn test_fat_tensor_norm() {
    let raw = RawTensor::<f64>::ones(vec![2, 3]);
    let indices = IndexSet::new(vec![Index::with_dim("i", 2), Index::with_dim("j", 3)], 0);
    let fat = FatTensor::new(raw, indices);

    let norm = fat.norm();
    assert!((norm - 6.0f64.sqrt()).abs() < EPSILON);
}

// ============================================================================
// FatTensor::normalize tests
// ============================================================================

#[test]
fn test_fat_tensor_normalize_inplace() {
    let raw = RawTensor::<f64>::ones(vec![2, 2]);
    let indices = IndexSet::new(vec![Index::with_dim("a", 2), Index::with_dim("b", 2)], 0);
    let mut fat = FatTensor::new(raw, indices.clone());

    let norm = fat.normalize();
    assert!((norm - 2.0).abs() < EPSILON);

    // Check normalized values
    if let Some(data) = fat.tensor.data() {
        for &val in data {
            assert!((val - 0.5).abs() < EPSILON);
        }
    }

    // Indices should be preserved
    assert_eq!(fat.indices, indices);
}

#[test]
fn test_fat_tensor_normalized_out_of_place() {
    let raw = RawTensor::<f64>::constant(vec![3, 3], 2.0);
    let indices = IndexSet::new(vec![Index::with_dim("x", 3), Index::with_dim("y", 3)], 0);
    let fat = FatTensor::new(raw, indices.clone());

    let (normalized, norm) = fat.normalized();
    assert!((norm - 6.0).abs() < EPSILON);

    // Original unchanged
    if let Some(data) = fat.tensor.data() {
        assert!((data[0] - 2.0).abs() < EPSILON);
    }

    // Normalized
    if let Some(data) = normalized.tensor.data() {
        for &val in data {
            assert!((val - 1.0 / 3.0).abs() < EPSILON);
        }
    }

    // Indices preserved
    assert_eq!(normalized.indices, indices);
}
