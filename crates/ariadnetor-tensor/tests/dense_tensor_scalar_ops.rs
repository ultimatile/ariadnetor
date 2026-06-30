//! Integration tests for the inherent unary scalar ops on the joined
//! `DenseTensor` surface (`scaled`, `norm`, `normalized`). The
//! `DenseTensorData::*` form is covered separately in `normalize_ops.rs`.

use ariadnetor_tensor::{DenseTensor, DenseTensorData, MemoryOrder};

/// Wrap a `DenseTensorData<T>` (built with a specific `MemoryOrder`
/// distinct from the host preferred order in some tests) into the joined
/// `DenseTensor<T>` surface, preserving the data's order.
fn t<T: Clone>(d: DenseTensorData<T>) -> DenseTensor<T> {
    DenseTensor::from_data(d)
}

// --- scaled ---

#[test]
fn test_scaled_f64() {
    let tensor = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    ));
    let scaled = tensor.scaled(2.5);
    assert_eq!(scaled.get([0, 0]), 2.5);
    assert_eq!(scaled.get([0, 1]), 5.0);
    assert_eq!(scaled.get([1, 0]), 7.5);
    assert_eq!(scaled.get([1, 1]), 10.0);
    // Original unchanged
    assert_eq!(tensor.get([0, 0]), 1.0);
}

#[test]
fn test_scaled_complex() {
    use num_complex::Complex;
    let tensor = t(DenseTensorData::from_raw_parts(
        vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
        vec![2],
        MemoryOrder::ColumnMajor,
    ));
    let scaled = tensor.scaled(Complex::new(2.0, 3.0));
    // (1+0i)*(2+3i) = 2+3i
    assert_eq!(scaled.data_slice()[0], Complex::new(2.0, 3.0));
    // (0+1i)*(2+3i) = -3+2i
    assert_eq!(scaled.data_slice()[1], Complex::new(-3.0, 2.0));
}

#[test]
fn test_scaled_column_major() {
    // CM layout for [[1,2],[3,4]]: col0=[1,3], col1=[2,4] -> flat [1,3,2,4]
    let tensor = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    ));
    let scaled = tensor.scaled(2.0);
    // Scaled flat: [2, 6, 4, 8]
    assert_eq!(scaled.data_slice(), &[2.0, 6.0, 4.0, 8.0]);
}

// --- norm ---

#[test]
fn test_norm_f64() {
    let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
    let n = tensor.norm();
    assert!((n - 2.0).abs() < 1e-10);
}

#[test]
fn test_norm_complex() {
    use num_complex::Complex;
    // |3+4i| = 5, so norm of single element [3+4i] = 5
    let tensor = t(DenseTensorData::from_raw_parts(
        vec![Complex::new(3.0, 4.0)],
        vec![1],
        MemoryOrder::ColumnMajor,
    ));
    let n: f64 = tensor.norm();
    assert!((n - 5.0).abs() < 1e-10);
}

#[test]
fn test_norm_column_major() {
    let tensor = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    ));
    // norm = sqrt(1+4+9+16) = sqrt(30)
    let n = tensor.norm();
    assert!((n - 30.0_f64.sqrt()).abs() < 1e-10);
}

// --- normalized ---

#[test]
fn test_normalized_f64() {
    let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
    let (normalized, n) = tensor.normalized();
    assert!((n - 2.0).abs() < 1e-10);
    assert!((normalized.norm() - 1.0).abs() < 1e-10);
    // Original unchanged
    assert_eq!(tensor.get([0, 0]), 1.0);
}

#[test]
#[should_panic(expected = "Cannot normalize zero tensor")]
fn test_normalized_zero_panics() {
    let tensor = DenseTensor::<f64>::zeros(vec![2, 2]);
    let _ = tensor.normalized();
}

#[test]
fn test_normalized_column_major() {
    let tensor = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    ));
    let (normalized, n) = tensor.normalized();
    assert!((normalized.norm() - 1.0).abs() < 1e-10);
    // Verify flat data is preserved (just scaled)
    let expected_scale = 1.0 / n;
    assert!((normalized.data_slice()[0] - 1.0 * expected_scale).abs() < 1e-10);
    assert!((normalized.data_slice()[1] - 3.0 * expected_scale).abs() < 1e-10);
}

// --- scalar-multiplication operators (`*`, `*=`) ---
//
// Convenience aliases for `scale` / `scaled` on the joined `DenseTensor`
// surface; equivalence to the named methods and the borrow-vs-consume
// split are the properties under test.

#[test]
fn test_mul_owned_f64() {
    let tensor = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    ));
    let product = tensor * 2.5;
    assert_eq!(product.get([0, 0]), 2.5);
    assert_eq!(product.get([0, 1]), 5.0);
    assert_eq!(product.get([1, 0]), 7.5);
    assert_eq!(product.get([1, 1]), 10.0);
}

#[test]
fn test_mul_ref_leaves_original() {
    let tensor = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    ));
    let product = &tensor * 2.5;
    // Borrowing form does not consume or mutate the original.
    assert_eq!(tensor.get([0, 0]), 1.0);
    assert_eq!(product.get([1, 0]), 7.5);
}

#[test]
fn test_mul_assign_f64() {
    let mut tensor = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    ));
    let mut expected = tensor.clone();
    expected.scale(2.5);
    tensor *= 2.5;
    assert_eq!(tensor.data_slice(), expected.data_slice());
}

#[test]
fn test_mul_complex() {
    use num_complex::Complex;
    let tensor = t(DenseTensorData::from_raw_parts(
        vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
        vec![2],
        MemoryOrder::ColumnMajor,
    ));
    let factor = Complex::new(2.0, 3.0);
    // (1+0i)*(2+3i) = 2+3i ; (0+1i)*(2+3i) = -3+2i
    let expected = &[Complex::new(2.0, 3.0), Complex::new(-3.0, 2.0)];
    // All three operator forms must agree for a complex factor.
    assert_eq!((&tensor * factor).data_slice(), expected);
    assert_eq!((tensor.clone() * factor).data_slice(), expected);
    let mut assigned = tensor;
    assigned *= factor;
    assert_eq!(assigned.data_slice(), expected);
}

#[test]
fn test_mul_column_major_flat() {
    let tensor = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    ));
    let product = &tensor * 2.0;
    assert_eq!(product.data_slice(), &[2.0, 6.0, 4.0, 8.0]);
}
