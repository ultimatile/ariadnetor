//! Tests for diagonal_scale.

use arnet_linalg::diagonal_scale_dense as diagonal_scale;
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_diagonal_scale_axis0() {
    // 2x3 matrix in CM layout, scale rows by [2, 3]
    // CM layout of [[1,2,3],[4,5,6]]: col0=[1,4], col1=[2,5], col2=[3,6]
    let backend = NativeBackend::new();
    let t = Dense::<f64>::new(
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let result = diagonal_scale(&backend, &t, &[2.0, 3.0], 0).unwrap();
    // Scale row 0 by 2, row 1 by 3 -> [[2,4,6],[12,15,18]]
    // CM: col0=[2,12], col1=[4,15], col2=[6,18]
    assert_eq!(result.data(), &[2.0, 12.0, 4.0, 15.0, 6.0, 18.0]);
}

#[test]
fn test_diagonal_scale_axis1() {
    // 2x3 matrix in CM layout, scale columns by [1, 2, 3]
    let backend = NativeBackend::new();
    let t = Dense::<f64>::new(
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let result = diagonal_scale(&backend, &t, &[1.0, 2.0, 3.0], 1).unwrap();
    // Scale col 0 by 1, col 1 by 2, col 2 by 3 -> [[1,4,9],[4,10,18]]
    // CM: col0=[1,4], col1=[4,10], col2=[9,18]
    assert_eq!(result.data(), &[1.0, 4.0, 4.0, 10.0, 9.0, 18.0]);
}

#[test]
fn test_diagonal_scale_rank1() {
    let backend = NativeBackend::new();
    let t = Dense::<f64>::new(vec![10.0, 20.0, 30.0], vec![3], MemoryOrder::ColumnMajor);
    let result = diagonal_scale(&backend, &t, &[2.0, 0.5, 3.0], 0).unwrap();
    assert_eq!(result.data(), &[20.0, 10.0, 90.0]);
}

#[test]
fn test_diagonal_scale_error_cases() {
    let backend = NativeBackend::new();
    let t = Dense::<f64>::new(vec![1.0; 6], vec![2, 3], MemoryOrder::ColumnMajor);
    // axis out of range
    assert!(diagonal_scale(&backend, &t, &[1.0, 2.0], 2).is_err());
    // matching weights length for axis 0
    assert!(diagonal_scale(&backend, &t, &[1.0, 2.0], 0).is_ok());
    // wrong weights length for axis 1
    assert!(diagonal_scale(&backend, &t, &[1.0, 2.0], 1).is_err());
}
