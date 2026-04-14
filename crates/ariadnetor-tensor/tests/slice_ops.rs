//! Tests for slice operations.

use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_slice_column_major() {
    // Column-major 3x3, slice rows 0..2, cols 1..3
    let t = Dense::<f64>::new(
        vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
        vec![3, 3],
    );
    let s = t.slice(&[(0, 2), (1, 3)], MemoryOrder::ColumnMajor);
    assert_eq!(s.shape(), &[2, 2]);
    // CM output: col0=[2,5], col1=[3,6] -> flat [2,5,3,6]
    assert_eq!(s.data(), &[2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_slice_1d() {
    let t = Dense::<f64>::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5]);
    let s = t.slice(&[(1, 4)], MemoryOrder::RowMajor);
    assert_eq!(s.shape(), &[3]);
    assert_eq!(s.data(), &[20.0, 30.0, 40.0]);
}

#[test]
fn test_slice_empty() {
    let t = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let s = t.slice(&[(1, 1), (0, 2)], MemoryOrder::RowMajor);
    assert_eq!(s.shape(), &[0, 2]);
    assert_eq!(s.len(), 0);
}
