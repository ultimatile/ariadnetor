//! Tests for slice operations.

mod common;

use ariadnetor_tensor::{DenseTensorData, MemoryOrder};
use common::{ORDERS, multi_indices, offset_within};

#[test]
fn test_slice_column_major() {
    // Column-major 3x3, slice rows 0..2, cols 1..3
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
        vec![3, 3],
        MemoryOrder::ColumnMajor,
    );
    let s = t.slice(&[(0, 2), (1, 3)]);
    assert_eq!(s.shape(), &[2, 2]);
    // CM output: col0=[2,5], col1=[3,6] -> flat [2,5,3,6]
    assert_eq!(s.data(), &[2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_slice_1d() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
        vec![5],
        MemoryOrder::ColumnMajor,
    );
    let s = t.slice(&[(1, 4)]);
    assert_eq!(s.shape(), &[3]);
    assert_eq!(s.data(), &[20.0, 30.0, 40.0]);
}

#[test]
fn test_slice_empty() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let s = t.slice(&[(1, 1), (0, 2)]);
    assert_eq!(s.shape(), &[0, 2]);
    assert_eq!(s.len(), 0);
}

// Rank-0 (scalar) slice must return an identity clone rather than
// underflow on `rank - 1` (RowMajor) or index an empty Vec
// (ColumnMajor). Two tests pin both pre-fix failure paths.

#[test]
fn test_slice_scalar_row_major_returns_clone() {
    let t = DenseTensorData::<f64>::from_raw_parts(vec![42.0], vec![], MemoryOrder::RowMajor);
    let s = t.slice(&[]);
    assert_eq!(s.shape(), &[] as &[usize]);
    assert_eq!(s.data(), &[42.0]);
    assert_eq!(s.order(), MemoryOrder::RowMajor);
}

#[test]
fn test_slice_scalar_column_major_returns_clone() {
    let t = DenseTensorData::<f64>::from_raw_parts(vec![42.0], vec![], MemoryOrder::ColumnMajor);
    let s = t.slice(&[]);
    assert_eq!(s.shape(), &[] as &[usize]);
    assert_eq!(s.data(), &[42.0]);
    assert_eq!(s.order(), MemoryOrder::ColumnMajor);
}

/// Rank-3 slice with a proper sub-range and a non-zero start on every axis:
/// the strip walk must carry across the middle axis, which rank ≤ 2 never
/// exercises.
#[test]
fn test_slice_rank3_matches_elementwise_reference() {
    let shape = [3, 4, 5];
    let starts = [1, 1, 2];
    let ranges = [(1, 3), (1, 3), (2, 5)];
    for order in ORDERS {
        let t = DenseTensorData::from_raw_parts(
            (0..60u32).map(f64::from).collect(),
            shape.to_vec(),
            order,
        );
        let s = t.slice(&ranges);
        assert_eq!(s.shape(), &[2, 2, 3]);
        assert_eq!(s.order(), order);
        for src in multi_indices(&shape) {
            if let Some(dst) = offset_within(&src, &starts, s.shape()) {
                assert_eq!(s.get(&dst), t.get(&src), "{order:?} at {dst:?}");
            }
        }
    }
}
