//! Tests for diagonal_scale optimization.

use arnet_linalg::diagonal_scale;
use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_diagonal_scale_row_major_axis0() {
    // 2×3 matrix, scale rows by [2, 3]
    let t = Dense::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let result = diagonal_scale(&t, &[2.0, 3.0], 0).unwrap();
    assert_eq!(result.get(&[0, 0]), 2.0);
    assert_eq!(result.get(&[0, 2]), 6.0);
    assert_eq!(result.get(&[1, 0]), 12.0);
    assert_eq!(result.get(&[1, 2]), 18.0);
}

#[test]
fn test_diagonal_scale_row_major_axis1() {
    // 2×3 matrix, scale columns by [1, 2, 3]
    let t = Dense::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let result = diagonal_scale(&t, &[1.0, 2.0, 3.0], 1).unwrap();
    assert_eq!(result.get(&[0, 0]), 1.0);
    assert_eq!(result.get(&[0, 1]), 4.0);
    assert_eq!(result.get(&[0, 2]), 9.0);
    assert_eq!(result.get(&[1, 0]), 4.0);
    assert_eq!(result.get(&[1, 1]), 10.0);
    assert_eq!(result.get(&[1, 2]), 18.0);
}

#[test]
fn test_diagonal_scale_column_major_axis0() {
    // Column-major 2×3, scale rows by [2, 3]
    let t = Dense::<f64>::from_data_with_order(
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let result = diagonal_scale(&t, &[2.0, 3.0], 0).unwrap();
    assert_eq!(result.memory_order(), MemoryOrder::ColumnMajor);
    assert_eq!(result.get(&[0, 0]), 2.0);
    assert_eq!(result.get(&[1, 0]), 12.0);
    assert_eq!(result.get(&[0, 2]), 6.0);
    assert_eq!(result.get(&[1, 2]), 18.0);
}

#[test]
fn test_diagonal_scale_column_major_axis1() {
    // Column-major 2×3, scale columns by [1, 2, 3]
    let t = Dense::<f64>::from_data_with_order(
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let result = diagonal_scale(&t, &[1.0, 2.0, 3.0], 1).unwrap();
    assert_eq!(result.get(&[0, 0]), 1.0);
    assert_eq!(result.get(&[0, 1]), 4.0);
    assert_eq!(result.get(&[0, 2]), 9.0);
    assert_eq!(result.get(&[1, 0]), 4.0);
    assert_eq!(result.get(&[1, 2]), 18.0);
}

#[test]
fn test_diagonal_scale_rank1() {
    let t =
        Dense::<f64>::from_data_with_order(vec![10.0, 20.0, 30.0], vec![3], MemoryOrder::RowMajor);
    let result = diagonal_scale(&t, &[2.0, 0.5, 3.0], 0).unwrap();
    assert_eq!(result.data(), &[20.0, 10.0, 90.0]);
}

#[test]
fn test_diagonal_scale_rank3() {
    // 2×3×4, scale along axis 1 by [1, 2, 3]
    let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
    let t = Dense::from_data_with_order(data, vec![2, 3, 4], MemoryOrder::RowMajor);
    let result = diagonal_scale(&t, &[1.0, 2.0, 3.0], 1).unwrap();

    // Verify element-by-element
    for i0 in 0..2 {
        for i1 in 0..3 {
            for i2 in 0..4 {
                let expected = t.get(&[i0, i1, i2]) * [1.0, 2.0, 3.0][i1];
                assert_eq!(
                    result.get(&[i0, i1, i2]),
                    expected,
                    "mismatch at [{i0},{i1},{i2}]"
                );
            }
        }
    }
}

#[test]
fn test_diagonal_scale_non_contiguous() {
    // Non-contiguous: shape [2, 2] with strides [3, 1]
    let t = Dense::<f64>::from_data_with_strides(
        vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0],
        vec![2, 2],
        vec![3, 1],
        0,
        MemoryOrder::RowMajor,
    );
    let result = diagonal_scale(&t, &[10.0, 20.0], 0).unwrap();
    assert_eq!(result.get(&[0, 0]), 10.0);
    assert_eq!(result.get(&[0, 1]), 20.0);
    assert_eq!(result.get(&[1, 0]), 60.0);
    assert_eq!(result.get(&[1, 1]), 80.0);
}

#[test]
fn test_diagonal_scale_error_cases() {
    let t = Dense::<f64>::from_data_with_order(vec![1.0; 6], vec![2, 3], MemoryOrder::RowMajor);
    // axis out of range
    assert!(diagonal_scale(&t, &[1.0, 2.0], 2).is_err());
    // wrong weights length
    assert!(diagonal_scale(&t, &[1.0, 2.0], 0).is_ok()); // 2 == shape[0]
    assert!(diagonal_scale(&t, &[1.0, 2.0], 1).is_err()); // 2 != shape[1]=3
}
