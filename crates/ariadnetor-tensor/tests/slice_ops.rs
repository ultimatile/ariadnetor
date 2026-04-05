//! Tests for slice optimization paths (strip copy, incremental flat index).

use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_slice_column_major() {
    // Column-major 3×3, slice rows 0..2, cols 1..3
    let t = Dense::<f64>::from_data_with_order(
        vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
        vec![3, 3],
        MemoryOrder::ColumnMajor,
    );
    let s = t.slice(&[(0, 2), (1, 3)]);
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.memory_order(), MemoryOrder::ColumnMajor);
    // Column-major output: col0=[2,5], col1=[3,6] → flat [2,5,3,6]
    assert_eq!(s.data(), &[2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_slice_1d() {
    let t = Dense::<f64>::from_data_with_order(
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
        vec![5],
        MemoryOrder::RowMajor,
    );
    let s = t.slice(&[(1, 4)]);
    assert_eq!(s.shape(), &[3]);
    assert_eq!(s.data(), &[20.0, 30.0, 40.0]);
}

#[test]
fn test_slice_empty() {
    let t = Dense::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let s = t.slice(&[(1, 1), (0, 2)]);
    assert_eq!(s.shape(), &[0, 2]);
    assert_eq!(s.len(), 0);
}

#[test]
fn test_slice_non_contiguous() {
    // Transposed (non-contiguous) tensor: original 2×3 row-major → 3×2 view
    let t = Dense::<f64>::from_data_with_strides(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3, 2],
        vec![1, 3],
        0,
        MemoryOrder::RowMajor,
    );
    // t is: [[1,4],[2,5],[3,6]]
    assert_eq!(t.get(&[0, 0]), 1.0);
    assert_eq!(t.get(&[0, 1]), 4.0);
    assert_eq!(t.get(&[1, 0]), 2.0);
    assert_eq!(t.get(&[2, 1]), 6.0);

    // Slice rows 0..2, all cols → [[1,4],[2,5]]
    let s = t.slice(&[(0, 2), (0, 2)]);
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.data(), &[1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn test_slice_vs_naive() {
    // Exhaustive check: slice a 4×5×3 tensor, compare against get()
    let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
    let t = Dense::from_data_with_order(data, vec![4, 5, 3], MemoryOrder::RowMajor);
    let ranges = [(1, 3), (0, 4), (1, 3)];
    let s = t.slice(&ranges);

    let new_shape: Vec<usize> = ranges.iter().map(|&(a, b)| b - a).collect();
    assert_eq!(s.shape(), &new_shape);

    for i0 in 0..new_shape[0] {
        for i1 in 0..new_shape[1] {
            for i2 in 0..new_shape[2] {
                let expected = t.get(&[i0 + ranges[0].0, i1 + ranges[1].0, i2 + ranges[2].0]);
                let actual = s.get(&[i0, i1, i2]);
                assert_eq!(actual, expected, "mismatch at [{i0},{i1},{i2}]");
            }
        }
    }
}
