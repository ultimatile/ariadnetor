//! Tests for expand optimization paths (strip copy, incremental flat index).

use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_expand_column_major() {
    // Column-major 2×2 with padding (1,1) on each axis → 4×4
    let t = Dense::<f64>::from_data_with_order(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let e = t.expand(&[(1, 1), (1, 1)]);
    assert_eq!(e.shape(), &[4, 4]);
    assert_eq!(e.memory_order(), MemoryOrder::ColumnMajor);
    // Verify source elements are placed correctly
    assert_eq!(e.get(&[1, 1]), 1.0);
    assert_eq!(e.get(&[2, 1]), 3.0);
    assert_eq!(e.get(&[1, 2]), 2.0);
    assert_eq!(e.get(&[2, 2]), 4.0);
    // Padding should be zero
    assert_eq!(e.get(&[0, 0]), 0.0);
    assert_eq!(e.get(&[3, 3]), 0.0);
}

#[test]
fn test_expand_3d() {
    let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
    let t = Dense::from_data_with_order(data, vec![2, 3, 4], MemoryOrder::RowMajor);
    let e = t.expand(&[(1, 0), (0, 1), (2, 2)]);
    assert_eq!(e.shape(), &[3, 4, 8]);
    // First row of padding
    assert_eq!(e.get(&[0, 0, 0]), 0.0);
    // Source data starts at [1, 0, 2]
    assert_eq!(e.get(&[1, 0, 2]), 1.0);
    assert_eq!(e.get(&[1, 0, 5]), 4.0);
    assert_eq!(e.get(&[2, 2, 5]), 24.0);
    // Padding after
    assert_eq!(e.get(&[1, 3, 0]), 0.0);
}

#[test]
fn test_expand_no_inner_pad() {
    // No padding on innermost axis → should hit strip-copy path
    let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let t = Dense::from_data_with_order(data, vec![3, 4], MemoryOrder::RowMajor);
    let e = t.expand(&[(2, 1), (0, 0)]);
    assert_eq!(e.shape(), &[6, 4]);
    // Rows 0-1: zeros
    assert_eq!(e.get(&[0, 0]), 0.0);
    assert_eq!(e.get(&[1, 3]), 0.0);
    // Rows 2-4: source data
    assert_eq!(e.get(&[2, 0]), 1.0);
    assert_eq!(e.get(&[2, 3]), 4.0);
    assert_eq!(e.get(&[4, 3]), 12.0);
    // Row 5: zeros
    assert_eq!(e.get(&[5, 0]), 0.0);
}

#[test]
fn test_expand_non_contiguous() {
    // Non-contiguous tensor: shape [2, 2] with strides [3, 1]
    let t = Dense::<f64>::from_data_with_strides(
        vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0],
        vec![2, 2],
        vec![3, 1],
        0,
        MemoryOrder::RowMajor,
    );
    let e = t.expand(&[(1, 1), (1, 1)]);
    assert_eq!(e.shape(), &[4, 4]);
    assert_eq!(e.get(&[1, 1]), 1.0);
    assert_eq!(e.get(&[1, 2]), 2.0);
    assert_eq!(e.get(&[2, 1]), 3.0);
    assert_eq!(e.get(&[2, 2]), 4.0);
    assert_eq!(e.get(&[0, 0]), 0.0);
    assert_eq!(e.get(&[3, 3]), 0.0);
}

#[test]
fn test_expand_vs_naive() {
    // Compare against element-by-element verification
    let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
    let t = Dense::from_data_with_order(data, vec![3, 4, 5], MemoryOrder::RowMajor);
    let padding = [(1, 2), (0, 1), (3, 0)];
    let e = t.expand(&padding);

    let new_shape = [3 + 1 + 2, 4 + 0 + 1, 5 + 3 + 0];
    assert_eq!(e.shape(), &new_shape);

    for i0 in 0..new_shape[0] {
        for i1 in 0..new_shape[1] {
            for i2 in 0..new_shape[2] {
                let in_src = i0 >= padding[0].0
                    && i0 < padding[0].0 + 3
                    && i1 >= padding[1].0
                    && i1 < padding[1].0 + 4
                    && i2 >= padding[2].0
                    && i2 < padding[2].0 + 5;
                let expected = if in_src {
                    t.get(&[i0 - padding[0].0, i1 - padding[1].0, i2 - padding[2].0])
                } else {
                    0.0
                };
                assert_eq!(
                    e.get(&[i0, i1, i2]),
                    expected,
                    "mismatch at [{i0},{i1},{i2}]"
                );
            }
        }
    }
}
