//! Tests for concatenate/stack optimization paths.

use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_concatenate_column_major_axis0() {
    // Column-major 2×2 tensors concatenated on axis 0
    let a = Dense::<f64>::from_data_with_order(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let b = Dense::<f64>::from_data_with_order(
        vec![5.0, 7.0, 6.0, 8.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let c = Dense::concatenate(&[&a, &b], 0);
    assert_eq!(c.shape(), &[4, 2]);
    assert_eq!(c.memory_order(), MemoryOrder::ColumnMajor);
    // Column-major 4×2: col0=[1,3,5,7], col1=[2,4,6,8]
    assert_eq!(c.data(), &[1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_concatenate_column_major_axis1() {
    // Column-major 2×2 tensors concatenated on axis 1 (outermost for col-major)
    let a = Dense::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let b = Dense::<f64>::from_data_with_order(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let c = Dense::concatenate(&[&a, &b], 1);
    assert_eq!(c.shape(), &[2, 4]);
    assert_eq!(c.memory_order(), MemoryOrder::ColumnMajor);
    // Column-major 2×4: block copy → [a_data, b_data]
    assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_concatenate_non_contiguous() {
    // Truly non-contiguous: shape [2, 2] with strides [3, 1] (gap between rows)
    // data: [1, 2, _, 3, 4, _] → t = [[1,2],[3,4]]
    let t = Dense::<f64>::from_data_with_strides(
        vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0],
        vec![2, 2],
        vec![3, 1],
        0,
        MemoryOrder::RowMajor,
    );
    assert_eq!(t.get(&[0, 0]), 1.0);
    assert_eq!(t.get(&[0, 1]), 2.0);
    assert_eq!(t.get(&[1, 0]), 3.0);
    assert_eq!(t.get(&[1, 1]), 4.0);

    let c = Dense::concatenate(&[&t, &t], 0);
    assert_eq!(c.shape(), &[4, 2]);
    assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_stack_non_contiguous() {
    // Same non-contiguous tensor
    let t = Dense::<f64>::from_data_with_strides(
        vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0],
        vec![2, 2],
        vec![3, 1],
        0,
        MemoryOrder::RowMajor,
    );
    let s = Dense::stack(&[&t, &t], 0);
    assert_eq!(s.shape(), &[2, 2, 2]);
    // axis 0 selects which tensor, rest is the 2×2 data in row-major
    assert_eq!(s.data(), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_concatenate_axis1_vs_naive() {
    // Verify strip-copy path produces correct results for non-outermost axis
    let data_a: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let data_b: Vec<f64> = (100..124).map(|i| i as f64).collect();
    let a = Dense::from_data_with_order(data_a, vec![2, 3, 4], MemoryOrder::RowMajor);
    let b = Dense::from_data_with_order(data_b, vec![2, 3, 4], MemoryOrder::RowMajor);
    let c = Dense::concatenate(&[&a, &b], 1);
    assert_eq!(c.shape(), &[2, 6, 4]);

    // Verify every element against manual indexing
    for i0 in 0..2 {
        for i1 in 0..6 {
            for i2 in 0..4 {
                let expected = if i1 < 3 {
                    a.get(&[i0, i1, i2])
                } else {
                    b.get(&[i0, i1 - 3, i2])
                };
                let actual = c.get(&[i0, i1, i2]);
                assert_eq!(actual, expected, "mismatch at [{i0},{i1},{i2}]");
            }
        }
    }
}

#[test]
fn test_stack_vs_naive() {
    let data_a: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let data_b: Vec<f64> = (100..112).map(|i| i as f64).collect();
    let a = Dense::from_data_with_order(data_a, vec![3, 4], MemoryOrder::RowMajor);
    let b = Dense::from_data_with_order(data_b, vec![3, 4], MemoryOrder::RowMajor);
    let s = Dense::stack(&[&a, &b], 1);
    assert_eq!(s.shape(), &[3, 2, 4]);

    // Verify every element: s[i,j,k] = tensors[j][i,k]
    for i in 0..3 {
        for j in 0..2 {
            for k in 0..4 {
                let expected = if j == 0 {
                    a.get(&[i, k])
                } else {
                    b.get(&[i, k])
                };
                let actual = s.get(&[i, j, k]);
                assert_eq!(actual, expected, "mismatch at [{i},{j},{k}]");
            }
        }
    }
}

#[test]
fn test_concatenate_order_mismatch_strides() {
    // Tensor with column-major strides but RowMajor order field.
    // This is the #94 edge case: contiguous_order() must trust strides, not order.
    let t = Dense::<f64>::from_data_with_strides(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3, 2],
        vec![1, 3],
        0,
        MemoryOrder::RowMajor,
    );
    // Logical: [[1,4],[2,5],[3,6]]
    let c = Dense::concatenate(&[&t, &t], 0);
    assert_eq!(c.shape(), &[6, 2]);
    // Row-major output: [1,4, 2,5, 3,6, 1,4, 2,5, 3,6]
    for i in 0..6 {
        let src_i = i % 3;
        assert_eq!(c.get(&[i, 0]), t.get(&[src_i, 0]), "row {i} col 0");
        assert_eq!(c.get(&[i, 1]), t.get(&[src_i, 1]), "row {i} col 1");
    }
}
