//! Tests for replace_slice optimization paths.

use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_replace_slice_column_major() {
    // Column-major 4×4 zeros, write 2×2 sub at (1, 1)
    let mut t =
        Dense::<f64>::from_data_with_order(vec![0.0; 16], vec![4, 4], MemoryOrder::ColumnMajor);
    let sub = Dense::<f64>::from_data_with_order(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    t.replace_slice(&sub, &[1, 1]);
    assert_eq!(t.get(&[1, 1]), 1.0);
    assert_eq!(t.get(&[2, 1]), 3.0);
    assert_eq!(t.get(&[1, 2]), 2.0);
    assert_eq!(t.get(&[2, 2]), 4.0);
    // Untouched elements remain zero
    assert_eq!(t.get(&[0, 0]), 0.0);
    assert_eq!(t.get(&[3, 3]), 0.0);
}

#[test]
fn test_replace_slice_non_contiguous_sub() {
    // Non-contiguous sub: shape [2, 2] with strides [3, 1]
    let mut dst =
        Dense::<f64>::from_data_with_order(vec![0.0; 9], vec![3, 3], MemoryOrder::RowMajor);
    let sub = Dense::<f64>::from_data_with_strides(
        vec![10.0, 20.0, 0.0, 30.0, 40.0, 0.0],
        vec![2, 2],
        vec![3, 1],
        0,
        MemoryOrder::RowMajor,
    );
    // sub logical: [[10, 20], [30, 40]]
    dst.replace_slice(&sub, &[0, 1]);
    assert_eq!(dst.get(&[0, 1]), 10.0);
    assert_eq!(dst.get(&[0, 2]), 20.0);
    assert_eq!(dst.get(&[1, 1]), 30.0);
    assert_eq!(dst.get(&[1, 2]), 40.0);
    assert_eq!(dst.get(&[0, 0]), 0.0);
}

#[test]
fn test_replace_slice_vs_naive() {
    // 4×5×3 dst, write 2×3×2 sub at (1, 1, 1)
    let dst_data: Vec<f64> = (0..60).map(|i| i as f64).collect();
    let mut dst =
        Dense::from_data_with_order(dst_data.clone(), vec![4, 5, 3], MemoryOrder::RowMajor);
    let sub_data: Vec<f64> = (100..112).map(|i| i as f64).collect();
    let sub = Dense::from_data_with_order(sub_data, vec![2, 3, 2], MemoryOrder::RowMajor);
    let begin = [1, 1, 1];

    dst.replace_slice(&sub, &begin);

    // Verify: elements inside the replaced region match sub, outside unchanged
    let orig = Dense::from_data_with_order(dst_data, vec![4, 5, 3], MemoryOrder::RowMajor);
    for i0 in 0..4 {
        for i1 in 0..5 {
            for i2 in 0..3 {
                let in_sub = i0 >= begin[0]
                    && i0 < begin[0] + 2
                    && i1 >= begin[1]
                    && i1 < begin[1] + 3
                    && i2 >= begin[2]
                    && i2 < begin[2] + 2;
                let expected = if in_sub {
                    sub.get(&[i0 - begin[0], i1 - begin[1], i2 - begin[2]])
                } else {
                    orig.get(&[i0, i1, i2])
                };
                assert_eq!(
                    dst.get(&[i0, i1, i2]),
                    expected,
                    "mismatch at [{i0},{i1},{i2}]"
                );
            }
        }
    }
}

#[test]
fn test_replace_slice_cow() {
    // Verify CoW: cloned tensor should not be affected
    let mut a = Dense::<f64>::from_data_with_order(vec![0.0; 4], vec![2, 2], MemoryOrder::RowMajor);
    let b = a.clone(); // shares Arc
    let sub = Dense::<f64>::from_data_with_order(vec![1.0], vec![1, 1], MemoryOrder::RowMajor);
    a.replace_slice(&sub, &[0, 0]);
    assert_eq!(a.get(&[0, 0]), 1.0);
    assert_eq!(b.get(&[0, 0]), 0.0); // b should be unaffected
}
