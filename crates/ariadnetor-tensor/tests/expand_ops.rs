//! Tests for expand operations.

use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_expand_column_major() {
    // CM 2x2 with padding (1,1) on each axis -> 4x4
    // CM data for [[1,2],[3,4]]: col0=[1,3], col1=[2,4] -> flat [1,3,2,4]
    let t = Dense::<f64>::new(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let e = t.expand(&[(1, 1), (1, 1)], MemoryOrder::ColumnMajor);
    assert_eq!(e.shape(), &[4, 4]);
    // Expected CM flat: col0=[0,0,0,0], col1=[0,1,3,0], col2=[0,2,4,0], col3=[0,0,0,0]
    assert_eq!(
        e.data(),
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
    );
}

#[test]
fn test_expand_3d_rm() {
    let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
    let t = Dense::new(data, vec![2, 3, 4], MemoryOrder::ColumnMajor);
    let e = t.expand(&[(1, 0), (0, 1), (2, 2)], MemoryOrder::RowMajor);
    assert_eq!(e.shape(), &[3, 4, 8]);
    // First row of padding: data[0] should be 0
    assert_eq!(e.data()[0], 0.0);
    // Source data starts at RM index [1, 0, 2] = 1*4*8 + 0*8 + 2 = 34
    assert_eq!(e.data()[34], 1.0);
}

#[test]
fn test_expand_no_inner_pad_rm() {
    // No padding on innermost axis -> strip-copy path
    let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let t = Dense::new(data, vec![3, 4], MemoryOrder::ColumnMajor);
    let e = t.expand(&[(2, 1), (0, 0)], MemoryOrder::RowMajor);
    assert_eq!(e.shape(), &[6, 4]);
    // Rows 0-1 are zeros (8 elements), row 2 starts at index 8
    assert_eq!(e.data()[0], 0.0);
    assert_eq!(e.data()[7], 0.0);
    assert_eq!(e.data()[8], 1.0); // row 2, col 0
    assert_eq!(e.data()[11], 4.0); // row 2, col 3
    assert_eq!(e.data()[19], 12.0); // row 4, col 3
    assert_eq!(e.data()[20], 0.0); // row 5, col 0
}
