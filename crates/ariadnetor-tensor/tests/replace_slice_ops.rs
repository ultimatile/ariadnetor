//! Tests for replace_slice operations.

use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_replace_slice_column_major() {
    // CM 4x4 zeros, write 2x2 sub at (1, 1)
    // CM sub for [[1,2],[3,4]]: col0=[1,3], col1=[2,4] -> flat [1,3,2,4]
    let mut t = Dense::<f64>::new(vec![0.0; 16], vec![4, 4], MemoryOrder::ColumnMajor);
    let sub = Dense::<f64>::new(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    t.replace_slice(&sub, &[1, 1]);
    // In CM 4x4: col j is at flat[j*4..j*4+4].
    // (1,1) -> flat[1*1 + 4*1] = flat[5]; but CM: flat[col*rows + row] = flat[1*4 + 1] = flat[5]
    assert_eq!(t.data()[5], 1.0); // (1,1)
    assert_eq!(t.data()[6], 3.0); // (2,1)
    assert_eq!(t.data()[9], 2.0); // (1,2)
    assert_eq!(t.data()[10], 4.0); // (2,2)
    assert_eq!(t.data()[0], 0.0); // (0,0) untouched
    assert_eq!(t.data()[15], 0.0); // (3,3) untouched
}

#[test]
fn test_replace_slice_cow() {
    // Verify CoW: cloned tensor should not be affected
    let mut a = Dense::<f64>::new(vec![0.0; 4], vec![2, 2], MemoryOrder::ColumnMajor);
    let b = a.clone();
    let sub = Dense::<f64>::new(vec![1.0], vec![1, 1], MemoryOrder::ColumnMajor);
    a.replace_slice(&sub, &[0, 0]);
    assert_eq!(a.data()[0], 1.0);
    assert_eq!(b.data()[0], 0.0); // b should be unaffected
}
