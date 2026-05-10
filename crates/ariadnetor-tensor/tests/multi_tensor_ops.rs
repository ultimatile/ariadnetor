//! Tests for concatenate and stack operations.

use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_concatenate_column_major_axis0() {
    // CM 2x2 tensors concatenated on axis 0
    let a = Dense::<f64>::new(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let b = Dense::<f64>::new(
        vec![5.0, 7.0, 6.0, 8.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let c = Dense::concatenate(&[&a, &b], 0, MemoryOrder::ColumnMajor);
    assert_eq!(c.shape(), &[4, 2]);
    // CM 4x2: col0=[1,3,5,7], col1=[2,4,6,8]
    assert_eq!(c.data(), &[1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_concatenate_column_major_axis1() {
    // CM 2x2 tensors concatenated on axis 1 (outermost for CM)
    let a = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let b = Dense::<f64>::new(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let c = Dense::concatenate(&[&a, &b], 1, MemoryOrder::ColumnMajor);
    assert_eq!(c.shape(), &[2, 4]);
    // CM 2x4: block copy -> [a_data, b_data]
    assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_concatenate_rm_axis0() {
    let a = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let c = Dense::concatenate(&[&a, &a], 0, MemoryOrder::RowMajor);
    assert_eq!(c.shape(), &[4, 2]);
    assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_stack_rm_axis0() {
    let a = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let s = Dense::stack(&[&a, &a], 0, MemoryOrder::RowMajor);
    assert_eq!(s.shape(), &[2, 2, 2]);
    assert_eq!(s.data(), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
}
