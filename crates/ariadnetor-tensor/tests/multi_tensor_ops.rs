//! Tests for concatenate and stack operations.

mod common;

use ariadnetor_tensor::{DenseTensorData, MemoryOrder};
use common::{ORDERS, multi_indices, offset_within};

#[test]
fn test_concatenate_column_major_axis0() {
    // CM 2x2 tensors concatenated on axis 0
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let b = DenseTensorData::<f64>::from_raw_parts(
        vec![5.0, 7.0, 6.0, 8.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let c = DenseTensorData::concatenate(&[&a, &b], 0);
    assert_eq!(c.shape(), &[4, 2]);
    // CM 4x2: col0=[1,3,5,7], col1=[2,4,6,8]
    assert_eq!(c.data(), &[1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_concatenate_column_major_axis1() {
    // CM 2x2 tensors concatenated on axis 1 (outermost for CM)
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let b = DenseTensorData::<f64>::from_raw_parts(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let c = DenseTensorData::concatenate(&[&a, &b], 1);
    assert_eq!(c.shape(), &[2, 4]);
    // CM 2x4: block copy -> [a_data, b_data]
    assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_concatenate_rm_axis0() {
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let c = DenseTensorData::concatenate(&[&a, &a], 0);
    assert_eq!(c.shape(), &[4, 2]);
    assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_stack_rm_axis0() {
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let s = DenseTensorData::stack(&[&a, &a], 0);
    assert_eq!(s.shape(), &[2, 2, 2]);
    assert_eq!(s.data(), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
}

/// A middle axis is outermost for neither RowMajor nor ColumnMajor, so each
/// source contributes a block of several contiguous strips per outer step.
#[test]
fn test_concatenate_rank3_middle_axis_matches_elementwise_reference() {
    for order in ORDERS {
        let a = DenseTensorData::from_raw_parts(
            (1..=12u32).map(f64::from).collect(),
            vec![2, 2, 3],
            order,
        );
        let b = DenseTensorData::from_raw_parts(
            (1..=6u32).map(|v| -f64::from(v)).collect(),
            vec![2, 1, 3],
            order,
        );
        let c = DenseTensorData::concatenate(&[&a, &b], 1);
        assert_eq!(c.shape(), &[2, 3, 3]);
        assert_eq!(c.order(), order);
        for idx in multi_indices(c.shape()) {
            let expected = offset_within(&idx, &[0, 0, 0], a.shape())
                .map(|r| a.get(&r))
                .or_else(|| offset_within(&idx, &[0, 2, 0], b.shape()).map(|r| b.get(&r)))
                .expect("every output index comes from a or b");
            assert_eq!(c.get(&idx), expected, "{order:?} at {idx:?}");
        }
    }
}
