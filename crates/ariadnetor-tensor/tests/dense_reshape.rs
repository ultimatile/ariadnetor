//! Inherent `DenseTensor::reshape` on the joined surface.
//!
//! Pins the contract: zero-copy via `DenseStorage::Clone`, `self.order()`
//! preserved, total-element-count panic surface owned by
//! `TensorData::new`.

use arnet_tensor::{DenseTensor, MemoryOrder};

fn build_2x3_row_major() -> DenseTensor<f64> {
    // The public constructor pins to the preferred (column-major) order, so
    // build the same logical `[[1,2,3],[4,5,6]]` content as the column-major
    // fixture, then `reordered` to the row-major tag this test asserts
    // reshape preserves.
    build_2x3_column_major().reordered(MemoryOrder::RowMajor)
}

fn build_2x3_column_major() -> DenseTensor<f64> {
    DenseTensor::from_raw_parts(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3])
}

#[test]
fn reshape_preserves_order_row_major() {
    let t = build_2x3_row_major();
    let r = t.reshape(vec![6]);

    assert_eq!(r.shape(), &[6]);
    assert_eq!(r.order(), MemoryOrder::RowMajor);
    assert_eq!(r.data_slice(), t.data_slice());
}

#[test]
fn reshape_preserves_order_column_major() {
    let t = build_2x3_column_major();
    let r = t.reshape(vec![6]);

    assert_eq!(r.shape(), &[6]);
    assert_eq!(r.order(), MemoryOrder::ColumnMajor);
    assert_eq!(r.data_slice(), t.data_slice());
}

#[test]
#[should_panic(expected = "storage.flat_len()")]
fn reshape_panics_on_total_mismatch() {
    let t = build_2x3_row_major();
    let _ = t.reshape(vec![5]);
}

#[test]
fn reshape_zero_copy_pointer_identity() {
    let t = build_2x3_row_major();
    let p_before = t.data_slice().as_ptr();
    let r = t.reshape(vec![3, 2]);
    let p_after = r.data_slice().as_ptr();

    assert_eq!(p_before, p_after, "reshape must share the storage Arc");
}
