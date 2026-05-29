//! Inherent `DenseTensor::reshape` on the joined surface.
//!
//! Pins the contract: zero-copy via `DenseStorage::Clone`, `self.order()`
//! and backend `Arc` preserved, total-element-count panic surface owned
//! by `TensorData::new`.

use std::sync::Arc;

use arnet_tensor::{DenseTensor, MemoryOrder, NativeBackend};

fn build_2x3_row_major() -> DenseTensor<f64> {
    // The public constructor pins to the backend's preferred (column-major)
    // order; `reordered` is the only public route to a row-major-tagged
    // tensor, which is exactly the order this test asserts reshape preserves.
    DenseTensor::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        NativeBackend::shared(),
    )
    .reordered(MemoryOrder::RowMajor)
}

fn build_2x3_column_major() -> DenseTensor<f64> {
    DenseTensor::from_raw_parts(
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        vec![2, 3],
        NativeBackend::shared(),
    )
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
fn reshape_preserves_backend_arc() {
    let t = build_2x3_row_major();
    let r = t.reshape(vec![3, 2]);

    assert!(Arc::ptr_eq(t.backend_arc(), r.backend_arc()));
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
