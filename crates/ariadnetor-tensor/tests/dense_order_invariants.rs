//! Invariant tests for `Dense<T>::order()` propagation across operations.
//!
//! Pins the layout-authority contract introduced when `Dense` gained an
//! explicit `order` field: every operation must declare what its output's
//! `order()` is, and consuming ops must enforce or propagate that order
//! consistently. Without these tests, a future refactor could silently
//! revert `order` to the implicit `backend.preferred_order()` model that
//! motivated the original boundary-contract bug.

use arnet_tensor::{Dense, MemoryOrder, normalize_to, reorder};
use std::borrow::Cow;

#[test]
fn dense_new_round_trips_source_order_row_major() {
    let t = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    assert_eq!(t.order(), MemoryOrder::RowMajor);
}

#[test]
fn dense_new_round_trips_source_order_column_major() {
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    assert_eq!(t.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn reorder_outputs_target_order() {
    let rm = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let cm = reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);
    assert_eq!(cm.order(), MemoryOrder::ColumnMajor);
    let back = reorder(&cm, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);
    assert_eq!(back.order(), MemoryOrder::RowMajor);
}

#[test]
fn map_preserves_order_row_major() {
    let t = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let m = t.map(|x| x * 2.0);
    assert_eq!(m.order(), MemoryOrder::RowMajor);
}

#[test]
fn map_preserves_order_column_major() {
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let m = t.map(|x| x * 2.0);
    assert_eq!(m.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn reshape_preserves_order_row_major() {
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let r = t.reshape(vec![6]);
    assert_eq!(r.order(), MemoryOrder::RowMajor);
}

#[test]
fn reshape_preserves_order_column_major() {
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let r = t.reshape(vec![6]);
    assert_eq!(r.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn map_with_index_outputs_iteration_order() {
    // `map_with_index` requires `order == self.order()`; build one
    // tensor per order to verify the output order tag matches the
    // requested iteration order in each case.
    let t_rm = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let row_major_out = t_rm.map_with_index(MemoryOrder::RowMajor, |_idx, &x| x);
    assert_eq!(row_major_out.order(), MemoryOrder::RowMajor);

    let t_cm = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let col_major_out = t_cm.map_with_index(MemoryOrder::ColumnMajor, |_idx, &x| x);
    assert_eq!(col_major_out.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn dense_linear_combine_rejects_mixed_orders() {
    let a = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let result = Dense::linear_combine(&[&a, &b], &[1.0, 1.0]);
    assert!(
        result.is_err(),
        "Dense::linear_combine must reject inputs with mismatched memory order"
    );
}

#[test]
fn normalize_to_borrows_when_order_matches() {
    let t = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let cow = normalize_to(&t, MemoryOrder::RowMajor);
    assert!(
        matches!(cow, Cow::Borrowed(_)),
        "normalize_to must return Cow::Borrowed when source order already matches target"
    );
}

#[test]
fn normalize_to_owns_when_order_differs() {
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let cow = normalize_to(&t, MemoryOrder::RowMajor);
    assert!(
        matches!(cow, Cow::Owned(_)),
        "normalize_to must return Cow::Owned when a reorder is performed"
    );
    assert_eq!(cow.order(), MemoryOrder::RowMajor);
}
