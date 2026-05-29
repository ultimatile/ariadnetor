//! Logical `fuse_legs` / `split_leg` on the Dense joined surface.
//!
//! Pins the contract: row-major (C-order) logical grouping that is
//! independent of the tensor's physical memory order, output order
//! equal to input order, backend `Arc` preserved, and the
//! fuse / split inverse relationship. The column-major cases are the
//! discriminating ones — they fail if the implementation degrades to a
//! raw `reshape` (which would leak the physical layout into the
//! logical grouping).

use std::sync::Arc;

use arnet_tensor::{DenseTensor, MemoryOrder, NativeBackend};

/// Logical `[[1,2,3],[4,5,6]]` stored row-major.
fn logical_2x3_row_major() -> DenseTensor<f64> {
    DenseTensor::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
        NativeBackend::shared(),
    )
}

/// Logical `[[1,2,3],[4,5,6]]` stored column-major.
fn logical_2x3_column_major() -> DenseTensor<f64> {
    DenseTensor::from_raw_parts(
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
        NativeBackend::shared(),
    )
}

#[test]
fn split_leg_row_major_logical_grouping() {
    let t = DenseTensor::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![6],
        MemoryOrder::RowMajor,
        NativeBackend::shared(),
    );
    let r = t.split_leg(0, &[2, 3]);

    assert_eq!(r.shape(), &[2, 3]);
    assert_eq!(r.order(), MemoryOrder::RowMajor);
    // Logical [[1,2,3],[4,5,6]] in row-major is the flat input unchanged.
    assert_eq!(r.data_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn split_leg_column_major_logical_grouping() {
    let t = DenseTensor::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![6],
        MemoryOrder::ColumnMajor,
        NativeBackend::shared(),
    );
    let r = t.split_leg(0, &[2, 3]);

    assert_eq!(r.shape(), &[2, 3]);
    // Order is preserved, not normalized to row-major.
    assert_eq!(r.order(), MemoryOrder::ColumnMajor);
    // Logical [[1,2,3],[4,5,6]] in column-major is [1,4,2,5,3,6] —
    // the discriminating check against a raw reshape.
    assert_eq!(r.data_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn fuse_legs_row_major_logical_flatten() {
    let r = logical_2x3_row_major().fuse_legs(0..2);

    assert_eq!(r.shape(), &[6]);
    assert_eq!(r.order(), MemoryOrder::RowMajor);
    assert_eq!(r.data_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn fuse_legs_column_major_logical_flatten() {
    let r = logical_2x3_column_major().fuse_legs(0..2);

    assert_eq!(r.shape(), &[6]);
    assert_eq!(r.order(), MemoryOrder::ColumnMajor);
    // Logical flatten in C-order recovers [1..6] from the column-major
    // buffer [1,4,2,5,3,6].
    assert_eq!(r.data_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn fuse_then_split_round_trips_column_major() {
    // Arbitrary 2x3x4 content; the round-trip identity is content- and
    // order-independent.
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let t = DenseTensor::<f64>::from_raw_parts(
        data.clone(),
        vec![2, 3, 4],
        MemoryOrder::ColumnMajor,
        NativeBackend::shared(),
    );

    let fused = t.fuse_legs(0..2);
    assert_eq!(fused.shape(), &[6, 4]);

    let restored = fused.split_leg(0, &[2, 3]);
    assert_eq!(restored.shape(), &[2, 3, 4]);
    assert_eq!(restored.order(), MemoryOrder::ColumnMajor);
    assert_eq!(restored.data_slice(), data.as_slice());
}

#[test]
fn chained_fuse_matches_single_multi_group_reshape() {
    // Mirrors the apply.rs (w_l, chi_l, d_bra, w_r, chi_r) -> 3D fuse:
    // fuse {0,1} and {3,4}, keep the middle axis.
    let total = 2 * 3 * 2 * 4 * 5;
    let data: Vec<f64> = (0..total).map(|x| x as f64).collect();
    let t = DenseTensor::<f64>::from_raw_parts(
        data,
        vec![2, 3, 2, 4, 5],
        MemoryOrder::ColumnMajor,
        NativeBackend::shared(),
    );

    let fused = t.fuse_legs(0..2).fuse_legs(2..4);

    assert_eq!(fused.shape(), &[6, 2, 20]);
    assert_eq!(fused.order(), MemoryOrder::ColumnMajor);

    // The chained fuse must reproduce the single multi-group reshape it
    // replaces: the explicit row-major sandwich on the same input. Shape
    // and order alone would not catch a wrong logical mapping in the
    // second fuse, so compare the flat data too.
    let reference = t
        .reordered(MemoryOrder::RowMajor)
        .reshape(vec![6, 2, 20])
        .reordered(MemoryOrder::ColumnMajor);
    assert_eq!(fused.data_slice(), reference.data_slice());
}

#[test]
fn fuse_legs_preserves_backend_arc() {
    let t = logical_2x3_row_major();
    let r = t.fuse_legs(0..2);
    assert!(Arc::ptr_eq(t.backend_arc(), r.backend_arc()));
}

#[test]
fn fuse_single_axis_is_logical_identity() {
    // A length-1 fuse range leaves the shape unchanged (the boundary
    // case absorb_from_left hits on rank-2 sites).
    let t = logical_2x3_column_major();
    let r = t.fuse_legs(1..2);
    assert_eq!(r.shape(), &[2, 3]);
    assert_eq!(r.order(), MemoryOrder::ColumnMajor);
    assert_eq!(r.data_slice(), t.data_slice());
}

#[test]
#[should_panic(expected = "fuse_legs")]
fn fuse_legs_panics_on_out_of_range() {
    let _ = logical_2x3_row_major().fuse_legs(0..5);
}

#[test]
#[should_panic(expected = "fuse_legs")]
fn fuse_legs_panics_on_empty_range() {
    let _ = logical_2x3_row_major().fuse_legs(1..1);
}

#[test]
#[should_panic(expected = "split_leg")]
fn split_leg_panics_on_empty_into() {
    let _ = logical_2x3_row_major().split_leg(0, &[]);
}

#[test]
#[should_panic(expected = "split_leg")]
fn split_leg_panics_on_product_mismatch() {
    // Axis 0 has extent 2; [2,2] has product 4.
    let _ = logical_2x3_row_major().split_leg(0, &[2, 2]);
}

#[test]
#[should_panic(expected = "split_leg")]
fn split_leg_panics_on_axis_out_of_range() {
    let _ = logical_2x3_row_major().split_leg(2, &[1]);
}
