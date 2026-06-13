//! Logical `fuse_legs` / `split_leg` on the Dense joined surface.
//!
//! Pins the contract: row-major (C-order) logical grouping that is
//! independent of the tensor's physical memory order, output order
//! equal to input order, and the fuse / split inverse relationship. The
//! column-major cases are the discriminating ones — they fail if the
//! implementation degrades to a raw `reshape` (which would leak the
//! physical layout into the logical grouping).

use arnet_tensor::{DenseTensor, MemoryOrder};

/// Logical `[[1,2,3],[4,5,6]]` stored row-major. The public constructor
/// pins to the preferred (column-major) order, so the row-major-tagged
/// tensor is obtained by reordering the column-major one — same logical
/// content, row-major buffer `[1,2,3,4,5,6]`.
fn logical_2x3_row_major() -> DenseTensor<f64> {
    logical_2x3_column_major().reordered(MemoryOrder::RowMajor)
}

/// Logical `[[1,2,3],[4,5,6]]` stored column-major.
fn logical_2x3_column_major() -> DenseTensor<f64> {
    DenseTensor::from_raw_parts(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3])
}

#[test]
fn split_leg_row_major_logical_grouping() {
    // Rank-1 order is metadata-only: the buffer is identical under either
    // order, so `reordered` changes only the logical tag here (it still
    // allocates a fresh buffer, since `from != to`).
    let t = DenseTensor::<f64>::from_raw_parts(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6])
        .reordered(MemoryOrder::RowMajor);
    let r = t.split_leg(0, &[2, 3]);

    assert_eq!(r.shape(), &[2, 3]);
    assert_eq!(r.order(), MemoryOrder::RowMajor);
    // Logical [[1,2,3],[4,5,6]] in row-major is the flat input unchanged.
    assert_eq!(r.data_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn split_leg_column_major_logical_grouping() {
    let t = DenseTensor::<f64>::from_raw_parts(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
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
fn reshape_logical_column_major_regroup() {
    // Logical [[1,2,3],[4,5,6]] (column-major buffer [1,4,2,5,3,6]).
    // reshape_logical to (3, 2) regroups in C-order to [[1,2],[3,4],[5,6]],
    // whose column-major buffer is [1,3,5,2,4,6]. Pins absolute logical
    // correctness (not just equivalence to the chained form) and order
    // preservation for the general escape-hatch path.
    let r = logical_2x3_column_major().reshape_logical(vec![3, 2]);

    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.order(), MemoryOrder::ColumnMajor);
    assert_eq!(r.data_slice(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn fuse_then_split_round_trips_column_major() {
    // Arbitrary 2x3x4 content; the round-trip identity is content- and
    // order-independent.
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let t = DenseTensor::<f64>::from_raw_parts(data.clone(), vec![2, 3, 4]);

    let fused = t.fuse_legs(0..2);
    assert_eq!(fused.shape(), &[6, 4]);

    let restored = fused.split_leg(0, &[2, 3]);
    assert_eq!(restored.shape(), &[2, 3, 4]);
    assert_eq!(restored.order(), MemoryOrder::ColumnMajor);
    assert_eq!(restored.data_slice(), data.as_slice());
}

fn check_reshape_logical_multi_group(order: MemoryOrder) {
    // The apply.rs MPO local fuse regroups (w_l, chi_l, d_bra, w_r, chi_r)
    // into (w_l*chi_l, d_bra, w_r*chi_r) with a single reshape_logical.
    // Two chained single-group fuse_legs must yield the identical result,
    // so the migration from the chained form to the single call preserves
    // behavior. Shape and order alone would miss a wrong logical mapping,
    // so compare the flat data too. The public API promises to preserve
    // the caller's memory order, so both orders are exercised.
    let total = 2 * 3 * 2 * 4 * 5;
    let data: Vec<f64> = (0..total).map(|x| x as f64).collect();
    // `reordered(order)` is a no-op buffer-wise when `order` is already
    // the preferred order; for row-major it flips the tag (and buffer)
    // so both layouts are genuinely exercised.
    let t = DenseTensor::<f64>::from_raw_parts(data, vec![2, 3, 2, 4, 5]).reordered(order);

    let via_reshape = t.reshape_logical(vec![6, 2, 20]);
    let via_chain = t.fuse_legs(0..2).fuse_legs(2..4);

    assert_eq!(via_reshape.shape(), &[6, 2, 20]);
    assert_eq!(via_reshape.order(), order);
    assert_eq!(via_reshape.data_slice(), via_chain.data_slice());
}

#[test]
fn reshape_logical_matches_chained_fuse_multi_group() {
    check_reshape_logical_multi_group(MemoryOrder::ColumnMajor);
    check_reshape_logical_multi_group(MemoryOrder::RowMajor);
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
