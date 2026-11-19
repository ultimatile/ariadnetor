//! Tests for expand operations.

mod common;

use ariadnetor_tensor::{DenseTensorData, MemoryOrder};
use common::{ORDERS, multi_indices, offset_within, ramp};

#[test]
fn test_expand_column_major() {
    // CM 2x2 with padding (1,1) on each axis -> 4x4
    // CM data for [[1,2],[3,4]]: col0=[1,3], col1=[2,4] -> flat [1,3,2,4]
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let e = t.expand(&[(1, 1), (1, 1)]);
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
    let t = DenseTensorData::from_raw_parts(data, vec![2, 3, 4], MemoryOrder::RowMajor);
    let e = t.expand(&[(1, 0), (0, 1), (2, 2)]);
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
    let t = DenseTensorData::from_raw_parts(data, vec![3, 4], MemoryOrder::RowMajor);
    let e = t.expand(&[(2, 1), (0, 0)]);
    assert_eq!(e.shape(), &[6, 4]);
    // Rows 0-1 are zeros (8 elements), row 2 starts at index 8
    assert_eq!(e.data()[0], 0.0);
    assert_eq!(e.data()[7], 0.0);
    assert_eq!(e.data()[8], 1.0); // row 2, col 0
    assert_eq!(e.data()[11], 4.0); // row 2, col 3
    assert_eq!(e.data()[19], 12.0); // row 4, col 3
    assert_eq!(e.data()[20], 0.0); // row 5, col 0
}

/// Check `expand` against the elementwise definition: each output element
/// is the source element shifted by the leading padding, or zero outside.
fn check_expand_rank3(shape: [usize; 3], padding: [(usize, usize); 3], order: MemoryOrder) {
    let t = ramp(&shape, order);
    let e = t.expand(&padding);
    let new_shape: Vec<usize> = shape
        .iter()
        .zip(&padding)
        .map(|(&s, &(before, after))| s + before + after)
        .collect();
    assert_eq!(e.shape(), new_shape.as_slice());
    assert_eq!(e.order(), order);
    let before: Vec<usize> = padding.iter().map(|&(b, _)| b).collect();
    for idx in multi_indices(e.shape()) {
        let expected = offset_within(&idx, &before, &shape).map_or(0.0, |src| t.get(&src));
        assert_eq!(e.get(&idx), expected, "{order:?} at {idx:?}");
    }
}

/// Rank-3 expand padding the middle axis: with no padding on the
/// contiguous axis, the strip walk must carry across the middle axis into
/// a wider destination; the contiguous-axis-padded case is checked too.
#[test]
fn test_expand_rank3_matches_elementwise_reference() {
    // The contiguous axis is last for RowMajor and first for ColumnMajor.
    check_expand_rank3([2, 3, 4], [(1, 2), (2, 1), (0, 0)], MemoryOrder::RowMajor);
    check_expand_rank3(
        [4, 3, 2],
        [(0, 0), (2, 1), (1, 2)],
        MemoryOrder::ColumnMajor,
    );
    for order in ORDERS {
        check_expand_rank3([2, 3, 4], [(1, 2), (2, 1), (1, 1)], order);
    }
}

#[test]
fn test_expand_scalar_keeps_value() {
    for order in ORDERS {
        let t = DenseTensorData::from_raw_parts(vec![7.5], vec![], order);
        let e = t.expand(&[]);
        assert_eq!(e.shape(), &[] as &[usize]);
        assert_eq!(e.data(), &[7.5]);
    }
}

#[test]
fn test_expand_empty_source_is_all_padding() {
    for order in ORDERS {
        let t = DenseTensorData::<f64>::from_raw_parts(vec![], vec![0, 2], order);
        let e = t.expand(&[(1, 1), (0, 1)]);
        assert_eq!(e.shape(), &[2, 3]);
        assert_eq!(e.data(), &[0.0; 6]);
    }
}
