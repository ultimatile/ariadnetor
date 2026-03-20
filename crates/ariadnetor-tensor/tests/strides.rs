//! Tests for strides, memory layout, and contiguity

use arnet_tensor::{DenseTensor, MemoryOrder, row_major_strides};

// ============================================================================
// Strides computation
// ============================================================================

#[test]
fn test_row_major_strides_2d() {
    assert_eq!(row_major_strides(&[3, 4]), vec![4, 1]);
}

#[test]
fn test_row_major_strides_3d() {
    assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
}

#[test]
fn test_row_major_strides_1d() {
    assert_eq!(row_major_strides(&[5]), vec![1]);
}

#[test]
fn test_row_major_strides_scalar() {
    assert_eq!(row_major_strides(&[]), Vec::<isize>::new());
}

// ============================================================================
// Layout queries
// ============================================================================

#[test]
fn test_from_data_is_row_major() {
    let t = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert!(t.is_row_major());
    assert!(t.is_contiguous());
    assert!(!t.is_column_major());
    assert_eq!(t.strides(), &[3, 1]);
    assert_eq!(t.offset(), 0);
}

#[test]
fn test_zeros_is_row_major() {
    let t = DenseTensor::<f64>::zeros(vec![3, 4]);
    assert!(t.is_row_major());
    assert!(t.is_contiguous());
}

#[test]
fn test_column_major_tensor() {
    // Create a 2×3 column-major tensor
    // Column-major data: columns stored contiguously
    // Logical: [[1,3,5],[2,4,6]], stored as [1,2,3,4,5,6]
    let t = DenseTensor::from_data_with_strides(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1, 2], // column-major strides
        0,
    );
    assert!(t.is_column_major());
    assert!(t.is_contiguous());
    assert!(!t.is_row_major());

    // Logical element (0,0) = data[0] = 1
    assert_eq!(t.get(&[0, 0]), 1.0);
    // Logical element (1,0) = data[1] = 2
    assert_eq!(t.get(&[1, 0]), 2.0);
    // Logical element (0,1) = data[2] = 3
    assert_eq!(t.get(&[0, 1]), 3.0);
    // Logical element (1,2) = data[5] = 6
    assert_eq!(t.get(&[1, 2]), 6.0);
}

// ============================================================================
// get/set with strides
// ============================================================================

#[test]
fn test_get_set_with_column_major() {
    let mut t = DenseTensor::from_data_with_strides(vec![0.0; 6], vec![2, 3], vec![1, 2], 0);

    t.set(&[0, 0], 1.0);
    t.set(&[1, 0], 2.0);
    t.set(&[0, 1], 3.0);
    t.set(&[1, 1], 4.0);
    t.set(&[0, 2], 5.0);
    t.set(&[1, 2], 6.0);

    assert_eq!(t.get(&[0, 0]), 1.0);
    assert_eq!(t.get(&[1, 0]), 2.0);
    assert_eq!(t.get(&[0, 1]), 3.0);
    assert_eq!(t.get(&[1, 1]), 4.0);
    assert_eq!(t.get(&[0, 2]), 5.0);
    assert_eq!(t.get(&[1, 2]), 6.0);
}

#[test]
fn test_get_with_offset() {
    // Data buffer has extra elements; tensor starts at offset 2
    let t = DenseTensor::from_data_with_strides(
        vec![99.0, 99.0, 1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        vec![2, 1], // row-major strides
        2,          // offset
    );

    assert_eq!(t.get(&[0, 0]), 1.0);
    assert_eq!(t.get(&[0, 1]), 2.0);
    assert_eq!(t.get(&[1, 0]), 3.0);
    assert_eq!(t.get(&[1, 1]), 4.0);
}

// ============================================================================
// to_contiguous
// ============================================================================

#[test]
fn test_to_contiguous_row_major_noop() {
    let t = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let c = t.to_contiguous(MemoryOrder::RowMajor);
    assert!(c.is_row_major());
    assert_eq!(c.data(), t.data());
}

#[test]
fn test_to_contiguous_col_to_row() {
    let t = DenseTensor::from_data_with_strides(
        vec![1.0, 3.0, 2.0, 4.0], // column-major: [[1,2],[3,4]]
        vec![2, 2],
        vec![1, 2],
        0,
    );
    assert!(t.is_column_major());

    let c = t.to_contiguous(MemoryOrder::RowMajor);
    assert!(c.is_row_major());
    // Row-major data for [[1,2],[3,4]] should be [1,2,3,4]
    assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_to_contiguous_row_to_col() {
    let t = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let c = t.to_contiguous(MemoryOrder::ColumnMajor);
    assert!(c.is_column_major());
    // Verify logical values are preserved
    assert_eq!(c.get(&[0, 0]), 1.0);
    assert_eq!(c.get(&[0, 1]), 2.0);
    assert_eq!(c.get(&[1, 0]), 3.0);
    assert_eq!(c.get(&[1, 1]), 4.0);
    // Convert back to row-major to check underlying data order
    let r = c.to_contiguous(MemoryOrder::RowMajor);
    assert_eq!(r.data(), &[1.0, 2.0, 3.0, 4.0]);
}

// ============================================================================
// reshape_view
// ============================================================================

#[test]
fn test_reshape_view_contiguous() {
    let t = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r = t.reshape_view(vec![3, 2]);
    assert!(r.is_some());
    let r = r.unwrap();
    assert_eq!(r.shape(), &[3, 2]);
    assert!(r.is_row_major());
    // Data unchanged: [1,2,3,4,5,6]
    assert_eq!(r.get(&[0, 0]), 1.0);
    assert_eq!(r.get(&[2, 1]), 6.0);
}

#[test]
fn test_reshape_view_column_major() {
    let t = DenseTensor::from_data_with_strides(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1, 2],
        0,
    );
    let r = t.reshape_view(vec![6]);
    assert!(r.is_some());
    let r = r.unwrap();
    assert!(r.is_column_major()); // 1D is both row and column major
    assert_eq!(r.shape(), &[6]);
}

#[test]
fn test_reshape_auto_copies_when_needed() {
    // Create a non-contiguous tensor (arbitrary strides)
    let t = DenseTensor::from_data_with_strides(
        vec![1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0],
        vec![2, 2],
        vec![4, 2], // non-standard strides (skip every other element)
        0,
    );
    assert!(!t.is_contiguous());

    // reshape should copy to contiguous first
    let r = t.reshape(vec![4]);
    assert_eq!(r.shape(), &[4]);
    assert_eq!(r.get(&[0]), 1.0);
    assert_eq!(r.get(&[1]), 2.0);
    assert_eq!(r.get(&[2]), 3.0);
    assert_eq!(r.get(&[3]), 4.0);
}

// ============================================================================
// Existing behavior preserved
// ============================================================================

#[test]
fn test_reshape_preserves_data_order() {
    let t = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r = t.reshape(vec![3, 2]);
    assert_eq!(r.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_map_preserves_logical_order_for_column_major() {
    // Column-major [[1,2],[3,4]] stored as [1,3,2,4]
    let t =
        DenseTensor::from_data_with_strides(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2], vec![1, 2], 0);
    // map(identity) should produce row-major [[1,2],[3,4]] = [1,2,3,4]
    let mapped = t.map(|x| *x);
    assert!(mapped.is_row_major());
    assert_eq!(mapped.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_concatenate_column_major_inputs() {
    // Two column-major 2×1 tensors: [[1],[2]] and [[3],[4]]
    let a = DenseTensor::from_data_with_strides(vec![1.0, 2.0], vec![2, 1], vec![1, 2], 0);
    let b = DenseTensor::from_data_with_strides(vec![3.0, 4.0], vec![2, 1], vec![1, 2], 0);

    let c = DenseTensor::concatenate(&[&a, &b], 1);
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 1.0);
    assert_eq!(c.get(&[0, 1]), 3.0);
    assert_eq!(c.get(&[1, 0]), 2.0);
    assert_eq!(c.get(&[1, 1]), 4.0);
}

#[test]
fn test_from_data_with_strides_bounds_check() {
    // Valid: 2×2 column-major, data length 4
    let _ok =
        DenseTensor::from_data_with_strides(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], vec![1, 2], 0);
}

#[test]
#[should_panic(expected = "reachable index range")]
fn test_from_data_with_strides_out_of_bounds() {
    // Invalid: offset=3 with 2×2 strides [1,2] → max index = 3+1+2 = 6, but data has only 4
    DenseTensor::from_data_with_strides(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], vec![1, 2], 3);
}

#[test]
#[should_panic(expected = "offset")]
fn test_from_data_with_strides_empty_tensor_bad_offset() {
    // Empty tensor (dim 0) but offset past buffer end
    DenseTensor::from_data_with_strides(vec![1.0, 2.0], vec![0, 3], vec![3, 1], 5);
}

#[test]
fn test_slice_with_strides() {
    let t = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    );
    let s = t.slice(&[(0, 2), (1, 3)]);
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.get(&[0, 0]), 2.0);
    assert_eq!(s.get(&[0, 1]), 3.0);
    assert_eq!(s.get(&[1, 0]), 5.0);
    assert_eq!(s.get(&[1, 1]), 6.0);
}
