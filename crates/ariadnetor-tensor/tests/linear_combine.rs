use ariadnetor_tensor::{DenseTensor, DenseTensorData, MemoryOrder, linear_combine};

/// Wrap a `DenseTensorData<T>` into the joined `DenseTensor<T>` surface.
/// Tests build `DenseTensorData` directly (often with a specific
/// `MemoryOrder` that is not `preferred_order()`) and feed it to the free
/// fn through this wrapper, which preserves the data's order.
fn t<T: Clone>(d: DenseTensorData<T>) -> DenseTensor<T> {
    DenseTensor::from_data(d)
}

#[test]
fn test_linear_combine_basic() {
    let a = DenseTensor::<f64>::filled(vec![2, 2], 1.0);
    let b = DenseTensor::<f64>::filled(vec![2, 2], 2.0);
    let result = linear_combine(&[&a, &b], &[3.0, 4.0]).unwrap();
    // 3*1 + 4*2 = 11
    assert_eq!(result.get([0, 0]), 11.0);
}

#[test]
fn test_linear_combine_shape_mismatch() {
    let a = DenseTensor::<f64>::filled(vec![2, 2], 1.0);
    let b = DenseTensor::<f64>::filled(vec![3, 3], 2.0);
    assert!(linear_combine(&[&a, &b], &[1.0, 1.0]).is_err());
}

#[test]
fn test_linear_combine_empty() {
    let result = linear_combine::<f64>(&[], &[]);
    assert!(result.is_err());
}

#[test]
fn test_linear_combine_length_mismatch() {
    let a = DenseTensor::<f64>::filled(vec![2, 2], 1.0);
    assert!(linear_combine(&[&a], &[1.0, 2.0]).is_err());
}

#[test]
fn test_linear_combine_column_major() {
    let a = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    ));
    let b = t(DenseTensorData::<f64>::from_raw_parts(
        vec![10.0, 30.0, 20.0, 40.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    ));
    let result = linear_combine(&[&a, &b], &[1.0, 0.1]).unwrap();
    // a[0,0]=1 + 0.1*b[0,0]=10 → 2.0
    assert!((result.get([0, 0]) - 2.0).abs() < 1e-10);
    // a[1,1]=4 + 0.1*b[1,1]=40 → 8.0
    assert!((result.get([1, 1]) - 8.0).abs() < 1e-10);
}
