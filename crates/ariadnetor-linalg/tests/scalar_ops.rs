use arnet_linalg::{diag, trace};
use arnet_native::NativeBackend;
use arnet_tensor::{DenseTensor, DenseTensorData, MemoryOrder};

/// Wrap a `DenseTensorData<T>` into a `DenseTensor<T, NativeBackend>`
/// pinned to the shared `NativeBackend`. Tests build `DenseTensorData`
/// directly (often with a specific `MemoryOrder` that is not
/// `preferred_order()`) and feed it to linalg pub fns through this wrapper.
fn t<T: Clone>(d: DenseTensorData<T>) -> DenseTensor<T, NativeBackend> {
    DenseTensor::with_backend(d, NativeBackend::shared())
}

// --- Trace tests ---

#[test]
fn test_trace_matrix() {
    // tr([[1,2],[3,4]]) = 1 + 4 = 5; the literal is the row-major
    // flat layout `[a, b, c, d]` for `[[a, b], [c, d]]`, so tag the
    // storage `RowMajor` to match (this happens to also be the order
    // `trace` normalizes to internally).
    let mat = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    ));
    let result = trace(&mat, &[(0, 1)]).unwrap();
    assert_eq!(result.shape(), &[1]);
    assert_eq!(result.data_slice()[0], 5.0);
}

#[test]
fn test_trace_3x3_identity() {
    // tr(I_3) = 3
    let mut data = vec![0.0; 9];
    data[0] = 1.0;
    data[4] = 1.0;
    data[8] = 1.0;
    let mat = t(DenseTensorData::from_raw_parts(
        data,
        vec![3, 3],
        MemoryOrder::ColumnMajor,
    ));
    let result = trace(&mat, &[(0, 1)]).unwrap();
    assert_eq!(result.data_slice()[0], 3.0);
}

#[test]
fn test_trace_partial_rank3() {
    // A[i,j,k] shape [2,3,3], trace over (1,2) → B[i] shape [2]
    // B[i] = Σ_j A[i,j,j]
    let mut data = vec![0.0; 18]; // 2*3*3
    // A[0,0,0]=1, A[0,1,1]=2, A[0,2,2]=3 → B[0] = 6
    data[0] = 1.0; // [0,0,0]
    data[4] = 2.0; // [0,1,1]
    data[8] = 3.0; // [0,2,2]
    // A[1,0,0]=4, A[1,1,1]=5, A[1,2,2]=6 → B[1] = 15
    data[9] = 4.0; // [1,0,0]
    data[13] = 5.0; // [1,1,1]
    data[17] = 6.0; // [1,2,2]
    let tensor = t(DenseTensorData::from_raw_parts(
        data,
        vec![2, 3, 3],
        MemoryOrder::RowMajor,
    ));
    let result = trace(&tensor, &[(1, 2)]).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.data_slice()[0], 6.0);
    assert_eq!(result.data_slice()[1], 15.0);
}

#[test]
fn test_trace_tci_example() {
    // TCI spec example: shape {3, 4, 2, 4, 2}, pairs {{1,3}, {2,4}} → shape {3}
    // result[i0] = Σ_{t1,t2} A[i0, t1, t2, t1, t2]
    let shape = vec![3, 4, 2, 4, 2];
    let total: usize = shape.iter().product();
    let mut data = vec![0.0f64; total];

    // Set specific elements to verify correctness
    // A[0, 0, 0, 0, 0] = 1.0
    // A[0, 1, 1, 1, 1] = 2.0
    // result[0] should be 3.0
    let strides = [64, 16, 8, 2, 1]; // row-major strides for shape [3,4,2,4,2]
    // A[0,0,0,0,0]
    data[0] = 1.0;
    // A[0,1,1,1,1]
    data[strides[1] + strides[2] + strides[3] + strides[4]] = 2.0;
    // A[1, 2, 0, 2, 0] = 5.0
    data[strides[0] + 2 * strides[1] + 2 * strides[3]] = 5.0;

    let tensor = t(DenseTensorData::from_raw_parts(
        data,
        shape,
        MemoryOrder::RowMajor,
    ));
    let result = trace(&tensor, &[(1, 3), (2, 4)]).unwrap();
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.data_slice()[0], 3.0);
    assert_eq!(result.data_slice()[1], 5.0);
    assert_eq!(result.data_slice()[2], 0.0);
}

#[test]
fn test_trace_full_contraction() {
    // All bonds paired → scalar
    // A[i,j,i,j] shape [2,3,2,3], pairs [(0,2),(1,3)]
    // result = Σ_{i,j} A[i,j,i,j]
    let shape = vec![2, 3, 2, 3];
    let total: usize = shape.iter().product();
    let mut data = vec![0.0f64; total];
    let strides = [18, 6, 3, 1]; // row-major strides for shape [2,3,2,3]
    // A[0,0,0,0]=1, A[1,1,1,1]=2, A[0,2,0,2]=3
    data[0] = 1.0;
    data[strides[0] + strides[1] + strides[2] + strides[3]] = 2.0;
    data[2 * strides[1] + 2 * strides[3]] = 3.0;
    let tensor = t(DenseTensorData::from_raw_parts(
        data,
        shape,
        MemoryOrder::RowMajor,
    ));
    let result = trace(&tensor, &[(0, 2), (1, 3)]).unwrap();
    assert_eq!(result.shape(), &[1]);
    assert_eq!(result.data_slice()[0], 6.0);
}

#[test]
fn test_trace_empty_pairs() {
    let tensor = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    ));
    let result = trace(&tensor, &[]).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.data_slice(), tensor.data_slice());
}

#[test]
fn test_trace_dimension_mismatch() {
    let tensor = t(DenseTensorData::from_raw_parts(
        vec![0.0; 6],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    ));
    assert!(trace(&tensor, &[(0, 1)]).is_err());
}

#[test]
fn test_trace_index_out_of_range() {
    let tensor = t(DenseTensorData::from_raw_parts(
        vec![0.0; 4],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    ));
    assert!(trace(&tensor, &[(0, 5)]).is_err());
}

#[test]
fn test_trace_self_pair() {
    let tensor = t(DenseTensorData::from_raw_parts(
        vec![0.0; 4],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    ));
    assert!(trace(&tensor, &[(1, 1)]).is_err());
}

#[test]
fn test_trace_duplicate_index() {
    let tensor = t(DenseTensorData::from_raw_parts(
        vec![0.0; 8],
        vec![2, 2, 2],
        MemoryOrder::ColumnMajor,
    ));
    assert!(trace(&tensor, &[(0, 1), (1, 2)]).is_err());
}

// --- Diag tests ---

#[test]
fn test_diag_extract_3x3() {
    let a = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
        MemoryOrder::ColumnMajor,
    ));
    let d = diag(&a).unwrap();
    assert_eq!(d.shape(), &[3]);
    assert_eq!(d.data_slice(), &[1.0, 5.0, 9.0]);
}

#[test]
fn test_diag_construct_3x3() {
    let v = t(DenseTensorData::<f64>::from_raw_parts(
        vec![2.0, 5.0, 8.0],
        vec![3],
        MemoryOrder::ColumnMajor,
    ));
    let m = diag(&v).unwrap();
    assert_eq!(m.shape(), &[3, 3]);
    assert_eq!(
        m.data_slice(),
        &[2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 8.0]
    );
}

#[test]
fn test_diag_identity() {
    let id = DenseTensor::<f64>::eye(3);
    let d = diag(&id).unwrap();
    assert_eq!(d.data_slice(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_diag_round_trip() {
    let v = t(DenseTensorData::<f64>::from_raw_parts(
        vec![3.0, 7.0],
        vec![2],
        MemoryOrder::ColumnMajor,
    ));
    let m = diag(&v).unwrap();
    let v2 = diag(&m).unwrap();
    assert_eq!(v2.data_slice(), v.data_slice());
}

#[test]
fn test_diag_complex() {
    use num_complex::Complex;

    let v = t(DenseTensorData::from_raw_parts(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
        vec![2],
        MemoryOrder::ColumnMajor,
    ));
    let m = diag(&v).unwrap();
    assert_eq!(m.shape(), &[2, 2]);
    assert_eq!(m.get(&[0, 0]), Complex::new(1.0, 2.0));
    assert_eq!(m.get(&[0, 1]), Complex::new(0.0, 0.0));
    assert_eq!(m.get(&[1, 0]), Complex::new(0.0, 0.0));
    assert_eq!(m.get(&[1, 1]), Complex::new(3.0, 4.0));
}

#[test]
fn test_diag_nonsquare_error() {
    let a = t(DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    ));
    assert!(diag(&a).is_err());
}

#[test]
fn test_diag_rank3_error() {
    let a = t(DenseTensorData::from_raw_parts(
        vec![0.0; 8],
        vec![2, 2, 2],
        MemoryOrder::ColumnMajor,
    ));
    assert!(diag(&a).is_err());
}
