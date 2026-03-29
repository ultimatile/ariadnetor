use arnet_linalg::{diag, linear_combine, norm, normalize, scale, trace};
use arnet_tensor::{DenseTensor, MemoryOrder};

// --- Scale tests ---

#[test]
fn test_scale_f64() {
    let tensor = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let scaled = scale(&tensor, 2.5);
    assert_eq!(scaled.get(&[0, 0]), 2.5);
    assert_eq!(scaled.get(&[0, 1]), 5.0);
    assert_eq!(scaled.get(&[1, 0]), 7.5);
    assert_eq!(scaled.get(&[1, 1]), 10.0);
    // Original unchanged
    assert_eq!(tensor.get(&[0, 0]), 1.0);
}

#[test]
fn test_scale_complex() {
    use num_complex::Complex;
    let tensor = DenseTensor::from_data_with_order(
        vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
        vec![2],
        MemoryOrder::RowMajor,
    );
    let scaled = scale(&tensor, Complex::new(2.0, 3.0));
    // (1+0i)*(2+3i) = 2+3i
    assert_eq!(scaled.get(&[0]), Complex::new(2.0, 3.0));
    // (0+1i)*(2+3i) = -3+2i
    assert_eq!(scaled.get(&[1]), Complex::new(-3.0, 2.0));
}

#[test]
fn test_scale_column_major() {
    let tensor = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 3.0, 2.0, 4.0], // ColumnMajor layout: col0=[1,3], col1=[2,4]
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let scaled = scale(&tensor, 2.0);
    assert_eq!(scaled.memory_order(), MemoryOrder::ColumnMajor);
    assert_eq!(scaled.get(&[0, 0]), 2.0);
    assert_eq!(scaled.get(&[0, 1]), 4.0);
    assert_eq!(scaled.get(&[1, 0]), 6.0);
    assert_eq!(scaled.get(&[1, 1]), 8.0);
}

#[test]
fn test_scale_non_contiguous() {
    // Shape [2,2] with stride gap — non-contiguous
    let tensor = DenseTensor::<f64>::from_data_with_strides(
        vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0],
        vec![2, 2],
        vec![4, 1],
        0,
        MemoryOrder::RowMajor,
    );
    let scaled = scale(&tensor, 3.0);
    assert_eq!(scaled.get(&[0, 0]), 3.0);
    assert_eq!(scaled.get(&[0, 1]), 6.0);
    assert_eq!(scaled.get(&[1, 0]), 9.0);
    assert_eq!(scaled.get(&[1, 1]), 12.0);
}

// --- Norm tests ---

#[test]
fn test_norm_f64() {
    let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
    let n = norm(&tensor);
    assert!((n - 2.0).abs() < 1e-10);
}

#[test]
fn test_norm_complex() {
    use num_complex::Complex;
    // |3+4i| = 5, so norm of single element [3+4i] = 5
    let tensor = DenseTensor::from_data_with_order(
        vec![Complex::new(3.0, 4.0)],
        vec![1],
        MemoryOrder::RowMajor,
    );
    let n: f64 = norm(&tensor);
    assert!((n - 5.0).abs() < 1e-10);
}

#[test]
fn test_norm_column_major() {
    let tensor = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    // norm = sqrt(1+4+9+16) = sqrt(30)
    let n = norm(&tensor);
    assert!((n - 30.0_f64.sqrt()).abs() < 1e-10);
}

// --- Normalize tests ---

#[test]
fn test_normalize_f64() {
    let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
    let (normalized, n) = normalize(&tensor);
    assert!((n - 2.0).abs() < 1e-10);
    assert!((norm(&normalized) - 1.0).abs() < 1e-10);
    // Original unchanged
    assert_eq!(tensor.get(&[0, 0]), 1.0);
}

#[test]
#[should_panic(expected = "Cannot normalize zero tensor")]
fn test_normalize_zero_panics() {
    let tensor = DenseTensor::<f64>::zeros(vec![2, 2]);
    let _ = normalize(&tensor);
}

#[test]
fn test_normalize_column_major() {
    let tensor = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let (normalized, n) = normalize(&tensor);
    assert_eq!(normalized.memory_order(), MemoryOrder::ColumnMajor);
    assert!((norm(&normalized) - 1.0).abs() < 1e-10);
    // Verify logical values are preserved (just scaled)
    let expected_scale = 1.0 / n;
    assert!((normalized.get(&[0, 0]) - 1.0 * expected_scale).abs() < 1e-10);
    assert!((normalized.get(&[1, 0]) - 3.0 * expected_scale).abs() < 1e-10);
}

// --- Linear combine tests ---

#[test]
fn test_linear_combine_basic() {
    let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
    let b = DenseTensor::<f64>::constant(vec![2, 2], 2.0);
    let result = linear_combine(&[&a, &b], &[3.0, 4.0]).unwrap();
    // 3*1 + 4*2 = 11
    assert_eq!(result.get(&[0, 0]), 11.0);
}

#[test]
fn test_linear_combine_shape_mismatch() {
    let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
    let b = DenseTensor::<f64>::constant(vec![3, 3], 2.0);
    assert!(linear_combine(&[&a, &b], &[1.0, 1.0]).is_err());
}

#[test]
fn test_linear_combine_empty() {
    let result = linear_combine::<f64>(&[], &[]);
    assert!(result.is_err());
}

#[test]
fn test_linear_combine_length_mismatch() {
    let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
    assert!(linear_combine(&[&a], &[1.0, 2.0]).is_err());
}

#[test]
fn test_linear_combine_column_major() {
    let a = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 3.0, 2.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let b = DenseTensor::<f64>::from_data_with_order(
        vec![10.0, 30.0, 20.0, 40.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let result = linear_combine(&[&a, &b], &[1.0, 0.1]).unwrap();
    assert_eq!(result.memory_order(), MemoryOrder::ColumnMajor);
    // a[0,0]=1 + 0.1*b[0,0]=10 → 2.0
    assert!((result.get(&[0, 0]) - 2.0).abs() < 1e-10);
    // a[1,1]=4 + 0.1*b[1,1]=40 → 8.0
    assert!((result.get(&[1, 1]) - 8.0).abs() < 1e-10);
}

// --- Trace tests ---

#[test]
fn test_trace_matrix() {
    // tr([[1,2],[3,4]]) = 1 + 4 = 5
    let mat = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let result = trace(&mat, &[(0, 1)]).unwrap();
    assert_eq!(result.shape(), &[1]);
    assert_eq!(result.get(&[0]), 5.0);
}

#[test]
fn test_trace_3x3_identity() {
    // tr(I_3) = 3
    let mut data = vec![0.0; 9];
    data[0] = 1.0;
    data[4] = 1.0;
    data[8] = 1.0;
    let mat = DenseTensor::<f64>::from_data_with_order(data, vec![3, 3], MemoryOrder::RowMajor);
    let result = trace(&mat, &[(0, 1)]).unwrap();
    assert_eq!(result.get(&[0]), 3.0);
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
    let tensor = DenseTensor::from_data_with_order(data, vec![2, 3, 3], MemoryOrder::RowMajor);
    let result = trace(&tensor, &[(1, 2)]).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.get(&[0]), 6.0);
    assert_eq!(result.get(&[1]), 15.0);
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

    let tensor = DenseTensor::from_data_with_order(data, shape, MemoryOrder::RowMajor);
    let result = trace(&tensor, &[(1, 3), (2, 4)]).unwrap();
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.get(&[0]), 3.0);
    assert_eq!(result.get(&[1]), 5.0);
    assert_eq!(result.get(&[2]), 0.0);
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
    let tensor = DenseTensor::from_data_with_order(data, shape, MemoryOrder::RowMajor);
    let result = trace(&tensor, &[(0, 2), (1, 3)]).unwrap();
    assert_eq!(result.shape(), &[1]);
    assert_eq!(result.get(&[0]), 6.0);
}

#[test]
fn test_trace_empty_pairs() {
    let tensor = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let result = trace(&tensor, &[]).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.data(), tensor.data());
}

#[test]
fn test_trace_dimension_mismatch() {
    let tensor =
        DenseTensor::<f64>::from_data_with_order(vec![0.0; 6], vec![2, 3], MemoryOrder::RowMajor);
    assert!(trace(&tensor, &[(0, 1)]).is_err());
}

#[test]
fn test_trace_index_out_of_range() {
    let tensor =
        DenseTensor::<f64>::from_data_with_order(vec![0.0; 4], vec![2, 2], MemoryOrder::RowMajor);
    assert!(trace(&tensor, &[(0, 5)]).is_err());
}

#[test]
fn test_trace_self_pair() {
    let tensor =
        DenseTensor::<f64>::from_data_with_order(vec![0.0; 4], vec![2, 2], MemoryOrder::RowMajor);
    assert!(trace(&tensor, &[(1, 1)]).is_err());
}

#[test]
fn test_trace_duplicate_index() {
    let tensor = DenseTensor::<f64>::from_data_with_order(
        vec![0.0; 8],
        vec![2, 2, 2],
        MemoryOrder::RowMajor,
    );
    assert!(trace(&tensor, &[(0, 1), (1, 2)]).is_err());
}

// --- Diag tests ---

#[test]
fn test_diag_extract_3x3() {
    let a = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
        MemoryOrder::RowMajor,
    );
    let d = diag(&a).unwrap();
    assert_eq!(d.shape(), &[3]);
    assert_eq!(d.data(), &[1.0, 5.0, 9.0]);
}

#[test]
fn test_diag_construct_3x3() {
    let v = DenseTensor::<f64>::from_data_with_order(
        vec![2.0, 5.0, 8.0],
        vec![3],
        MemoryOrder::RowMajor,
    );
    let m = diag(&v).unwrap();
    assert_eq!(m.shape(), &[3, 3]);
    assert_eq!(m.data(), &[2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 8.0]);
}

#[test]
fn test_diag_identity() {
    let id = DenseTensor::<f64>::eye(3);
    let d = diag(&id).unwrap();
    assert_eq!(d.data(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_diag_round_trip() {
    let v =
        DenseTensor::<f64>::from_data_with_order(vec![3.0, 7.0], vec![2], MemoryOrder::RowMajor);
    let m = diag(&v).unwrap();
    let v2 = diag(&m).unwrap();
    assert_eq!(v2.data(), v.data());
}

#[test]
fn test_diag_complex() {
    use num_complex::Complex;

    let v = DenseTensor::from_data_with_order(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
        vec![2],
        MemoryOrder::RowMajor,
    );
    let m = diag(&v).unwrap();
    assert_eq!(m.shape(), &[2, 2]);
    assert_eq!(m.get(&[0, 0]), Complex::new(1.0, 2.0));
    assert_eq!(m.get(&[0, 1]), Complex::new(0.0, 0.0));
    assert_eq!(m.get(&[1, 0]), Complex::new(0.0, 0.0));
    assert_eq!(m.get(&[1, 1]), Complex::new(3.0, 4.0));
}

#[test]
fn test_diag_nonsquare_error() {
    let a = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    assert!(diag(&a).is_err());
}

#[test]
fn test_diag_rank3_error() {
    let a = DenseTensor::<f64>::from_data_with_order(
        vec![0.0; 8],
        vec![2, 2, 2],
        MemoryOrder::RowMajor,
    );
    assert!(diag(&a).is_err());
}
