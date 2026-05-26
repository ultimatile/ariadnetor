use arnet_linalg::{contract, inverse, solve};
use arnet_native::NativeBackend;
use arnet_tensor::{DenseTensor, DenseTensorData, MemoryOrder};

/// Create Dense from row-major data, converted to column-major for NativeBackend.
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> DenseTensor<T, NativeBackend> {
    let rm = DenseTensorData::from_raw_parts(data, shape, MemoryOrder::RowMajor);
    let cm = arnet_tensor::reorder_data(&rm, MemoryOrder::ColumnMajor);
    DenseTensor::with_backend(cm, NativeBackend::shared())
}

/// Convert column-major Dense back to row-major so `.get()` returns correct values.
fn to_rm<T: Clone>(tensor: &DenseTensor<T, NativeBackend>) -> DenseTensor<T, NativeBackend> {
    let rm = arnet_tensor::reorder_data(tensor.data(), MemoryOrder::RowMajor);
    DenseTensor::with_backend(rm, NativeBackend::shared())
}

#[test]
fn test_solve_f64_2x2() {
    // A = [[2, 1], [5, 3]], B = [[4], [7]]
    // Solution: X = [[5], [-6]]
    let a = cm(vec![2.0_f64, 1.0, 5.0, 3.0], vec![2, 2]);
    let b = cm(vec![4.0_f64, 7.0], vec![2, 1]);

    let x = to_rm(&solve(&a, &b, 1).unwrap());
    assert_eq!(x.shape(), &[2, 1]);
    assert!((x.get(&[0, 0]) - 5.0).abs() < 1e-10);
    assert!((x.get(&[1, 0]) - (-6.0)).abs() < 1e-10);
}

#[test]
fn test_solve_f64_identity() {
    // A = I, B = [[1, 2], [3, 4]] → X = B
    let a = cm(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]);
    let b = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let x = to_rm(&solve(&a, &b, 1).unwrap());
    let b_rm = to_rm(&b);
    assert_eq!(x.shape(), &[2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (x.get(&[i, j]) - b_rm.get(&[i, j])).abs() < 1e-10,
                "mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_solve_f64_multiple_rhs() {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // Verify A * X = B
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let x = solve(&a, &b, 1).unwrap();
    assert_eq!(x.shape(), &[2, 2]);

    // x is in CM (preferred_order); contract expects CM inputs
    let ax = to_rm(&contract(&a, &x, "ij,jk->ik").unwrap());
    let b_rm = to_rm(&b);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (ax.get(&[i, j]) - b_rm.get(&[i, j])).abs() < 1e-10,
                "A*X != B at [{i},{j}]"
            );
        }
    }
}

#[test]
fn test_solve_c64() {
    use num_complex::Complex;

    // A = [[1+i, 2], [0, 3-i]], B = [[1], [1]]
    let a = cm(
        vec![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, -1.0),
        ],
        vec![2, 2],
    );
    let b = DenseTensor::with_backend(
        DenseTensorData::from_raw_parts(
            vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
            vec![2, 1],
            MemoryOrder::ColumnMajor,
        ),
        NativeBackend::shared(),
    );

    let x = solve(&a, &b, 1).unwrap();

    // x is in CM; contract expects CM inputs
    let ax = to_rm(&contract(&a, &x, "ij,jk->ik").unwrap());
    let b_rm = to_rm(&b);
    for i in 0..2 {
        let diff = (ax.get(&[i, 0]) - b_rm.get(&[i, 0])).norm();
        assert!(diff < 1e-10, "A*X != B at [{i},0], diff={diff}");
    }
}

#[test]
fn test_solve_f32() {
    let a = cm(vec![2.0_f32, 1.0, 5.0, 3.0], vec![2, 2]);
    let b = cm(vec![4.0_f32, 7.0], vec![2, 1]);

    let x = to_rm(&solve(&a, &b, 1).unwrap());
    assert!((x.get(&[0, 0]) - 5.0).abs() < 1e-4);
    assert!((x.get(&[1, 0]) - (-6.0)).abs() < 1e-4);
}

#[test]
fn test_solve_invalid_nonsquare() {
    // 2×3 matrix — not square
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = cm(vec![1.0, 2.0], vec![2, 1]);

    assert!(solve(&a, &b, 1).is_err());
}

#[test]
fn test_solve_invalid_nrow() {
    let a = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let b = cm(vec![1.0, 2.0], vec![2, 1]);

    assert!(solve(&a, &b, 0).is_err());
    assert!(solve(&a, &b, 2).is_err());
}

// --- inverse tests ---

#[test]
fn test_inverse_f64_2x2() {
    // A = [[2, 1], [5, 3]], det = 1
    // A⁻¹ = [[3, -1], [-5, 2]]
    let a = cm(vec![2.0_f64, 1.0, 5.0, 3.0], vec![2, 2]);
    let a_inv = to_rm(&inverse(&a, 1).unwrap());

    assert_eq!(a_inv.shape(), &[2, 2]);
    assert!((a_inv.get(&[0, 0]) - 3.0).abs() < 1e-10);
    assert!((a_inv.get(&[0, 1]) - (-1.0)).abs() < 1e-10);
    assert!((a_inv.get(&[1, 0]) - (-5.0)).abs() < 1e-10);
    assert!((a_inv.get(&[1, 1]) - 2.0).abs() < 1e-10);

    // Verify A * A⁻¹ = I. inverse() returns CM.
    let a_inv_cm = inverse(&a, 1).unwrap();
    let product = to_rm(&contract(&a, &a_inv_cm, "ij,jk->ik").unwrap());
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (product.get(&[i, j]) - expected).abs() < 1e-10,
                "A*A⁻¹[{i},{j}] = {}, expected {expected}",
                product.get(&[i, j])
            );
        }
    }
}

#[test]
fn test_inverse_diagonal() {
    // inv(diag(2, 5)) = diag(0.5, 0.2)
    let a = cm(vec![2.0_f64, 0.0, 0.0, 5.0], vec![2, 2]);
    let a_inv = to_rm(&inverse(&a, 1).unwrap());

    assert!((a_inv.get(&[0, 0]) - 0.5).abs() < 1e-10);
    assert!(a_inv.get(&[0, 1]).abs() < 1e-10);
    assert!(a_inv.get(&[1, 0]).abs() < 1e-10);
    assert!((a_inv.get(&[1, 1]) - 0.2).abs() < 1e-10);
}

#[test]
fn test_inverse_identity() {
    // inv(I) = I
    let a = cm(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]);
    let a_inv = to_rm(&inverse(&a, 1).unwrap());

    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((a_inv.get(&[i, j]) - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn test_inverse_orthogonal() {
    // Rotation matrix: inv(Q) = Q^T
    let angle = std::f64::consts::FRAC_PI_4; // 45 degrees
    let c = angle.cos();
    let s = angle.sin();
    let q = cm(vec![c, -s, s, c], vec![2, 2]);
    let q_inv = to_rm(&inverse(&q, 1).unwrap());

    // Q^T = [[c, s], [-s, c]]
    assert!((q_inv.get(&[0, 0]) - c).abs() < 1e-10);
    assert!((q_inv.get(&[0, 1]) - s).abs() < 1e-10);
    assert!((q_inv.get(&[1, 0]) - (-s)).abs() < 1e-10);
    assert!((q_inv.get(&[1, 1]) - c).abs() < 1e-10);
}

#[test]
fn test_inverse_c64() {
    use num_complex::Complex;

    let a = cm(
        vec![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(3.0, -1.0),
        ],
        vec![2, 2],
    );

    // inverse() returns CM; use directly in contract
    let a_inv = inverse(&a, 1).unwrap();
    let product = to_rm(&contract(&a, &a_inv, "ij,jk->ik").unwrap());
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
            let diff = (product.get(&[i, j]) - expected).norm();
            assert!(diff < 1e-10, "A*A⁻¹[{i},{j}] off by {diff}");
        }
    }
}

#[test]
fn test_inverse_f32() {
    let a = cm(vec![2.0_f32, 1.0, 5.0, 3.0], vec![2, 2]);
    let a_inv = to_rm(&inverse(&a, 1).unwrap());

    assert!((a_inv.get(&[0, 0]) - 3.0).abs() < 1e-4);
    assert!((a_inv.get(&[0, 1]) - (-1.0)).abs() < 1e-4);
    assert!((a_inv.get(&[1, 0]) - (-5.0)).abs() < 1e-4);
    assert!((a_inv.get(&[1, 1]) - 2.0).abs() < 1e-4);
}

#[test]
fn test_inverse_invalid_nonsquare() {
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert!(inverse(&a, 1).is_err());
}

#[test]
fn test_inverse_invalid_nrow() {
    let a = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    assert!(inverse(&a, 0).is_err());
    assert!(inverse(&a, 2).is_err());
}
