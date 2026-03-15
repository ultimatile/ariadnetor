use arnet_core::backend::{BackendError, ComputeBackend, SolveDescriptor};
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;

/// Solve the linear system AX = B via LU decomposition.
///
/// The input tensor `a` is reshaped as a square matrix by grouping the first
/// `nrow_a` axes as rows and the remaining axes as columns. The tensor `b`
/// must have compatible leading dimension (same number of rows as A).
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `a` - Coefficient tensor (must reshape to n×n square matrix)
/// * `b` - Right-hand side tensor (must have n rows when reshaped)
/// * `nrow_a` - Number of leading axes to group as rows for A
///
/// # Returns
///
/// Solution tensor X with the same shape as B (reshaped as n×nrhs).
///
/// # Errors
///
/// Returns `BackendError` if `nrow_a` is out of range, the matrix A is non-square,
/// dimensions are incompatible, or the backend fails.
pub fn solve<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &DenseTensor<T>,
    b: &DenseTensor<T>,
    nrow_a: usize,
) -> Result<DenseTensor<T>, BackendError> {
    let a_shape = a.shape();
    let a_rank = a.rank();

    if nrow_a == 0 || nrow_a >= a_rank {
        return Err(BackendError::InvalidDimension(format!(
            "nrow_a must be in 1..rank, got nrow_a={nrow_a} for rank={a_rank}"
        )));
    }

    let m: usize = a_shape[..nrow_a].iter().product();
    let n_a: usize = a_shape[nrow_a..].iter().product();

    if m != n_a {
        return Err(BackendError::InvalidDimension(format!(
            "solve requires a square coefficient matrix, got {m}×{n_a}"
        )));
    }

    let n = m;
    let b_total = b.data().len();

    if !b_total.is_multiple_of(n) {
        return Err(BackendError::InvalidDimension(format!(
            "B total elements ({b_total}) must be divisible by n ({n})"
        )));
    }

    let nrhs = b_total / n;

    let mut x_data = vec![T::zero(); n * nrhs];

    let desc = SolveDescriptor {
        n,
        nrhs,
        a: a.data(),
        b: b.data(),
        x: &mut x_data,
    };

    backend.solve(desc)?;

    Ok(DenseTensor::from_data(x_data, b.shape().to_vec()))
}

/// Compute the inverse of a square matrix via LU decomposition.
///
/// Solves `AX = I` using [`solve`] and returns `X = A⁻¹`.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Inverse matrix with the same shape as the input (n×n).
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range, the matrix is non-square,
/// singular, or the backend fails.
pub fn inverse<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, BackendError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(BackendError::InvalidDimension(format!(
            "nrow must be in 1..rank, got nrow={nrow} for rank={rank}"
        )));
    }

    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();

    if m != n {
        return Err(BackendError::InvalidDimension(format!(
            "inverse requires a square matrix, got {m}×{n}"
        )));
    }

    // Identity matrix as right-hand side
    let mut id_data = vec![T::zero(); n * n];
    for i in 0..n {
        id_data[i * n + i] = T::one();
    }
    let identity = DenseTensor::from_data(id_data, vec![n, n]);

    // Solve AX = I → X = A⁻¹
    let a_flat = DenseTensor::from_data(tensor.data().to_vec(), vec![n, n]);
    let result = solve(backend, &a_flat, &identity, 1)?;

    Ok(DenseTensor::from_data(
        result.data().to_vec(),
        shape.to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::contract;
    use arnet_cpu::CpuBackend;

    #[test]
    fn test_solve_f64_2x2() {
        let backend = CpuBackend::new();

        // A = [[2, 1], [5, 3]], B = [[4], [7]]
        // Solution: X = [[5], [-6]]
        let a = DenseTensor::<f64>::from_data(vec![2.0, 1.0, 5.0, 3.0], vec![2, 2]);
        let b = DenseTensor::<f64>::from_data(vec![4.0, 7.0], vec![2, 1]);

        let x = solve(&backend, &a, &b, 1).unwrap();
        assert_eq!(x.shape(), &[2, 1]);
        assert!((x.get(&[0, 0]) - 5.0).abs() < 1e-10);
        assert!((x.get(&[1, 0]) - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_solve_f64_identity() {
        let backend = CpuBackend::new();

        // A = I, B = [[1, 2], [3, 4]] → X = B
        let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let b = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let x = solve(&backend, &a, &b, 1).unwrap();
        assert_eq!(x.shape(), &[2, 2]);
        for i in 0..4 {
            assert!(
                (x.data()[i] - b.data()[i]).abs() < 1e-10,
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_solve_f64_multiple_rhs() {
        let backend = CpuBackend::new();

        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // Verify A * X = B
        let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = DenseTensor::<f64>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let x = solve(&backend, &a, &b, 1).unwrap();
        assert_eq!(x.shape(), &[2, 2]);

        // Verify by computing A * X and comparing with B
        let ax = contract(&backend, &a, &x, "ij,jk->ik").unwrap();
        for i in 0..4 {
            assert!(
                (ax.data()[i] - b.data()[i]).abs() < 1e-10,
                "A*X != B at index {i}"
            );
        }
    }

    #[test]
    fn test_solve_c64() {
        use num_complex::Complex;

        let backend = CpuBackend::new();

        // A = [[1+i, 2], [0, 3-i]], B = [[1], [1]]
        let a = DenseTensor::from_data(
            vec![
                Complex::new(1.0, 1.0),
                Complex::new(2.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(3.0, -1.0),
            ],
            vec![2, 2],
        );
        let b = DenseTensor::from_data(
            vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
            vec![2, 1],
        );

        let x = solve(&backend, &a, &b, 1).unwrap();

        // Verify A * X = B
        let ax = contract(&backend, &a, &x, "ij,jk->ik").unwrap();
        for i in 0..2 {
            let diff = (ax.data()[i] - b.data()[i]).norm();
            assert!(diff < 1e-10, "A*X != B at index {i}, diff={diff}");
        }
    }

    #[test]
    fn test_solve_f32() {
        let backend = CpuBackend::new();

        let a = DenseTensor::<f32>::from_data(vec![2.0, 1.0, 5.0, 3.0], vec![2, 2]);
        let b = DenseTensor::<f32>::from_data(vec![4.0, 7.0], vec![2, 1]);

        let x = solve(&backend, &a, &b, 1).unwrap();
        assert!((x.get(&[0, 0]) - 5.0).abs() < 1e-4);
        assert!((x.get(&[1, 0]) - (-6.0)).abs() < 1e-4);
    }

    #[test]
    fn test_solve_invalid_nonsquare() {
        let backend = CpuBackend::new();

        // 2×3 matrix — not square
        let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = DenseTensor::<f64>::from_data(vec![1.0, 2.0], vec![2, 1]);

        assert!(solve(&backend, &a, &b, 1).is_err());
    }

    #[test]
    fn test_solve_invalid_nrow() {
        let backend = CpuBackend::new();
        let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = DenseTensor::<f64>::from_data(vec![1.0, 2.0], vec![2, 1]);

        assert!(solve(&backend, &a, &b, 0).is_err());
        assert!(solve(&backend, &a, &b, 2).is_err());
    }

    // --- inverse tests ---

    #[test]
    fn test_inverse_f64_2x2() {
        let backend = CpuBackend::new();

        // A = [[2, 1], [5, 3]], det = 1
        // A⁻¹ = [[3, -1], [-5, 2]]
        let a = DenseTensor::<f64>::from_data(vec![2.0, 1.0, 5.0, 3.0], vec![2, 2]);
        let a_inv = inverse(&backend, &a, 1).unwrap();

        assert_eq!(a_inv.shape(), &[2, 2]);
        assert!((a_inv.get(&[0, 0]) - 3.0).abs() < 1e-10);
        assert!((a_inv.get(&[0, 1]) - (-1.0)).abs() < 1e-10);
        assert!((a_inv.get(&[1, 0]) - (-5.0)).abs() < 1e-10);
        assert!((a_inv.get(&[1, 1]) - 2.0).abs() < 1e-10);

        // Verify A * A⁻¹ = I
        let product = contract(&backend, &a, &a_inv, "ij,jk->ik").unwrap();
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
        let backend = CpuBackend::new();

        // inv(diag(2, 5)) = diag(0.5, 0.2)
        let a = DenseTensor::<f64>::from_data(vec![2.0, 0.0, 0.0, 5.0], vec![2, 2]);
        let a_inv = inverse(&backend, &a, 1).unwrap();

        assert!((a_inv.get(&[0, 0]) - 0.5).abs() < 1e-10);
        assert!(a_inv.get(&[0, 1]).abs() < 1e-10);
        assert!(a_inv.get(&[1, 0]).abs() < 1e-10);
        assert!((a_inv.get(&[1, 1]) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_identity() {
        let backend = CpuBackend::new();

        // inv(I) = I
        let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let a_inv = inverse(&backend, &a, 1).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((a_inv.get(&[i, j]) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_inverse_orthogonal() {
        let backend = CpuBackend::new();

        // Rotation matrix: inv(Q) = Q^T
        let angle = std::f64::consts::FRAC_PI_4; // 45 degrees
        let c = angle.cos();
        let s = angle.sin();
        let q = DenseTensor::<f64>::from_data(vec![c, -s, s, c], vec![2, 2]);
        let q_inv = inverse(&backend, &q, 1).unwrap();

        // Q^T = [[c, s], [-s, c]]
        assert!((q_inv.get(&[0, 0]) - c).abs() < 1e-10);
        assert!((q_inv.get(&[0, 1]) - s).abs() < 1e-10);
        assert!((q_inv.get(&[1, 0]) - (-s)).abs() < 1e-10);
        assert!((q_inv.get(&[1, 1]) - c).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_c64() {
        use num_complex::Complex;

        let backend = CpuBackend::new();

        let a = DenseTensor::from_data(
            vec![
                Complex::new(1.0, 1.0),
                Complex::new(2.0, 0.0),
                Complex::new(0.0, 1.0),
                Complex::new(3.0, -1.0),
            ],
            vec![2, 2],
        );
        let a_inv = inverse(&backend, &a, 1).unwrap();

        // Verify A * A⁻¹ = I
        let product = contract(&backend, &a, &a_inv, "ij,jk->ik").unwrap();
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
        let backend = CpuBackend::new();

        let a = DenseTensor::<f32>::from_data(vec![2.0, 1.0, 5.0, 3.0], vec![2, 2]);
        let a_inv = inverse(&backend, &a, 1).unwrap();

        assert!((a_inv.get(&[0, 0]) - 3.0).abs() < 1e-4);
        assert!((a_inv.get(&[0, 1]) - (-1.0)).abs() < 1e-4);
        assert!((a_inv.get(&[1, 0]) - (-5.0)).abs() < 1e-4);
        assert!((a_inv.get(&[1, 1]) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_inverse_invalid_nonsquare() {
        let backend = CpuBackend::new();
        let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert!(inverse(&backend, &a, 1).is_err());
    }

    #[test]
    fn test_inverse_invalid_nrow() {
        let backend = CpuBackend::new();
        let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        assert!(inverse(&backend, &a, 0).is_err());
        assert!(inverse(&backend, &a, 2).is_err());
    }
}
