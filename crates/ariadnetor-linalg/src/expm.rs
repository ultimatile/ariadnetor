use std::any::TypeId;

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_tensor::{ComputeBackendTensorExt, Dense, DenseTensorData};
use num_traits::{Float, NumCast, One, ToPrimitive, Zero};

use crate::contract::contract_dense;
use crate::eigen::eigh_dense;
use crate::error::LinalgError;
use crate::scalar_ops::{diagonal_scale_dense, linear_combine_dense};
use crate::solve::solve_dense;
use crate::transpose::conjugate_transpose_dense;
use arnet_tensor::reorder;

/// Matrix exponential for Hermitian (self-adjoint) matrices via eigendecomposition.
///
/// Computes `exp(A) = V diag(exp(lambda)) V dagger` where `A = V diag(lambda) V dagger` is the
/// eigendecomposition obtained from [`eigh`].
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input Hermitian tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Matrix exponential with the same shape as the input (n x n).
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn expm_hermitian<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<DenseTensorData<T>, LinalgError> {
    let d = Dense::from_tensor_data(tensor.clone());
    expm_hermitian_dense(backend, &d, nrow).map(|r| r.into_tensor_data())
}

/// Dense-typed sister of [`expm_hermitian`].
pub fn expm_hermitian_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<Dense<T>, LinalgError> {
    let (w, v) = eigh_dense(backend, tensor, nrow)?;

    // V_scaled[i,j] = V[i,j] * exp(lambda_j)
    let exp_w: Vec<T::Real> = w.data().iter().map(|&lam| lam.exp()).collect();
    let v_scaled = diagonal_scale_dense(backend, &v, &exp_w, 1)?;

    // V dagger = conjugate transpose of V
    let v_dagger = conjugate_transpose_dense(backend, &v, &[1, 0])?;

    contract_dense(backend, &v_scaled, &v_dagger, "ij,jk->ik")
}

/// Matrix exponential for anti-Hermitian (skew-adjoint) matrices via eigendecomposition.
///
/// For anti-Hermitian A (where A dagger = -A), computes `exp(A)` by noting that
/// `H = iA` is Hermitian and using [`eigh`] on H.
///
/// The result satisfies `exp(A) = V diag(exp(-i*lambda)) V dagger` where `H = V diag(lambda) V dagger`.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input anti-Hermitian tensor (must be complex type, must reshape to square)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Matrix exponential with the same shape as the input (n x n). The result is unitary.
///
/// # Errors
///
/// Returns `LinalgError` if the input is a real type (f32/f64), `nrow` is out of range,
/// the matrix is non-square, or the backend fails.
pub fn expm_antihermitian<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<DenseTensorData<T>, LinalgError> {
    let d = Dense::from_tensor_data(tensor.clone());
    expm_antihermitian_dense(backend, &d, nrow).map(|r| r.into_tensor_data())
}

/// Dense-typed sister of [`expm_antihermitian`].
pub fn expm_antihermitian_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<Dense<T>, LinalgError> {
    // Reject real types -- multiplication by i is not representable
    let tid = TypeId::of::<T>();
    if tid == TypeId::of::<f64>() || tid == TypeId::of::<f32>() {
        return Err(LinalgError::InvalidArgument(
            "expm_antihermitian requires complex input type (Complex<f64> or Complex<f32>)".into(),
        ));
    }

    let order = backend.preferred_order();

    // Reorder to RowMajor for element-wise iA computation
    let rm = reorder(tensor, tensor.order(), MemoryOrder::RowMajor);
    let data = rm.data();
    let shape = tensor.shape();

    // Compute H = iA: element-wise multiply by imaginary unit
    // i * (a + bi) = -b + ai
    let ia_data: Vec<T> = data
        .iter()
        .map(|&x| T::from_real_imag(-x.im(), x.re()))
        .collect();
    // ia data is in RowMajor; reorder to preferred_order for backend consumption
    let ia_rm = Dense::new(ia_data, shape.to_vec(), MemoryOrder::RowMajor);
    let ia = reorder(&ia_rm, MemoryOrder::RowMajor, order);

    // eigh(iA) -> real eigenvalues lambda, eigenvectors V
    let (w, v_orig) = eigh_dense(backend, &ia, nrow)?;

    // V_scaled[i,j] = V[i,j] * exp(-i*lambda_j)
    // exp(-i*lambda) = cos(lambda) - i*sin(lambda)
    let exp_neg_i_w: Vec<T> = w
        .data()
        .iter()
        .map(|&lam| T::from_real_imag(lam.cos(), -lam.sin()))
        .collect();

    let v_scaled = diagonal_scale_dense(backend, &v_orig, &exp_neg_i_w, 1)?;

    // V dagger = conjugate transpose of V
    let v_dagger = conjugate_transpose_dense(backend, &v_orig, &[1, 0])?;

    contract_dense(backend, &v_scaled, &v_dagger, "ij,jk->ik")
}

// ---------------------------------------------------------------------------
// General matrix exponential via Scaling and Squaring + Pade approximation
// ---------------------------------------------------------------------------

/// 1-norm of an n x n row-major matrix (maximum absolute column sum).
fn norm_1<T: Scalar>(data: &[T], n: usize) -> T::Real {
    let mut max_col_sum = <T::Real as Zero>::zero();
    for j in 0..n {
        let mut col_sum = <T::Real as Zero>::zero();
        for i in 0..n {
            col_sum = col_sum + data[i * n + j].abs();
        }
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }
    max_col_sum
}

/// Matrix multiplication helper: C = A * B (both n x n, stored as Dense).
fn matmul<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &Dense<T>,
    b: &Dense<T>,
) -> Result<Dense<T>, LinalgError> {
    contract_dense(backend, a, b, "ij,jk->ik")
}

/// Validate the `nrow` argument for [`expm`]: must satisfy `1 <= nrow < rank`.
fn validate_expm_nrow(nrow: usize, rank: usize) -> Result<(), LinalgError> {
    if nrow == 0 || nrow >= rank {
        return Err(LinalgError::InvalidArgument(format!(
            "nrow must satisfy 1 <= nrow < rank, got nrow={nrow} for rank={rank}"
        )));
    }
    Ok(())
}

/// Higham (2005) Algorithm 5.1 steps 2-3: pick the smallest `s >= 0`
/// with `norm / 2^s <= theta` and return `B = A / 2^s` along with `s`.
/// `s == 0` returns the input without copying. For `s > 62` the `2^s`
/// factor is built via a doubling loop to stay clear of the `u64` shift
/// limit.
fn compute_scaling_decision<T: Scalar>(
    a: Dense<T>,
    norm: T::Real,
    theta: T::Real,
) -> (Dense<T>, usize) {
    let s = if norm > theta {
        let ratio = norm / theta;
        ratio.log2().ceil().to_usize().unwrap_or(0)
    } else {
        0
    };

    if s == 0 {
        return (a, 0);
    }

    let two_pow_s: T::Real = if s <= 62 {
        <T::Real as NumCast>::from(1u64 << s).unwrap()
    } else {
        let mut v = <T::Real as One>::one();
        for _ in 0..s {
            v = v + v;
        }
        v
    };
    let scale_factor = <T::Real as One>::one() / two_pow_s;
    let b = scale_real(&a, scale_factor);
    (b, s)
}

/// Scale each element of a tensor by a real factor.
fn scale_real<T: Scalar>(tensor: &Dense<T>, factor: T::Real) -> Dense<T> {
    let data: Vec<T> = tensor
        .data()
        .iter()
        .map(|&x| x.scale_real(factor))
        .collect();
    Dense::new(data, tensor.shape().to_vec(), tensor.order())
}

/// Pade approximant coefficients b_0..b_m for [m/m] approximant.
/// b_k = (2m-k)! * m! / ((2m)! * k! * (m-k)!)
fn pade_coefficients(m: usize) -> Vec<f64> {
    let mut b = vec![1.0; m + 1];
    for k in 1..=m {
        b[k] = b[k - 1] * ((m - k + 1) as f64) / ((k * (2 * m - k + 1)) as f64);
    }
    b
}

/// Convert an f64 coefficient to a scalar T via T::Real.
fn coeff<T: Scalar>(c: f64) -> T {
    T::one().scale_real(<T::Real as NumCast>::from(c).unwrap())
}

/// Evaluate Pade [m/m] approximant for small m in {3, 5, 7, 9}.
///
/// Computes U (odd part) and V (even part) such that exp(A) ~ (V+U)(V-U)^{-1}.
/// Uses Horner-like evaluation to minimize matrix multiplications.
fn pade_uv_small<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &Dense<T>,
    n: usize,
    m: usize,
) -> Result<(Dense<T>, Dense<T>), LinalgError> {
    let b = pade_coefficients(m);
    // `id.order()` must match `a.order()` for the `linear_combine`
    // calls below; `backend.eye` produces an identity in
    // `backend.preferred_order()`, which is the order `a` is in by
    // construction in `expm`.
    let id = backend.eye::<T>(n);

    // Compute needed powers of A
    let a2 = matmul(backend, a, a)?;

    // Build even polynomial P_even and odd polynomial P_odd
    // V = b_0 I + b_2 A^2 + b_4 A^4 + ...
    // U = A(b_1 I + b_3 A^2 + b_5 A^4 + ...)
    match m {
        3 => {
            // V = b_0 I + b_2 A^2
            // U = A(b_1 I + b_3 A^2)
            let v = linear_combine_dense(&[&id, &a2], &[coeff::<T>(b[0]), coeff::<T>(b[2])])?;
            let u_inner = linear_combine_dense(&[&id, &a2], &[coeff::<T>(b[1]), coeff::<T>(b[3])])?;
            let u = matmul(backend, a, &u_inner)?;
            Ok((u, v))
        }
        5 => {
            let a4 = matmul(backend, &a2, &a2)?;
            let v = linear_combine_dense(
                &[&id, &a2, &a4],
                &[coeff::<T>(b[0]), coeff::<T>(b[2]), coeff::<T>(b[4])],
            )?;
            let u_inner = linear_combine_dense(
                &[&id, &a2, &a4],
                &[coeff::<T>(b[1]), coeff::<T>(b[3]), coeff::<T>(b[5])],
            )?;
            let u = matmul(backend, a, &u_inner)?;
            Ok((u, v))
        }
        7 => {
            let a4 = matmul(backend, &a2, &a2)?;
            let a6 = matmul(backend, &a4, &a2)?;
            let v = linear_combine_dense(
                &[&id, &a2, &a4, &a6],
                &[
                    coeff::<T>(b[0]),
                    coeff::<T>(b[2]),
                    coeff::<T>(b[4]),
                    coeff::<T>(b[6]),
                ],
            )?;
            let u_inner = linear_combine_dense(
                &[&id, &a2, &a4, &a6],
                &[
                    coeff::<T>(b[1]),
                    coeff::<T>(b[3]),
                    coeff::<T>(b[5]),
                    coeff::<T>(b[7]),
                ],
            )?;
            let u = matmul(backend, a, &u_inner)?;
            Ok((u, v))
        }
        9 => {
            let a4 = matmul(backend, &a2, &a2)?;
            let a6 = matmul(backend, &a4, &a2)?;
            let a8 = matmul(backend, &a6, &a2)?;
            let v = linear_combine_dense(
                &[&id, &a2, &a4, &a6, &a8],
                &[
                    coeff::<T>(b[0]),
                    coeff::<T>(b[2]),
                    coeff::<T>(b[4]),
                    coeff::<T>(b[6]),
                    coeff::<T>(b[8]),
                ],
            )?;
            let u_inner = linear_combine_dense(
                &[&id, &a2, &a4, &a6, &a8],
                &[
                    coeff::<T>(b[1]),
                    coeff::<T>(b[3]),
                    coeff::<T>(b[5]),
                    coeff::<T>(b[7]),
                    coeff::<T>(b[9]),
                ],
            )?;
            let u = matmul(backend, a, &u_inner)?;
            Ok((u, v))
        }
        _ => Err(LinalgError::InvalidArgument(format!(
            "pade_uv_small: unsupported order m={m}"
        ))),
    }
}

/// Evaluate Pade [13/13] approximant (Higham 2005, Algorithm 2.3).
///
/// Efficient evaluation using only A^2, A^4, A^6 (3 multiplications for powers,
/// plus 3 for the polynomial evaluation = 6 total).
fn pade_uv_13<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &Dense<T>,
    n: usize,
) -> Result<(Dense<T>, Dense<T>), LinalgError> {
    let b = pade_coefficients(13);
    // See `pade_uv_small` for the order-matching rationale.
    let id = backend.eye::<T>(n);

    let a2 = matmul(backend, a, a)?;
    let a4 = matmul(backend, &a2, &a2)?;
    let a6 = matmul(backend, &a4, &a2)?;

    // W1 = b13 A^6 + b11 A^4 + b9 A^2
    let w1 = linear_combine_dense(
        &[&a6, &a4, &a2],
        &[coeff::<T>(b[13]), coeff::<T>(b[11]), coeff::<T>(b[9])],
    )?;

    // W2 = b7 A^6 + b5 A^4 + b3 A^2 + b1 I
    let w2 = linear_combine_dense(
        &[&a6, &a4, &a2, &id],
        &[
            coeff::<T>(b[7]),
            coeff::<T>(b[5]),
            coeff::<T>(b[3]),
            coeff::<T>(b[1]),
        ],
    )?;

    // U = A (A^6 W1 + W2)
    let a6w1 = matmul(backend, &a6, &w1)?;
    let u_inner = linear_combine_dense(&[&a6w1, &w2], &[T::one(), T::one()])?;
    let u = matmul(backend, a, &u_inner)?;

    // W3 = b12 A^6 + b10 A^4 + b8 A^2
    let w3 = linear_combine_dense(
        &[&a6, &a4, &a2],
        &[coeff::<T>(b[12]), coeff::<T>(b[10]), coeff::<T>(b[8])],
    )?;

    // W4 = b6 A^6 + b4 A^4 + b2 A^2 + b0 I
    let w4 = linear_combine_dense(
        &[&a6, &a4, &a2, &id],
        &[
            coeff::<T>(b[6]),
            coeff::<T>(b[4]),
            coeff::<T>(b[2]),
            coeff::<T>(b[0]),
        ],
    )?;

    // V = A^6 W3 + W4
    let a6w3 = matmul(backend, &a6, &w3)?;
    let v = linear_combine_dense(&[&a6w3, &w4], &[T::one(), T::one()])?;

    Ok((u, v))
}

/// Matrix exponential for general square matrices via scaling and squaring
/// with Pade approximation (Higham 2005). Auto-selects the Pade order from
/// the 1-norm of the input.
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn expm<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<DenseTensorData<T>, LinalgError> {
    let d = Dense::from_tensor_data(tensor.clone());
    expm_dense(backend, &d, nrow).map(|r| r.into_tensor_data())
}

/// Dense-typed sister of [`expm`].
pub fn expm_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<Dense<T>, LinalgError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    validate_expm_nrow(nrow, rank)?;

    let m_dim: usize = shape[..nrow].iter().product();
    let n_dim: usize = shape[nrow..].iter().product();

    if m_dim != n_dim {
        return Err(LinalgError::InvalidArgument(format!(
            "expm requires a square matrix, got {m_dim}x{n_dim}"
        )));
    }

    let n = m_dim;
    let order = backend.preferred_order();

    // Flatten to n x n row-major for internal computation (norm_1 expects row-major)
    let rm = reorder(tensor, tensor.order(), MemoryOrder::RowMajor);
    // Construct the working matrix in preferred_order for backend operations
    let a = reorder(
        &Dense::new(rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor),
        MemoryOrder::RowMajor,
        order,
    );

    let norm = norm_1::<T>(rm.data(), n);

    // Norm thresholds from Higham (2005), Table 2.3
    // Converted to T::Real for comparison
    let theta: [(usize, f64); 5] = [
        (3, 1.495_585_217_958_292e-2),
        (5, 2.539_398_330_063_23e-1),
        (7, 9.504_178_996_162_932e-1),
        (9, 2.097_847_961_257_068),
        (13, 5.371_920_351_148_152),
    ];

    // Try lower-order Pade first (fewer matrix multiplications)
    for &(pade_order, thresh) in &theta[..4] {
        let thresh_real: T::Real = <T::Real as num_traits::NumCast>::from(thresh).unwrap();
        if norm <= thresh_real {
            let (u, v) = pade_uv_small(backend, &a, n, pade_order)?;
            let result = solve_pade(backend, &u, &v)?;
            // RM intermediate for correct axis-split, then back to preferred order
            let result_rm = reorder(&result, order, MemoryOrder::RowMajor);
            let reshaped = Dense::new(
                result_rm.data().to_vec(),
                shape.to_vec(),
                MemoryOrder::RowMajor,
            );
            return Ok(reorder(&reshaped, MemoryOrder::RowMajor, order));
        }
    }

    // Use [13/13] Pade with scaling: pick s and form B = A / 2^s
    let theta_13: T::Real = <T::Real as num_traits::NumCast>::from(theta[4].1).unwrap();
    let (b, s) = compute_scaling_decision::<T>(a, norm, theta_13);

    let (u, v) = pade_uv_13(backend, &b, n)?;
    let mut result = solve_pade(backend, &u, &v)?;

    // Repeated squaring: R = R^2 for s iterations
    for _ in 0..s {
        result = matmul(backend, &result, &result)?;
    }

    // RM intermediate for correct axis-split, then back to preferred order
    let result_rm = reorder(&result, order, MemoryOrder::RowMajor);
    let reshaped = Dense::new(
        result_rm.data().to_vec(),
        shape.to_vec(),
        MemoryOrder::RowMajor,
    );
    Ok(reorder(&reshaped, MemoryOrder::RowMajor, order))
}

/// Solve (V - U) X = V + U for the Pade approximant.
fn solve_pade<T: Scalar>(
    backend: &impl ComputeBackend,
    u: &Dense<T>,
    v: &Dense<T>,
) -> Result<Dense<T>, LinalgError> {
    // V - U
    let neg_one: T = coeff::<T>(-1.0);
    let lhs = linear_combine_dense(&[v, u], &[T::one(), neg_one])?;

    // V + U
    let rhs = linear_combine_dense(&[v, u], &[T::one(), T::one()])?;

    // Reshape rhs to n x n for solve (nrow_a=1 since shape is [n, n])
    solve_dense(backend, &lhs, &rhs, 1)
}

#[cfg(test)]
mod tests {
    use super::{compute_scaling_decision, validate_expm_nrow};
    use arnet_core::backend::MemoryOrder;
    use arnet_tensor::Dense;

    #[test]
    fn validate_expm_nrow_rejects_zero() {
        assert!(validate_expm_nrow(0, 2).is_err());
    }

    #[test]
    fn validate_expm_nrow_rejects_equal_to_rank() {
        assert!(validate_expm_nrow(2, 2).is_err());
    }

    #[test]
    fn validate_expm_nrow_rejects_greater_than_rank() {
        assert!(validate_expm_nrow(3, 2).is_err());
    }

    #[test]
    fn validate_expm_nrow_accepts_valid() {
        assert!(validate_expm_nrow(1, 3).is_ok());
        assert!(validate_expm_nrow(2, 3).is_ok());
    }

    /// Doubling-loop branch (`s > 62`): kills `v + v -> v - v` (yields 0)
    /// and `v + v -> v * v` (yields 1) mutations on the accumulator.
    #[test]
    fn compute_scaling_decision_above_shift_threshold() {
        let theta = 1.0_f64;
        let norm = (1u64 << 62) as f64 * 1.5;
        let a = Dense::<f64>::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            MemoryOrder::ColumnMajor,
        );
        let (b, s) = compute_scaling_decision::<f64>(a, norm, theta);
        assert_eq!(s, 63);
        let expected_factor = 1.0_f64 / 2.0_f64.powi(63);
        for (i, &x) in b.data().iter().enumerate() {
            let expected = (i + 1) as f64 * expected_factor;
            assert!(
                (x - expected).abs() < 1e-30,
                "b[{i}] = {x}, expected {expected}"
            );
        }
    }

    /// `s == 0` early return: asserts pointer identity to kill the
    /// `== with !=` mutation (which would skip the early return and
    /// allocate a copy whose data matches but whose pointer differs).
    #[test]
    fn compute_scaling_decision_zero_steps_returns_input_unchanged() {
        let a = Dense::<f64>::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            MemoryOrder::ColumnMajor,
        );
        let a_ptr = a.data().as_ptr();
        let (b, s) = compute_scaling_decision::<f64>(a, 0.5, 1.0);
        assert_eq!(s, 0);
        assert_eq!(b.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            b.data().as_ptr(),
            a_ptr,
            "s = 0 path must return input without copying"
        );
    }
}
