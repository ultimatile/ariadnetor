use std::any::TypeId;

use arnet_core::backend::{BackendError, ComputeBackend};
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;
use num_traits::{Float, NumCast, One, ToPrimitive, Zero};

use crate::contract::contract;
use crate::eigen::eigh;
use crate::scalar_ops::linear_combine;
use crate::solve::solve;

/// Matrix exponential for Hermitian (self-adjoint) matrices via eigendecomposition.
///
/// Computes `exp(A) = V diag(exp(λ)) V†` where `A = V diag(λ) V†` is the
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
/// Matrix exponential with the same shape as the input (n×n).
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn expm_hermitian<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, BackendError> {
    let (w, v) = eigh(backend, tensor, nrow)?;
    let n = w.data().len();

    // V_scaled[i,j] = V[i,j] * exp(λ_j)
    let exp_w: Vec<T::Real> = w.data().iter().map(|&lam| lam.exp()).collect();
    let mut vd_data = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            vd_data[i * n + j] = v.data()[i * n + j].scale_real(exp_w[j]);
        }
    }
    let v_scaled = DenseTensor::from_data(vd_data, vec![n, n]);

    // V†[i,j] = V[j,i].conj()
    let mut vh_data = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            vh_data[i * n + j] = v.data()[j * n + i].conj();
        }
    }
    let v_dagger = DenseTensor::from_data(vh_data, vec![n, n]);

    contract(backend, &v_scaled, &v_dagger, "ij,jk->ik")
}

/// Matrix exponential for anti-Hermitian (skew-adjoint) matrices via eigendecomposition.
///
/// For anti-Hermitian A (where A† = -A), computes `exp(A)` by noting that
/// `H = iA` is Hermitian and using [`eigh`] on H.
///
/// The result satisfies `exp(A) = V diag(exp(-iλ)) V†` where `H = V diag(λ) V†`.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input anti-Hermitian tensor (must be complex type, must reshape to square)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Matrix exponential with the same shape as the input (n×n). The result is unitary.
///
/// # Errors
///
/// Returns `BackendError` if the input is a real type (f32/f64), `nrow` is out of range,
/// the matrix is non-square, or the backend fails.
pub fn expm_antihermitian<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, BackendError> {
    // Reject real types — multiplication by i is not representable
    let tid = TypeId::of::<T>();
    if tid == TypeId::of::<f64>() || tid == TypeId::of::<f32>() {
        return Err(BackendError::InvalidDimension(
            "expm_antihermitian requires complex input type (Complex<f64> or Complex<f32>)"
                .into(),
        ));
    }

    let data = tensor.data();
    let shape = tensor.shape();

    // Compute H = iA: element-wise multiply by imaginary unit
    // i * (a + bi) = -b + ai
    let ia_data: Vec<T> = data
        .iter()
        .map(|&x| T::from_real_imag(-x.im(), x.re()))
        .collect();
    let ia = DenseTensor::from_data(ia_data, shape.to_vec());

    // eigh(iA) → real eigenvalues λ, eigenvectors V
    let (w, v) = eigh(backend, &ia, nrow)?;
    let n = w.data().len();

    // V_scaled[i,j] = V[i,j] * exp(-iλ_j)
    // exp(-iλ) = cos(λ) - i*sin(λ)
    let exp_neg_i_w: Vec<T> = w
        .data()
        .iter()
        .map(|&lam| T::from_real_imag(lam.cos(), -lam.sin()))
        .collect();

    let mut vd_data = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            vd_data[i * n + j] = v.data()[i * n + j] * exp_neg_i_w[j];
        }
    }
    let v_scaled = DenseTensor::from_data(vd_data, vec![n, n]);

    // V†[i,j] = V[j,i].conj()
    let mut vh_data = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            vh_data[i * n + j] = v.data()[j * n + i].conj();
        }
    }
    let v_dagger = DenseTensor::from_data(vh_data, vec![n, n]);

    contract(backend, &v_scaled, &v_dagger, "ij,jk->ik")
}

// ---------------------------------------------------------------------------
// General matrix exponential via Scaling and Squaring + Padé approximation
// ---------------------------------------------------------------------------

/// Identity matrix of size n×n.
fn eye<T: Scalar>(n: usize) -> DenseTensor<T> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one();
    }
    DenseTensor::from_data(data, vec![n, n])
}

/// 1-norm of an n×n matrix (maximum absolute column sum).
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

/// Matrix multiplication helper: C = A * B (both n×n, stored as DenseTensor).
fn matmul<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &DenseTensor<T>,
    b: &DenseTensor<T>,
) -> Result<DenseTensor<T>, BackendError> {
    contract(backend, a, b, "ij,jk->ik")
}

/// Scale each element of a tensor by a real factor.
fn scale_real<T: Scalar>(tensor: &DenseTensor<T>, factor: T::Real) -> DenseTensor<T> {
    let data: Vec<T> = tensor.data().iter().map(|&x| x.scale_real(factor)).collect();
    DenseTensor::from_data(data, tensor.shape().to_vec())
}

/// Padé approximant coefficients b_0..b_m for [m/m] approximant.
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

/// Evaluate Padé [m/m] approximant for small m ∈ {3, 5, 7, 9}.
///
/// Computes U (odd part) and V (even part) such that exp(A) ≈ (V+U)(V-U)^{-1}.
/// Uses Horner-like evaluation to minimize matrix multiplications.
fn pade_uv_small<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &DenseTensor<T>,
    n: usize,
    m: usize,
) -> Result<(DenseTensor<T>, DenseTensor<T>), BackendError> {
    let b = pade_coefficients(m);
    let id = eye::<T>(n);

    // Compute needed powers of A
    let a2 = matmul(backend, a, a)?;

    // Build even polynomial P_even and odd polynomial P_odd
    // V = b_0 I + b_2 A² + b_4 A⁴ + ...
    // U = A(b_1 I + b_3 A² + b_5 A⁴ + ...)
    match m {
        3 => {
            // V = b_0 I + b_2 A²
            // U = A(b_1 I + b_3 A²)
            let v = linear_combine(
                &[&id, &a2],
                &[coeff::<T>(b[0]), coeff::<T>(b[2])],
            )
            .map_err(BackendError::ExecutionFailed)?;
            let u_inner = linear_combine(
                &[&id, &a2],
                &[coeff::<T>(b[1]), coeff::<T>(b[3])],
            )
            .map_err(BackendError::ExecutionFailed)?;
            let u = matmul(backend, a, &u_inner)?;
            Ok((u, v))
        }
        5 => {
            let a4 = matmul(backend, &a2, &a2)?;
            let v = linear_combine(
                &[&id, &a2, &a4],
                &[
                    coeff::<T>(b[0]),
                    coeff::<T>(b[2]),
                    coeff::<T>(b[4]),
                ],
            )
            .map_err(BackendError::ExecutionFailed)?;
            let u_inner = linear_combine(
                &[&id, &a2, &a4],
                &[
                    coeff::<T>(b[1]),
                    coeff::<T>(b[3]),
                    coeff::<T>(b[5]),
                ],
            )
            .map_err(BackendError::ExecutionFailed)?;
            let u = matmul(backend, a, &u_inner)?;
            Ok((u, v))
        }
        7 => {
            let a4 = matmul(backend, &a2, &a2)?;
            let a6 = matmul(backend, &a4, &a2)?;
            let v = linear_combine(
                &[&id, &a2, &a4, &a6],
                &[
                    coeff::<T>(b[0]),
                    coeff::<T>(b[2]),
                    coeff::<T>(b[4]),
                    coeff::<T>(b[6]),
                ],
            )
            .map_err(BackendError::ExecutionFailed)?;
            let u_inner = linear_combine(
                &[&id, &a2, &a4, &a6],
                &[
                    coeff::<T>(b[1]),
                    coeff::<T>(b[3]),
                    coeff::<T>(b[5]),
                    coeff::<T>(b[7]),
                ],
            )
            .map_err(BackendError::ExecutionFailed)?;
            let u = matmul(backend, a, &u_inner)?;
            Ok((u, v))
        }
        9 => {
            let a4 = matmul(backend, &a2, &a2)?;
            let a6 = matmul(backend, &a4, &a2)?;
            let a8 = matmul(backend, &a6, &a2)?;
            let v = linear_combine(
                &[&id, &a2, &a4, &a6, &a8],
                &[
                    coeff::<T>(b[0]),
                    coeff::<T>(b[2]),
                    coeff::<T>(b[4]),
                    coeff::<T>(b[6]),
                    coeff::<T>(b[8]),
                ],
            )
            .map_err(BackendError::ExecutionFailed)?;
            let u_inner = linear_combine(
                &[&id, &a2, &a4, &a6, &a8],
                &[
                    coeff::<T>(b[1]),
                    coeff::<T>(b[3]),
                    coeff::<T>(b[5]),
                    coeff::<T>(b[7]),
                    coeff::<T>(b[9]),
                ],
            )
            .map_err(BackendError::ExecutionFailed)?;
            let u = matmul(backend, a, &u_inner)?;
            Ok((u, v))
        }
        _ => Err(BackendError::ExecutionFailed(format!(
            "pade_uv_small: unsupported order m={m}"
        ))),
    }
}

/// Evaluate Padé [13/13] approximant (Higham 2005, Algorithm 2.3).
///
/// Efficient evaluation using only A², A⁴, A⁶ (3 multiplications for powers,
/// plus 3 for the polynomial evaluation = 6 total).
fn pade_uv_13<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &DenseTensor<T>,
    n: usize,
) -> Result<(DenseTensor<T>, DenseTensor<T>), BackendError> {
    let b = pade_coefficients(13);
    let id = eye::<T>(n);

    let a2 = matmul(backend, a, a)?;
    let a4 = matmul(backend, &a2, &a2)?;
    let a6 = matmul(backend, &a4, &a2)?;

    // W₁ = b₁₃ A⁶ + b₁₁ A⁴ + b₉ A²
    let w1 = linear_combine(
        &[&a6, &a4, &a2],
        &[
            coeff::<T>(b[13]),
            coeff::<T>(b[11]),
            coeff::<T>(b[9]),
        ],
    )
    .map_err(BackendError::ExecutionFailed)?;

    // W₂ = b₇ A⁶ + b₅ A⁴ + b₃ A² + b₁ I
    let w2 = linear_combine(
        &[&a6, &a4, &a2, &id],
        &[
            coeff::<T>(b[7]),
            coeff::<T>(b[5]),
            coeff::<T>(b[3]),
            coeff::<T>(b[1]),
        ],
    )
    .map_err(BackendError::ExecutionFailed)?;

    // U = A (A⁶ W₁ + W₂)
    let a6w1 = matmul(backend, &a6, &w1)?;
    let u_inner = linear_combine(&[&a6w1, &w2], &[T::one(), T::one()])
        .map_err(BackendError::ExecutionFailed)?;
    let u = matmul(backend, a, &u_inner)?;

    // W₃ = b₁₂ A⁶ + b₁₀ A⁴ + b₈ A²
    let w3 = linear_combine(
        &[&a6, &a4, &a2],
        &[
            coeff::<T>(b[12]),
            coeff::<T>(b[10]),
            coeff::<T>(b[8]),
        ],
    )
    .map_err(BackendError::ExecutionFailed)?;

    // W₄ = b₆ A⁶ + b₄ A⁴ + b₂ A² + b₀ I
    let w4 = linear_combine(
        &[&a6, &a4, &a2, &id],
        &[
            coeff::<T>(b[6]),
            coeff::<T>(b[4]),
            coeff::<T>(b[2]),
            coeff::<T>(b[0]),
        ],
    )
    .map_err(BackendError::ExecutionFailed)?;

    // V = A⁶ W₃ + W₄
    let a6w3 = matmul(backend, &a6, &w3)?;
    let v = linear_combine(&[&a6w3, &w4], &[T::one(), T::one()])
        .map_err(BackendError::ExecutionFailed)?;

    Ok((u, v))
}

/// Matrix exponential for general square matrices via scaling and squaring
/// with Padé approximation (Higham 2005).
///
/// Automatically selects the optimal Padé order based on the 1-norm of the
/// input matrix, then applies scaling and squaring for numerical stability.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Matrix exponential with the same shape as the input (n×n).
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn expm<T: Scalar>(
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

    let m_dim: usize = shape[..nrow].iter().product();
    let n_dim: usize = shape[nrow..].iter().product();

    if m_dim != n_dim {
        return Err(BackendError::InvalidDimension(format!(
            "expm requires a square matrix, got {m_dim}×{n_dim}"
        )));
    }

    let n = m_dim;

    // Flatten to n×n for internal computation
    let a = DenseTensor::from_data(tensor.data().to_vec(), vec![n, n]);

    let norm = norm_1::<T>(a.data(), n);

    // Norm thresholds from Higham (2005), Table 2.3
    // Converted to T::Real for comparison
    let theta: [(usize, f64); 5] = [
        (3, 1.495_585_217_958_292e-2),
        (5, 2.539_398_330_063_23e-1),
        (7, 9.504_178_996_162_932e-1),
        (9, 2.097_847_961_257_068),
        (13, 5.371_920_351_148_152),
    ];

    // Try lower-order Padé first (fewer matrix multiplications)
    for &(order, thresh) in &theta[..4] {
        let thresh_real: T::Real = <T::Real as num_traits::NumCast>::from(thresh).unwrap();
        if norm <= thresh_real {
            let (u, v) = pade_uv_small(backend, &a, n, order)?;
            let result = solve_pade(backend, &u, &v)?;
            return Ok(DenseTensor::from_data(
                result.data().to_vec(),
                shape.to_vec(),
            ));
        }
    }

    // Use [13/13] Padé with scaling
    let theta_13: T::Real = <T::Real as num_traits::NumCast>::from(theta[4].1).unwrap();
    let s = if norm > theta_13 {
        let ratio = norm / theta_13;
        ratio.log2().ceil().to_usize().unwrap_or(0)
    } else {
        0
    };

    // Scale: B = A / 2^s
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

    let b = if s > 0 {
        scale_real(&a, scale_factor)
    } else {
        a
    };

    let (u, v) = pade_uv_13(backend, &b, n)?;
    let mut result = solve_pade(backend, &u, &v)?;

    // Repeated squaring: R = R² for s iterations
    for _ in 0..s {
        result = matmul(backend, &result, &result)?;
    }

    Ok(DenseTensor::from_data(
        result.data().to_vec(),
        shape.to_vec(),
    ))
}

/// Solve (V - U) X = V + U for the Padé approximant.
fn solve_pade<T: Scalar>(
    backend: &impl ComputeBackend,
    u: &DenseTensor<T>,
    v: &DenseTensor<T>,
) -> Result<DenseTensor<T>, BackendError> {
    // V - U
    let neg_one: T = coeff::<T>(-1.0);
    let lhs = linear_combine(&[v, u], &[T::one(), neg_one])
        .map_err(BackendError::ExecutionFailed)?;

    // V + U
    let rhs = linear_combine(&[v, u], &[T::one(), T::one()])
        .map_err(BackendError::ExecutionFailed)?;

    // Reshape rhs to n×n for solve (nrow_a=1 since shape is [n, n])
    solve(backend, &lhs, &rhs, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eigen::EighResult;
    use arnet_cpu::CpuBackend;

    #[test]
    fn test_expm_hermitian_diagonal_f64() {
        let backend = CpuBackend::new();

        // exp(diag(1, 2)) = diag(e, e²)
        let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let result = expm_hermitian(&backend, &a, 1).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        let e = std::f64::consts::E;
        assert!((result.get(&[0, 0]) - e).abs() < 1e-10);
        assert!(result.get(&[0, 1]).abs() < 1e-10);
        assert!(result.get(&[1, 0]).abs() < 1e-10);
        assert!((result.get(&[1, 1]) - e * e).abs() < 1e-10);
    }

    #[test]
    fn test_expm_hermitian_zero_f64() {
        let backend = CpuBackend::new();

        // exp(0) = I
        let a = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
        let result = expm_hermitian(&backend, &a, 1).unwrap();

        assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-10);
        assert!(result.get(&[0, 1]).abs() < 1e-10);
        assert!(result.get(&[1, 0]).abs() < 1e-10);
        assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expm_hermitian_identity_f64() {
        let backend = CpuBackend::new();

        // exp(I) = e * I
        let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let result = expm_hermitian(&backend, &a, 1).unwrap();

        let e = std::f64::consts::E;
        assert!((result.get(&[0, 0]) - e).abs() < 1e-10);
        assert!(result.get(&[0, 1]).abs() < 1e-10);
        assert!(result.get(&[1, 0]).abs() < 1e-10);
        assert!((result.get(&[1, 1]) - e).abs() < 1e-10);
    }

    #[test]
    fn test_expm_hermitian_2x2_symmetric() {
        let backend = CpuBackend::new();

        // A = [[0, 1], [1, 0]] (Pauli X), eigenvalues ±1
        // exp(A) = cosh(1)*I + sinh(1)*A
        let a = DenseTensor::<f64>::from_data(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]);
        let result = expm_hermitian(&backend, &a, 1).unwrap();

        let c = 1.0f64.cosh();
        let s = 1.0f64.sinh();
        assert!((result.get(&[0, 0]) - c).abs() < 1e-10);
        assert!((result.get(&[0, 1]) - s).abs() < 1e-10);
        assert!((result.get(&[1, 0]) - s).abs() < 1e-10);
        assert!((result.get(&[1, 1]) - c).abs() < 1e-10);
    }

    #[test]
    fn test_expm_hermitian_c64() {
        use num_complex::Complex;

        let backend = CpuBackend::new();

        // Hermitian: A = [[2, 1-i], [1+i, 3]]
        // eigenvalues from eigh: λ₁ ≈ 1.0, λ₂ ≈ 4.0
        // Verify exp(A) is Hermitian and exp(A) = V diag(exp(λ)) V†
        let a = DenseTensor::from_data(
            vec![
                Complex::new(2.0, 0.0),
                Complex::new(1.0, -1.0),
                Complex::new(1.0, 1.0),
                Complex::new(3.0, 0.0),
            ],
            vec![2, 2],
        );
        let result = expm_hermitian(&backend, &a, 1).unwrap();

        // exp(A) should be Hermitian: result[i,j] = conj(result[j,i])
        let r00 = result.get(&[0, 0]);
        let r01 = result.get(&[0, 1]);
        let r10 = result.get(&[1, 0]);
        let r11 = result.get(&[1, 1]);

        // Diagonal should be real
        assert!(f64::abs(r00.im) < 1e-10, "r00 not real: {r00}");
        assert!(f64::abs(r11.im) < 1e-10, "r11 not real: {r11}");

        // Off-diagonal: r01 = conj(r10)
        assert!((r01 - r10.conj()).norm() < 1e-10, "not Hermitian: r01={r01}, r10={r10}");

        // Verify via eigenvalue comparison: tr(exp(A)) = exp(λ₁) + exp(λ₂)
        let (w, _): EighResult<Complex<f64>> = eigh(&backend, &a, 1).unwrap();
        let expected_trace: f64 = w.data()[0].exp() + w.data()[1].exp();
        let actual_trace: f64 = r00.re + r11.re;
        assert!(
            f64::abs(actual_trace - expected_trace) < 1e-10,
            "trace mismatch: actual={actual_trace}, expected={expected_trace}"
        );
    }

    #[test]
    fn test_expm_hermitian_f32() {
        let backend = CpuBackend::new();

        // exp(diag(1, 2)) = diag(e, e²)
        let a = DenseTensor::<f32>::from_data(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let result = expm_hermitian(&backend, &a, 1).unwrap();

        let e = std::f32::consts::E;
        assert!((result.get(&[0, 0]) - e).abs() < 1e-4);
        assert!((result.get(&[1, 1]) - e * e).abs() < 1e-4);
    }

    #[test]
    fn test_expm_hermitian_invalid_nonsquare() {
        let backend = CpuBackend::new();
        let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert!(expm_hermitian(&backend, &a, 1).is_err());
    }

    #[test]
    fn test_expm_hermitian_invalid_nrow() {
        let backend = CpuBackend::new();
        let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        assert!(expm_hermitian(&backend, &a, 0).is_err());
        assert!(expm_hermitian(&backend, &a, 2).is_err());
    }

    // --- expm_antihermitian tests ---

    #[test]
    fn test_expm_antihermitian_unitarity_c64() {
        use num_complex::Complex;
        use num_traits::Zero;

        let backend = CpuBackend::new();

        // Anti-Hermitian: A = [[0, i], [-i, 0]] = i * σ_x
        // A† = [[0, i], [-i, 0]]† = [[0, i]*, [-i, 0]*]^T = [[0, i], [-i, 0]] ... wait
        // A† = [[0, -(-i)], [i*, 0]] = [[0, i], [-i, 0]]? No.
        // A = [[0, i], [-i, 0]], A† = conj(A)^T = [[0, i], [-i, 0]]
        // That's Hermitian, not anti-Hermitian. Let me fix:
        // Anti-Hermitian: A = [[i, 1], [-1, -i]] → A† = [[-i, -1], [1, i]] = -A ✓
        let a = DenseTensor::from_data(
            vec![
                Complex::new(0.0, 1.0),
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.0),
                Complex::new(0.0, -1.0),
            ],
            vec![2, 2],
        );
        let u = expm_antihermitian(&backend, &a, 1).unwrap();

        // exp(A) should be unitary: U†U = I
        let mut uh_data = vec![Complex::<f64>::zero(); 4];
        for i in 0..2 {
            for j in 0..2 {
                uh_data[i * 2 + j] = u.data()[j * 2 + i].conj();
            }
        }
        let u_dagger = DenseTensor::from_data(uh_data, vec![2, 2]);
        let product = contract(&backend, &u_dagger, &u, "ij,jk->ik").unwrap();

        // Should be identity
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let val = product.get(&[i, j]);
                assert!(
                    (val - Complex::new(expected, 0.0)).norm() < 1e-10,
                    "U†U[{i},{j}] = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_expm_antihermitian_pauli_z() {
        use num_complex::Complex;

        let backend = CpuBackend::new();

        // A = -iσ_z * t = [[-it, 0], [0, it]] which is anti-Hermitian
        // exp(A) = [[exp(-it), 0], [0, exp(it)]]
        let t = 0.5;
        let a = DenseTensor::from_data(
            vec![
                Complex::new(0.0, -t),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, t),
            ],
            vec![2, 2],
        );
        let result = expm_antihermitian(&backend, &a, 1).unwrap();

        // exp(-it) = cos(t) - i*sin(t)
        let exp_neg_it = Complex::new(t.cos(), -t.sin());
        let exp_pos_it = Complex::new(t.cos(), t.sin());

        assert!(
            (result.get(&[0, 0]) - exp_neg_it).norm() < 1e-10,
            "r00 = {}, expected {exp_neg_it}",
            result.get(&[0, 0])
        );
        assert!(result.get(&[0, 1]).norm() < 1e-10);
        assert!(result.get(&[1, 0]).norm() < 1e-10);
        assert!(
            (result.get(&[1, 1]) - exp_pos_it).norm() < 1e-10,
            "r11 = {}, expected {exp_pos_it}",
            result.get(&[1, 1])
        );
    }

    #[test]
    fn test_expm_antihermitian_zero_c64() {
        use num_complex::Complex;
        use num_traits::Zero;

        let backend = CpuBackend::new();

        // exp(0) = I
        let a = DenseTensor::from_data(vec![Complex::<f64>::zero(); 4], vec![2, 2]);
        let result = expm_antihermitian(&backend, &a, 1).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let val = result.get(&[i, j]);
                assert!(
                    (val - Complex::new(expected, 0.0)).norm() < 1e-10,
                    "result[{i},{j}] = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_expm_antihermitian_real_type_error() {
        let backend = CpuBackend::new();

        // Real types should return error
        let a_f64 = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
        assert!(expm_antihermitian(&backend, &a_f64, 1).is_err());

        let a_f32 = DenseTensor::<f32>::from_data(vec![0.0; 4], vec![2, 2]);
        assert!(expm_antihermitian(&backend, &a_f32, 1).is_err());
    }

    #[test]
    fn test_expm_antihermitian_invalid_nonsquare() {
        use num_complex::Complex;
        use num_traits::Zero;

        let backend = CpuBackend::new();
        let a = DenseTensor::from_data(vec![Complex::<f64>::zero(); 6], vec![2, 3]);
        assert!(expm_antihermitian(&backend, &a, 1).is_err());
    }

    // --- expm (general) tests ---

    #[test]
    fn test_expm_diagonal_f64() {
        let backend = CpuBackend::new();

        // exp(diag(1, 2)) = diag(e, e²)
        let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let result = expm(&backend, &a, 1).unwrap();

        let e = std::f64::consts::E;
        assert!((result.get(&[0, 0]) - e).abs() < 1e-10);
        assert!(result.get(&[0, 1]).abs() < 1e-10);
        assert!(result.get(&[1, 0]).abs() < 1e-10);
        assert!((result.get(&[1, 1]) - e * e).abs() < 1e-10);
    }

    #[test]
    fn test_expm_zero_f64() {
        let backend = CpuBackend::new();

        // exp(0) = I
        let a = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
        let result = expm(&backend, &a, 1).unwrap();

        assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-10);
        assert!(result.get(&[0, 1]).abs() < 1e-10);
        assert!(result.get(&[1, 0]).abs() < 1e-10);
        assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expm_identity_f64() {
        let backend = CpuBackend::new();

        // exp(I) = e * I
        let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let result = expm(&backend, &a, 1).unwrap();

        let e = std::f64::consts::E;
        assert!((result.get(&[0, 0]) - e).abs() < 1e-10);
        assert!(result.get(&[0, 1]).abs() < 1e-10);
        assert!(result.get(&[1, 0]).abs() < 1e-10);
        assert!((result.get(&[1, 1]) - e).abs() < 1e-10);
    }

    #[test]
    fn test_expm_nilpotent_f64() {
        let backend = CpuBackend::new();

        // N = [[0, 1], [0, 0]] is nilpotent (N² = 0)
        // exp(N) = I + N = [[1, 1], [0, 1]]
        let a = DenseTensor::<f64>::from_data(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]);
        let result = expm(&backend, &a, 1).unwrap();

        assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-10);
        assert!((result.get(&[0, 1]) - 1.0).abs() < 1e-10);
        assert!(result.get(&[1, 0]).abs() < 1e-10);
        assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expm_general_2x2_f64() {
        let backend = CpuBackend::new();

        // A = [[1, 2], [3, 4]] — compare with eigendecomposition result
        // eigenvalues: λ = (5 ± √33) / 2
        // tr(exp(A)) = exp(λ₁) + exp(λ₂)
        let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = expm(&backend, &a, 1).unwrap();

        let sqrt33 = 33.0f64.sqrt();
        let l1 = (5.0 - sqrt33) / 2.0;
        let l2 = (5.0 + sqrt33) / 2.0;
        let expected_trace = l1.exp() + l2.exp();
        let actual_trace = result.get(&[0, 0]) + result.get(&[1, 1]);
        assert!(
            (actual_trace - expected_trace).abs() < 1e-8,
            "trace mismatch: actual={actual_trace}, expected={expected_trace}"
        );

        // det(exp(A)) = exp(tr(A)) = exp(5)
        let expected_det = 5.0f64.exp();
        let actual_det = result.get(&[0, 0]) * result.get(&[1, 1])
            - result.get(&[0, 1]) * result.get(&[1, 0]);
        assert!(
            (actual_det - expected_det).abs() < 1e-6,
            "det mismatch: actual={actual_det}, expected={expected_det}"
        );
    }

    #[test]
    fn test_expm_general_3x3_f64() {
        let backend = CpuBackend::new();

        // A = [[0,1,0],[0,0,1],[0,0,0]] (upper triangular nilpotent, N³=0)
        // exp(A) = I + A + A²/2 = [[1,1,0.5],[0,1,1],[0,0,1]]
        let a = DenseTensor::<f64>::from_data(
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![3, 3],
        );
        let result = expm(&backend, &a, 1).unwrap();

        assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-10);
        assert!((result.get(&[0, 1]) - 1.0).abs() < 1e-10);
        assert!((result.get(&[0, 2]) - 0.5).abs() < 1e-10);
        assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
        assert!((result.get(&[1, 2]) - 1.0).abs() < 1e-10);
        assert!((result.get(&[2, 2]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expm_complex_f64() {
        use num_complex::Complex;

        let backend = CpuBackend::new();

        // Complex diagonal: exp(diag(i, -i)) = diag(exp(i), exp(-i))
        let a = DenseTensor::from_data(
            vec![
                Complex::new(0.0, 1.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, -1.0),
            ],
            vec![2, 2],
        );
        let result = expm(&backend, &a, 1).unwrap();

        // exp(i) = cos(1) + i*sin(1)
        let exp_i = Complex::new(1.0f64.cos(), 1.0f64.sin());
        let exp_neg_i = Complex::new(1.0f64.cos(), -1.0f64.sin());

        assert!((result.get(&[0, 0]) - exp_i).norm() < 1e-10);
        assert!(result.get(&[0, 1]).norm() < 1e-10);
        assert!(result.get(&[1, 0]).norm() < 1e-10);
        assert!((result.get(&[1, 1]) - exp_neg_i).norm() < 1e-10);
    }

    #[test]
    fn test_expm_large_norm_f64() {
        let backend = CpuBackend::new();

        // A = 10*I — triggers scaling (||A||_1 = 10 > θ_13)
        // exp(10*I) = e^10 * I
        let a = DenseTensor::<f64>::from_data(vec![10.0, 0.0, 0.0, 10.0], vec![2, 2]);
        let result = expm(&backend, &a, 1).unwrap();

        let e10 = 10.0f64.exp();
        assert!(
            (result.get(&[0, 0]) - e10).abs() / e10 < 1e-10,
            "r00 = {}, expected {e10}",
            result.get(&[0, 0])
        );
        assert!(result.get(&[0, 1]).abs() < 1e-5);
        assert!(result.get(&[1, 0]).abs() < 1e-5);
        assert!(
            (result.get(&[1, 1]) - e10).abs() / e10 < 1e-10,
            "r11 = {}, expected {e10}",
            result.get(&[1, 1])
        );
    }

    #[test]
    fn test_expm_f32() {
        let backend = CpuBackend::new();

        let a = DenseTensor::<f32>::from_data(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let result = expm(&backend, &a, 1).unwrap();

        let e = std::f32::consts::E;
        assert!((result.get(&[0, 0]) - e).abs() < 1e-4);
        assert!((result.get(&[1, 1]) - e * e).abs() < 1e-3);
    }

    #[test]
    fn test_expm_invalid_nonsquare() {
        let backend = CpuBackend::new();
        let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert!(expm(&backend, &a, 1).is_err());
    }

    #[test]
    fn test_expm_invalid_nrow() {
        let backend = CpuBackend::new();
        let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        assert!(expm(&backend, &a, 0).is_err());
        assert!(expm(&backend, &a, 2).is_err());
    }
}
