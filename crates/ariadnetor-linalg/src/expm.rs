use std::any::TypeId;

use arnet_core::backend::{BackendError, ComputeBackend};
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;
use num_traits::Float;

use crate::contract::contract;
use crate::eigen::eigh;

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
}
