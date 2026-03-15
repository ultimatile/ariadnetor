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
}
