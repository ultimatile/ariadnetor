use arnet_core::backend::{BackendError, ComputeBackend, LqDescriptor, QrDescriptor, SvdDescriptor};
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;
use num_traits::{Float, ToPrimitive, Zero};

/// Result of a thin SVD decomposition: `(U, S, Vt)`.
///
/// - `U`: Left singular vectors
/// - `S`: Singular values (real-valued, descending)
/// - `Vt`: Right singular vectors transposed
pub type SvdResult<T> = (DenseTensor<T>, DenseTensor<<T as Scalar>::Real>, DenseTensor<T>);

/// Result of a truncated SVD decomposition: `(U, S, Vt, trunc_err)`.
///
/// - `U`: Left singular vectors (truncated)
/// - `S`: Singular values (real-valued, descending, truncated)
/// - `Vt`: Right singular vectors transposed (truncated)
/// - `trunc_err`: Truncation error — Frobenius norm of discarded singular values
pub type TruncSvdResult<T> = (
    DenseTensor<T>,
    DenseTensor<<T as Scalar>::Real>,
    DenseTensor<T>,
    <T as Scalar>::Real,
);

/// Parameters for truncated SVD.
///
/// Controls bond dimension via maximum rank (`chi_max`) and/or
/// target truncation error (`target_trunc_err`). When both are set,
/// the stricter (smaller) bound applies.
#[derive(Debug, Clone)]
pub struct TruncSvdParams {
    /// Maximum number of singular values to keep.
    pub chi_max: Option<usize>,
    /// Target truncation error threshold. Singular values are discarded from
    /// the smallest until the Frobenius norm of discarded values would exceed
    /// this threshold.
    pub target_trunc_err: Option<f64>,
}

/// Compute thin SVD of a tensor reshaped as a matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// Returns `(U, S, Vt)` where A ≈ U * diag(S) * Vt.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// * `U` - Left singular vectors, shape `[m, k]` where `m = product(shape[..nrow])`, `k = min(m, n)`
/// * `S` - Singular values (real, descending), shape `[k]`
/// * `Vt` - Right singular vectors transposed, shape `[k, n]` where `n = product(shape[nrow..])`
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range or the backend fails.
pub fn svd<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<SvdResult<T>, BackendError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(BackendError::InvalidDimension(format!(
            "nrow must be in 1..rank, got nrow={nrow} for rank={rank}"
        )));
    }

    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();
    let k = m.min(n);

    let mut u_data = vec![T::zero(); m * k];
    let mut s_data = vec![T::Real::zero(); k];
    let mut vt_data = vec![T::zero(); k * n];

    let desc = SvdDescriptor {
        m,
        n,
        a: tensor.data(),
        u: &mut u_data,
        s: &mut s_data,
        vt: &mut vt_data,
    };

    backend.svd(desc)?;

    let u_tensor = DenseTensor::from_data(u_data, vec![m, k]);
    let s_tensor = DenseTensor::from_data(s_data, vec![k]);
    let vt_tensor = DenseTensor::from_data(vt_data, vec![k, n]);

    Ok((u_tensor, s_tensor, vt_tensor))
}

/// Compute truncated SVD of a tensor reshaped as a matrix.
///
/// Performs a full thin SVD via [`svd`], then truncates singular values
/// according to `params`. The truncation keeps at most `chi_max` singular
/// values, and further discards the smallest values whose cumulative
/// Frobenius norm exceeds `target_trunc_err`.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
/// * `params` - Truncation parameters (`chi_max` and/or `target_trunc_err`)
///
/// # Returns
///
/// * `U` - Left singular vectors, shape `[m, chi]`
/// * `S` - Singular values (real, descending), shape `[chi]`
/// * `Vt` - Right singular vectors transposed, shape `[chi, n]`
/// * `trunc_err` - Frobenius norm of discarded singular values
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range or the backend fails.
pub fn trunc_svd<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<TruncSvdResult<T>, BackendError> {
    let (u_full, s_full, vt_full) = svd(backend, tensor, nrow)?;

    let shape = tensor.shape();
    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();
    let k_full = m.min(n);

    // Determine how many singular values to keep
    let mut chi = k_full;

    // Apply chi_max bound
    if let Some(chi_max) = params.chi_max {
        if chi_max == 0 {
            return Err(BackendError::InvalidDimension(
                "chi_max must be at least 1".into(),
            ));
        }
        chi = chi.min(chi_max);
    }

    // Apply target_trunc_err bound: keep the largest singular values
    // such that the norm of discarded values stays within the threshold
    if let Some(target_err) = params.target_trunc_err {
        // Accumulate discarded norm² from the smallest singular value upward.
        // Compare in f64 to avoid precision issues with the user-specified threshold.
        let target_sq = target_err * target_err;
        let s_data = s_full.data();
        let mut discarded_sq = 0.0_f64;
        let mut chi_err = k_full;
        for i in (0..k_full).rev() {
            let si = s_data[i];
            let si_sq: f64 = (si * si).to_f64().unwrap();
            let new_discarded_sq = discarded_sq + si_sq;
            if new_discarded_sq > target_sq {
                break;
            }
            discarded_sq = new_discarded_sq;
            chi_err = i;
        }
        // Ensure at least one singular value is kept even with aggressive error threshold
        chi = chi.min(chi_err).max(1);
    }

    if chi == k_full {
        // No truncation needed
        return Ok((u_full, s_full, vt_full, T::Real::zero()));
    }

    // Compute truncation error: Frobenius norm of discarded singular values
    let s_data = s_full.data();
    let mut err_sq = T::Real::zero();
    for &si in &s_data[chi..] {
        err_sq = err_sq + si * si;
    }
    let trunc_err = err_sq.sqrt();

    // Truncate U: [m, k_full] → [m, chi]
    let u_data = u_full.data();
    let mut u_trunc = vec![T::zero(); m * chi];
    for i in 0..m {
        u_trunc[i * chi..(i + 1) * chi].copy_from_slice(&u_data[i * k_full..i * k_full + chi]);
    }

    // Truncate S: [k_full] → [chi]
    let s_trunc: Vec<T::Real> = s_data[..chi].to_vec();

    // Truncate Vt: [k_full, n] → [chi, n]
    let vt_data = vt_full.data();
    let vt_trunc: Vec<T> = vt_data[..chi * n].to_vec();

    let u_tensor = DenseTensor::from_data(u_trunc, vec![m, chi]);
    let s_tensor = DenseTensor::from_data(s_trunc, vec![chi]);
    let vt_tensor = DenseTensor::from_data(vt_trunc, vec![chi, n]);

    Ok((u_tensor, s_tensor, vt_tensor, trunc_err))
}

/// Result of a thin QR decomposition: `(Q, R)`.
///
/// - `Q`: Orthogonal/unitary matrix, shape `[m, k]` where `k = min(m, n)`
/// - `R`: Upper triangular matrix, shape `[k, n]`
pub type QrResult<T> = (DenseTensor<T>, DenseTensor<T>);

/// Result of a thin LQ decomposition: `(L, Q)`.
///
/// - `L`: Lower triangular matrix, shape `[m, k]` where `k = min(m, n)`
/// - `Q`: Orthogonal/unitary matrix, shape `[k, n]`
pub type LqResult<T> = (DenseTensor<T>, DenseTensor<T>);

/// Compute thin QR decomposition of a tensor reshaped as a matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// Returns `(Q, R)` where A = Q * R.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// * `Q` - Orthogonal matrix, shape `[m, k]` where `m = product(shape[..nrow])`, `k = min(m, n)`
/// * `R` - Upper triangular matrix, shape `[k, n]` where `n = product(shape[nrow..])`
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range or the backend fails.
pub fn qr<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<QrResult<T>, BackendError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(BackendError::InvalidDimension(format!(
            "nrow must be in 1..rank, got nrow={nrow} for rank={rank}"
        )));
    }

    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();
    let k = m.min(n);

    let mut q_data = vec![T::zero(); m * k];
    let mut r_data = vec![T::zero(); k * n];

    let desc = QrDescriptor {
        m,
        n,
        a: tensor.data(),
        q: &mut q_data,
        r: &mut r_data,
    };

    backend.qr(desc)?;

    let q_tensor = DenseTensor::from_data(q_data, vec![m, k]);
    let r_tensor = DenseTensor::from_data(r_data, vec![k, n]);

    Ok((q_tensor, r_tensor))
}

/// Compute thin LQ decomposition of a tensor reshaped as a matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// Returns `(L, Q)` where A = L * Q.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// * `L` - Lower triangular matrix, shape `[m, k]` where `m = product(shape[..nrow])`, `k = min(m, n)`
/// * `Q` - Orthogonal matrix, shape `[k, n]` where `n = product(shape[nrow..])`
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range or the backend fails.
pub fn lq<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<LqResult<T>, BackendError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(BackendError::InvalidDimension(format!(
            "nrow must be in 1..rank, got nrow={nrow} for rank={rank}"
        )));
    }

    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();
    let k = m.min(n);

    let mut l_data = vec![T::zero(); m * k];
    let mut q_data = vec![T::zero(); k * n];

    let desc = LqDescriptor {
        m,
        n,
        a: tensor.data(),
        l: &mut l_data,
        q: &mut q_data,
    };

    backend.lq(desc)?;

    let l_tensor = DenseTensor::from_data(l_data, vec![m, k]);
    let q_tensor = DenseTensor::from_data(q_data, vec![k, n]);

    Ok((l_tensor, q_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arnet_cpu::CpuBackend;

    // --- SVD tests ---

    #[test]
    fn test_svd_f64_2d() {
        let backend = CpuBackend::new();
        // A = [[1, 2], [3, 4]] shape [2, 2], nrow=1 → 2×2 matrix
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();

        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 2]);

        // Singular values should be positive and descending
        assert!(s.get(&[0]) > s.get(&[1]));
        assert!(s.get(&[1]) >= 0.0);

        // Reconstruct: A ≈ U * diag(S) * Vt
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += u.get(&[i, k]) * s.get(&[k]) * vt.get(&[k, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-10,
                    "Reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_svd_f64_rectangular() {
        let backend = CpuBackend::new();
        // shape [2, 3], nrow=1 → 2×3 matrix, k=2
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );

        let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();

        let (m, n, k) = (2, 3, 2);
        assert_eq!(u.shape(), &[m, k]);
        assert_eq!(s.shape(), &[k]);
        assert_eq!(vt.shape(), &[k, n]);

        // Reconstruct
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += u.get(&[i, l]) * s.get(&[l]) * vt.get(&[l, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-10,
                    "Reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_svd_f64_higher_rank() {
        let backend = CpuBackend::new();
        // shape [2, 3, 4], nrow=2 → m=6, n=4, k=4
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

        let (u, s, vt) = svd(&backend, &tensor, 2).unwrap();

        let (m, n, k) = (6, 4, 4);
        assert_eq!(u.shape(), &[m, k]);
        assert_eq!(s.shape(), &[k]);
        assert_eq!(vt.shape(), &[k, n]);

        // Reconstruct and verify against original flat data
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += u.get(&[i, l]) * s.get(&[l]) * vt.get(&[l, j]);
                }
                let orig = tensor.data()[i * n + j];
                assert!(
                    (val - orig).abs() < 1e-9,
                    "Reconstruction mismatch at ({i},{j}): {val} vs {orig}"
                );
            }
        }
    }

    #[test]
    fn test_svd_f32() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();

        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 2]);

        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0f32;
                for k in 0..2 {
                    val += u.get(&[i, k]) * s.get(&[k]) * vt.get(&[k, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-4,
                    "Reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_svd_invalid_nrow() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // nrow=0 is invalid
        assert!(svd(&backend, &tensor, 0).is_err());
        // nrow=rank is invalid
        assert!(svd(&backend, &tensor, 2).is_err());
    }

    // --- Truncated SVD tests ---

    #[test]
    fn test_trunc_svd_chi_max() {
        let backend = CpuBackend::new();
        // 3×4 matrix with rank > 1
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 4],
        );

        let params = TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        };
        let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

        // Truncated to chi=2
        assert_eq!(u.shape(), &[3, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 4]);

        // Singular values should be positive and descending
        assert!(s.get(&[0]) > s.get(&[1]));
        assert!(s.get(&[1]) > 0.0);

        // Truncation error should be positive (we discarded one singular value)
        assert!(trunc_err > 0.0);

        // Verify truncation error equals the discarded singular value
        let (_, s_full, _, _) = trunc_svd(
            &backend,
            &tensor,
            1,
            &TruncSvdParams {
                chi_max: None,
                target_trunc_err: None,
            },
        )
        .unwrap();
        let expected_err = s_full.get(&[2]);
        assert!(
            (trunc_err - expected_err).abs() < 1e-10,
            "trunc_err={trunc_err} vs expected={expected_err}"
        );
    }

    #[test]
    fn test_trunc_svd_chi_max_zero_is_error() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let params = TruncSvdParams {
            chi_max: Some(0),
            target_trunc_err: None,
        };
        assert!(trunc_svd(&backend, &tensor, 1, &params).is_err());
    }

    #[test]
    fn test_trunc_svd_chi_max_no_truncation() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // chi_max >= k=2 means no truncation
        let params = TruncSvdParams {
            chi_max: Some(5),
            target_trunc_err: None,
        };
        let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 2]);
        assert_eq!(trunc_err, 0.0);
    }

    #[test]
    fn test_trunc_svd_target_trunc_err() {
        let backend = CpuBackend::new();
        // 4×4 matrix
        let data: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let tensor = DenseTensor::from_data(data, vec![4, 4]);

        // Full SVD first to know the singular values
        let (_, s_full, _, _) = trunc_svd(
            &backend,
            &tensor,
            1,
            &TruncSvdParams {
                chi_max: None,
                target_trunc_err: None,
            },
        )
        .unwrap();

        // Set threshold just above the smallest singular value
        let smallest_sv = s_full.get(&[s_full.len() - 1]);
        let params = TruncSvdParams {
            chi_max: None,
            target_trunc_err: Some(smallest_sv + 1e-10),
        };
        let (_u, s, _vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

        // Should have discarded the smallest singular value
        assert!(s.len() < s_full.len());
        // Truncation error should be approximately equal to the discarded singular value
        assert!(trunc_err <= smallest_sv + 1e-10);
    }

    #[test]
    fn test_trunc_svd_both_params() {
        let backend = CpuBackend::new();
        // 4×4 matrix
        let data: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let tensor = DenseTensor::from_data(data, vec![4, 4]);

        // Full SVD to get singular values
        let (_, s_full, _, _) = trunc_svd(
            &backend,
            &tensor,
            1,
            &TruncSvdParams {
                chi_max: None,
                target_trunc_err: None,
            },
        )
        .unwrap();
        let k_full = s_full.len();

        // chi_max is the binding constraint: target_trunc_err=0 forces keeping all,
        // but chi_max limits to 2
        let params = TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: Some(0.0),
        };
        let (_, s, _, _) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
        assert_eq!(s.len(), 2);

        // target_trunc_err is the binding constraint: chi_max allows all,
        // but large target_trunc_err allows aggressive truncation
        let params = TruncSvdParams {
            chi_max: Some(k_full),
            target_trunc_err: Some(1e10),
        };
        let (_, s, _, _) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
        // Large allowed error → aggressive truncation → minimum 1 value kept
        assert_eq!(s.len(), 1);

        // Neither constraint truncates: chi_max=k_full, target_trunc_err=0
        let params = TruncSvdParams {
            chi_max: Some(k_full),
            target_trunc_err: Some(0.0),
        };
        let (_, s, _, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
        assert_eq!(s.len(), k_full);
        assert_eq!(trunc_err, 0.0);
    }

    #[test]
    fn test_trunc_svd_f32() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f32>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );

        let params = TruncSvdParams {
            chi_max: Some(1),
            target_trunc_err: None,
        };
        let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

        assert_eq!(u.shape(), &[2, 1]);
        assert_eq!(s.shape(), &[1]);
        assert_eq!(vt.shape(), &[1, 3]);
        assert!(trunc_err > 0.0);
    }

    #[test]
    fn test_trunc_svd_reconstruction() {
        let backend = CpuBackend::new();
        // Verify that truncated reconstruction is a valid low-rank approximation
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 4],
        );

        let params = TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        };
        let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

        let (m, n, chi) = (3, 4, 2);

        // Reconstruct: A_approx = U * diag(S) * Vt
        let mut recon_err_sq = 0.0;
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..chi {
                    val += u.get(&[i, l]) * s.get(&[l]) * vt.get(&[l, j]);
                }
                let diff = val - tensor.data()[i * n + j];
                recon_err_sq += diff * diff;
            }
        }
        let recon_err = recon_err_sq.sqrt();

        // Reconstruction error should equal the truncation error
        // (Eckart-Young theorem: ||A - A_k||_F = sqrt(sum of discarded σ²))
        assert!(
            (recon_err - trunc_err).abs() < 1e-10,
            "recon_err={recon_err} vs trunc_err={trunc_err}"
        );
    }

    // --- QR tests ---

    #[test]
    fn test_qr_f64_2d() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let (q, r) = qr(&backend, &tensor, 1).unwrap();

        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);

        // Reconstruct: A ≈ Q * R
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += q.get(&[i, k]) * r.get(&[k, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-10,
                    "QR reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_qr_f64_rectangular() {
        let backend = CpuBackend::new();
        // shape [3, 2], nrow=1 → 3×2 matrix, k=2
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        );

        let (q, r) = qr(&backend, &tensor, 1).unwrap();

        let (m, n, k) = (3, 2, 2);
        assert_eq!(q.shape(), &[m, k]);
        assert_eq!(r.shape(), &[k, n]);

        // Reconstruct
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += q.get(&[i, l]) * r.get(&[l, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-10,
                    "QR reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_qr_f64_orthogonality() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        );

        let (q, _r) = qr(&backend, &tensor, 1).unwrap();

        let (m, k) = (3, 2);
        // Q^T * Q should be identity (k×k)
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for l in 0..m {
                    dot += q.get(&[l, i]) * q.get(&[l, j]);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q orthogonality failed: Q^T*Q[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_qr_f64_higher_rank() {
        let backend = CpuBackend::new();
        // shape [2, 3, 4], nrow=2 → m=6, n=4, k=4
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

        let (q, r) = qr(&backend, &tensor, 2).unwrap();

        let (m, n, k) = (6, 4, 4);
        assert_eq!(q.shape(), &[m, k]);
        assert_eq!(r.shape(), &[k, n]);

        // Reconstruct and verify
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += q.get(&[i, l]) * r.get(&[l, j]);
                }
                let orig = tensor.data()[i * n + j];
                assert!(
                    (val - orig).abs() < 1e-9,
                    "QR reconstruction mismatch at ({i},{j}): {val} vs {orig}"
                );
            }
        }
    }

    #[test]
    fn test_qr_f32() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let (q, r) = qr(&backend, &tensor, 1).unwrap();

        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);

        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0f32;
                for k in 0..2 {
                    val += q.get(&[i, k]) * r.get(&[k, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-4,
                    "QR reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_qr_invalid_nrow() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        assert!(qr(&backend, &tensor, 0).is_err());
        assert!(qr(&backend, &tensor, 2).is_err());
    }

    // --- LQ tests ---

    #[test]
    fn test_lq_f64_2d() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let (l, q) = lq(&backend, &tensor, 1).unwrap();

        assert_eq!(l.shape(), &[2, 2]);
        assert_eq!(q.shape(), &[2, 2]);

        // Reconstruct: A ≈ L * Q
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += l.get(&[i, k]) * q.get(&[k, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-10,
                    "LQ reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_lq_f64_rectangular() {
        let backend = CpuBackend::new();
        // shape [2, 3], nrow=1 → 2×3 matrix, k=2
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );

        let (l, q) = lq(&backend, &tensor, 1).unwrap();

        let (m, n, k) = (2, 3, 2);
        assert_eq!(l.shape(), &[m, k]);
        assert_eq!(q.shape(), &[k, n]);

        // Reconstruct
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for ll in 0..k {
                    val += l.get(&[i, ll]) * q.get(&[ll, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-10,
                    "LQ reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_lq_f64_orthogonality() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );

        let (_l, q) = lq(&backend, &tensor, 1).unwrap();

        let (k, n) = (2, 3);
        // Q * Q^T should be identity (k×k)
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for l in 0..n {
                    dot += q.get(&[i, l]) * q.get(&[j, l]);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q orthogonality failed: Q*Q^T[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_lq_f64_higher_rank() {
        let backend = CpuBackend::new();
        // shape [2, 3, 4], nrow=1 → m=2, n=12, k=2
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

        let (l, q) = lq(&backend, &tensor, 1).unwrap();

        let (m, n, k) = (2, 12, 2);
        assert_eq!(l.shape(), &[m, k]);
        assert_eq!(q.shape(), &[k, n]);

        // Reconstruct and verify
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for ll in 0..k {
                    val += l.get(&[i, ll]) * q.get(&[ll, j]);
                }
                let orig = tensor.data()[i * n + j];
                assert!(
                    (val - orig).abs() < 1e-9,
                    "LQ reconstruction mismatch at ({i},{j}): {val} vs {orig}"
                );
            }
        }
    }

    #[test]
    fn test_lq_f32() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let (l, q) = lq(&backend, &tensor, 1).unwrap();

        assert_eq!(l.shape(), &[2, 2]);
        assert_eq!(q.shape(), &[2, 2]);

        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0f32;
                for k in 0..2 {
                    val += l.get(&[i, k]) * q.get(&[k, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-4,
                    "LQ reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_lq_invalid_nrow() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        assert!(lq(&backend, &tensor, 0).is_err());
        assert!(lq(&backend, &tensor, 2).is_err());
    }
}
