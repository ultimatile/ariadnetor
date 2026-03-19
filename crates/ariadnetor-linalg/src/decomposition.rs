use arnet_core::backend::{
    BackendError, ComputeBackend, LqDescriptor, QrDescriptor, SvdDescriptor,
};
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;
use num_traits::{Float, ToPrimitive, Zero};

/// Result of a thin SVD decomposition: `(U, S, Vt)`.
///
/// - `U`: Left singular vectors
/// - `S`: Singular values (real-valued, descending)
/// - `Vt`: Right singular vectors transposed
pub type SvdResult<T> = (
    DenseTensor<T>,
    DenseTensor<<T as Scalar>::Real>,
    DenseTensor<T>,
);

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
