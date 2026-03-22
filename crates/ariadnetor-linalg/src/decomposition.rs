use arnet_core::backend::{
    BackendError, ComputeBackend, LqDescriptor, QrDescriptor, SvdDescriptor,
};
use arnet_core::scalar::Scalar;
use arnet_tensor::{ComputeBackendTensorExt, DenseTensor, MemoryOrder};
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

/// Reshape tensor to 2D (m×n) using row-major axis merge, then convert to target order.
///
/// For rank > 2 tensors, direct column-major flattening merges axes in
/// column-major order, which differs from the standard mathematical reshape.
/// This function ensures row-major merge semantics regardless of input layout.
fn reshape_for_backend<T: Scalar>(
    tensor: &DenseTensor<T>,
    m: usize,
    n: usize,
    order: MemoryOrder,
) -> DenseTensor<T> {
    // Row-major reshape to 2D (standard mathematical convention)
    let rm = tensor.to_contiguous(MemoryOrder::RowMajor);
    let mat_2d = DenseTensor::from_data(rm.data().to_vec(), vec![m, n]);
    // Convert to backend's preferred order
    mat_2d.to_contiguous(order)
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

    let order = backend.preferred_order();
    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();
    let k = m.min(n);

    // Reshape to 2D using row-major merge (standard mathematical convention),
    // then convert to the backend's preferred order.
    let mat_2d = reshape_for_backend(tensor, m, n, order);

    let mut u_data = vec![T::zero(); m * k];
    let mut s_data = vec![T::Real::zero(); k];
    let mut vt_data = vec![T::zero(); k * n];

    let desc = SvdDescriptor {
        m,
        n,
        a: mat_2d.data_contiguous(),
        u: &mut u_data,
        s: &mut s_data,
        vt: &mut vt_data,
    };

    backend.svd(desc)?;

    let u_tensor = backend.make_tensor(u_data, vec![m, k]);
    let s_tensor = DenseTensor::from_data(s_data, vec![k]);
    let vt_tensor = backend.make_tensor(vt_data, vec![k, n]);

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

    // S is always 1D — data() works for any 1D contiguous tensor
    let s_data = s_full.data();

    // Apply target_trunc_err bound: keep the largest singular values
    // such that the norm of discarded values stays within the threshold
    if let Some(target_err) = params.target_trunc_err {
        // Accumulate discarded norm² from the smallest singular value upward.
        // Compare in f64 to avoid precision issues with the user-specified threshold.
        let target_sq = target_err * target_err;
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
    let mut err_sq = T::Real::zero();
    for &si in &s_data[chi..] {
        err_sq = err_sq + si * si;
    }
    let trunc_err = err_sq.sqrt();

    // Truncate S: [k_full] → [chi]
    let s_trunc: Vec<T::Real> = s_data[..chi].to_vec();

    // Layout-aware truncation of U and Vt
    let order = backend.preferred_order();
    let u_raw = u_full.data_contiguous();
    let vt_raw = vt_full.data_contiguous();

    let (u_trunc, vt_trunc) = match order {
        MemoryOrder::RowMajor => {
            // U row-major [m, k_full] → [m, chi]: copy first chi elements per row
            let mut u_t = vec![T::zero(); m * chi];
            for i in 0..m {
                u_t[i * chi..(i + 1) * chi].copy_from_slice(&u_raw[i * k_full..i * k_full + chi]);
            }
            // Vt row-major [k_full, n] → [chi, n]: take first chi rows
            let vt_t: Vec<T> = vt_raw[..chi * n].to_vec();
            (u_t, vt_t)
        }
        MemoryOrder::ColumnMajor => {
            // U col-major [m, k_full]: columns are contiguous blocks of m elements.
            // First chi columns = first m*chi elements.
            let u_t: Vec<T> = u_raw[..m * chi].to_vec();
            // Vt col-major [k_full, n]: each column has k_full elements.
            // Extract first chi elements (rows) from each of the n columns.
            let mut vt_t = vec![T::zero(); chi * n];
            for j in 0..n {
                vt_t[j * chi..j * chi + chi].copy_from_slice(&vt_raw[j * k_full..j * k_full + chi]);
            }
            (u_t, vt_t)
        }
    };

    let u_tensor = backend.make_tensor(u_trunc, vec![m, chi]);
    let s_tensor = DenseTensor::from_data(s_trunc, vec![chi]);
    let vt_tensor = backend.make_tensor(vt_trunc, vec![chi, n]);

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

    let order = backend.preferred_order();
    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();
    let k = m.min(n);

    let mat_2d = reshape_for_backend(tensor, m, n, order);

    let mut q_data = vec![T::zero(); m * k];
    let mut r_data = vec![T::zero(); k * n];

    let desc = QrDescriptor {
        m,
        n,
        a: mat_2d.data_contiguous(),
        q: &mut q_data,
        r: &mut r_data,
    };

    backend.qr(desc)?;

    let q_tensor = backend.make_tensor(q_data, vec![m, k]);
    let r_tensor = backend.make_tensor(r_data, vec![k, n]);

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

    let order = backend.preferred_order();
    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();
    let k = m.min(n);

    let mat_2d = reshape_for_backend(tensor, m, n, order);

    let mut l_data = vec![T::zero(); m * k];
    let mut q_data = vec![T::zero(); k * n];

    let desc = LqDescriptor {
        m,
        n,
        a: mat_2d.data_contiguous(),
        l: &mut l_data,
        q: &mut q_data,
    };

    backend.lq(desc)?;

    let l_tensor = backend.make_tensor(l_data, vec![m, k]);
    let q_tensor = backend.make_tensor(q_data, vec![k, n]);

    Ok((l_tensor, q_tensor))
}
