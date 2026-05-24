use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder, QrDescriptor, SvdDescriptor};
use arnet_tensor::{ComputeBackendTensorExt, Dense, DenseTensor};
use num_traits::{Float, ToPrimitive, Zero};

use crate::error::LinalgError;
use crate::tensor_bridge::{wrap_dense, wrap_dense_as};
use arnet_tensor::reorder;

/// Result of a thin SVD decomposition: `(U, S, Vt)`.
///
/// - `U`: Left singular vectors
/// - `S`: Singular values (real-valued, descending)
/// - `Vt`: Right singular vectors transposed
pub type SvdResult<T, B> = (
    DenseTensor<T, B>,
    DenseTensor<<T as Scalar>::Real, B>,
    DenseTensor<T, B>,
);

/// Internal kernel form of [`SvdResult`] operating on legacy `Dense<T>`.
pub(crate) type SvdResultDense<T> = (Dense<T>, Dense<<T as Scalar>::Real>, Dense<T>);

/// Result of a truncated SVD decomposition: `(U, S, Vt, trunc_err)`.
///
/// - `U`: Left singular vectors (truncated)
/// - `S`: Singular values (real-valued, descending, truncated)
/// - `Vt`: Right singular vectors transposed (truncated)
/// - `trunc_err`: Truncation error -- Frobenius norm of discarded singular values
pub type TruncSvdResult<T, B> = (
    DenseTensor<T, B>,
    DenseTensor<<T as Scalar>::Real, B>,
    DenseTensor<T, B>,
    <T as Scalar>::Real,
);

/// Internal kernel form of [`TruncSvdResult`] operating on legacy `Dense<T>`.
pub(crate) type TruncSvdResultDense<T> = (
    Dense<T>,
    Dense<<T as Scalar>::Real>,
    Dense<T>,
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

mod lq;
pub(crate) use lq::lq_with_policy_dense;
pub use lq::{LqResult, lq, lq_with_policy};

/// Reshape tensor to 2D (m x n) using row-major axis merge, then convert to target order.
///
/// For rank > 2 tensors, direct column-major flattening merges axes in
/// column-major order, which differs from the standard mathematical reshape.
/// This function ensures row-major merge semantics regardless of input layout.
pub(super) fn reshape_for_backend<T: Scalar>(
    tensor: &Dense<T>,
    m: usize,
    n: usize,
    order: MemoryOrder,
) -> Dense<T> {
    // Reorder to RowMajor layout, reshape to 2D, then reorder to backend order
    let rm = reorder(tensor, tensor.order(), MemoryOrder::RowMajor);
    let mat_2d = Dense::new(rm.data().to_vec(), vec![m, n], MemoryOrder::RowMajor);
    // mat_2d data is in RowMajor; convert to backend's preferred order
    reorder(&mat_2d, MemoryOrder::RowMajor, order)
}

/// Compute thin SVD of a tensor reshaped as a matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// Returns `(U, S, Vt)` where A ~ U * diag(S) * Vt.
///
/// # Arguments
///
/// * `tensor` - Input tensor (backend flows from the tensor)
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
/// Returns `LinalgError` if `nrow` is out of range or the backend fails.
pub fn svd<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<SvdResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let (u, s, vt) = svd_dense(tensor.backend(), &dense, nrow)?;
    Ok((
        wrap_dense(u, backend_arc.clone()),
        wrap_dense_as(s, backend_arc.clone()),
        wrap_dense(vt, backend_arc),
    ))
}

/// Internal kernel for [`svd`] on legacy `Dense<T>`.
pub(crate) fn svd_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<SvdResultDense<T>, LinalgError> {
    let (m, n) = if nrow == 0 || nrow >= tensor.rank() {
        (0, 0)
    } else {
        let m: usize = tensor.shape()[..nrow].iter().product();
        let n: usize = tensor.shape()[nrow..].iter().product();
        (m, n)
    };
    let policy = backend.par_for_svd(m, n);
    svd_with_policy_dense(backend, tensor, nrow, policy)
}

/// Thin SVD with caller-specified execution policy.
///
/// Expert-layer counterpart of [`svd`]; the default wrapper consults
/// `backend.par_for_svd`, while this entry point takes `policy` directly.
pub fn svd_with_policy<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<SvdResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let (u, s, vt) = svd_with_policy_dense(tensor.backend(), &dense, nrow, policy)?;
    Ok((
        wrap_dense(u, backend_arc.clone()),
        wrap_dense_as(s, backend_arc.clone()),
        wrap_dense(vt, backend_arc),
    ))
}

/// Internal kernel for [`svd_with_policy`] on legacy `Dense<T>`.
pub(crate) fn svd_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<SvdResultDense<T>, LinalgError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(LinalgError::InvalidArgument(format!(
            "nrow must satisfy 1 <= nrow < rank, got nrow={nrow} for rank={rank}"
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
        a: mat_2d.data(),
        u: &mut u_data,
        s: &mut s_data,
        vt: &mut vt_data,
        order,
        policy,
    };

    backend.svd(desc)?;

    let u_tensor = backend.make_tensor(u_data, vec![m, k]);
    let s_tensor = Dense::new(s_data, vec![k], order);
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
/// * `tensor` - Input tensor (backend flows from the tensor)
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
/// Returns `LinalgError` if `nrow` is out of range or the backend fails.
pub fn trunc_svd<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<TruncSvdResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let (u, s, vt, err) = trunc_svd_dense(tensor.backend(), &dense, nrow, params)?;
    Ok((
        wrap_dense(u, backend_arc.clone()),
        wrap_dense_as(s, backend_arc.clone()),
        wrap_dense(vt, backend_arc),
        err,
    ))
}

/// Internal kernel for [`trunc_svd`] on legacy `Dense<T>`.
pub(crate) fn trunc_svd_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<TruncSvdResultDense<T>, LinalgError> {
    let (m, n) = if nrow == 0 || nrow >= tensor.rank() {
        (0, 0)
    } else {
        let m: usize = tensor.shape()[..nrow].iter().product();
        let n: usize = tensor.shape()[nrow..].iter().product();
        (m, n)
    };
    let policy = backend.par_for_svd(m, n);
    trunc_svd_with_policy_dense(backend, tensor, nrow, params, policy)
}

/// Truncated SVD with caller-specified execution policy.
///
/// Expert-layer counterpart of [`trunc_svd`]; the default wrapper consults
/// `backend.par_for_svd`, while this entry point takes `policy` directly.
pub fn trunc_svd_with_policy<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
    params: &TruncSvdParams,
    policy: ExecPolicy,
) -> Result<TruncSvdResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let (u, s, vt, err) =
        trunc_svd_with_policy_dense(tensor.backend(), &dense, nrow, params, policy)?;
    Ok((
        wrap_dense(u, backend_arc.clone()),
        wrap_dense_as(s, backend_arc.clone()),
        wrap_dense(vt, backend_arc),
        err,
    ))
}

/// Internal kernel for [`trunc_svd_with_policy`] on legacy `Dense<T>`.
pub(crate) fn trunc_svd_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
    params: &TruncSvdParams,
    policy: ExecPolicy,
) -> Result<TruncSvdResultDense<T>, LinalgError> {
    let (u_full, s_full, vt_full) = svd_with_policy_dense(backend, tensor, nrow, policy)?;

    let shape = tensor.shape();
    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();
    let k_full = m.min(n);

    // Determine how many singular values to keep
    let mut chi = k_full;

    // Apply chi_max bound
    if let Some(chi_max) = params.chi_max {
        if chi_max == 0 {
            return Err(LinalgError::InvalidArgument(
                "chi_max must be at least 1".into(),
            ));
        }
        chi = chi.min(chi_max);
    }

    // S is always 1D -- data() works for any 1D contiguous tensor
    let s_data = s_full.data();

    // Apply target_trunc_err bound: keep the largest singular values
    // such that the norm of discarded values stays within the threshold
    if let Some(target_err) = params.target_trunc_err {
        // Accumulate discarded norm^2 from the smallest singular value upward.
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

    // Truncate S: [k_full] -> [chi]
    let s_trunc: Vec<T::Real> = s_data[..chi].to_vec();

    // Layout-aware truncation of U and Vt
    let order = backend.preferred_order();
    let u_raw = u_full.data();
    let vt_raw = vt_full.data();

    let (u_trunc, vt_trunc) = match order {
        // TODO: untested -- no RowMajor backend exists yet
        //
        // U row-major [m, k_full] -> [m, chi]: copy first chi elements per row
        // let mut u_t = vec![T::zero(); m * chi];
        // for i in 0..m {
        //     u_t[i * chi..(i + 1) * chi]
        //         .copy_from_slice(&u_raw[i * k_full..i * k_full + chi]);
        // }
        // Vt row-major [k_full, n] -> [chi, n]: take first chi rows
        // let vt_t: Vec<T> = vt_raw[..chi * n].to_vec();
        // (u_t, vt_t)
        MemoryOrder::RowMajor => {
            todo!("RowMajor truncation slicing")
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
    let s_tensor = Dense::new(s_trunc, vec![chi], order);
    let vt_tensor = backend.make_tensor(vt_trunc, vec![chi, n]);

    Ok((u_tensor, s_tensor, vt_tensor, trunc_err))
}

/// Result of a thin QR decomposition: `(Q, R)`.
///
/// - `Q`: Orthogonal/unitary matrix, shape `[m, k]` where `k = min(m, n)`
/// - `R`: Upper triangular matrix, shape `[k, n]`
pub type QrResult<T, B> = (DenseTensor<T, B>, DenseTensor<T, B>);

/// Internal kernel form of [`QrResult`] operating on legacy `Dense<T>`.
pub(crate) type QrResultDense<T> = (Dense<T>, Dense<T>);

/// Compute thin QR decomposition of a tensor reshaped as a matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// Returns `(Q, R)` where A = Q * R.
///
/// # Arguments
///
/// * `tensor` - Input tensor (backend flows from the tensor)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// * `Q` - Orthogonal matrix, shape `[m, k]` where `m = product(shape[..nrow])`, `k = min(m, n)`
/// * `R` - Upper triangular matrix, shape `[k, n]` where `n = product(shape[nrow..])`
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range or the backend fails.
pub fn qr<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<QrResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let (q, r) = qr_dense(tensor.backend(), &dense, nrow)?;
    Ok((
        wrap_dense(q, backend_arc.clone()),
        wrap_dense(r, backend_arc),
    ))
}

/// Internal kernel for [`qr`] on legacy `Dense<T>`.
pub(crate) fn qr_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<QrResultDense<T>, LinalgError> {
    let (m, n) = if nrow == 0 || nrow >= tensor.rank() {
        (0, 0)
    } else {
        let m: usize = tensor.shape()[..nrow].iter().product();
        let n: usize = tensor.shape()[nrow..].iter().product();
        (m, n)
    };
    let policy = backend.par_for_qr(m, n);
    qr_with_policy_dense(backend, tensor, nrow, policy)
}

/// Thin QR with caller-specified execution policy.
///
/// Expert-layer counterpart of [`qr`]; the default wrapper consults
/// `backend.par_for_qr`, while this entry point takes `policy` directly.
pub fn qr_with_policy<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<QrResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let (q, r) = qr_with_policy_dense(tensor.backend(), &dense, nrow, policy)?;
    Ok((
        wrap_dense(q, backend_arc.clone()),
        wrap_dense(r, backend_arc),
    ))
}

/// Internal kernel for [`qr_with_policy`] on legacy `Dense<T>`.
pub(crate) fn qr_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<QrResultDense<T>, LinalgError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(LinalgError::InvalidArgument(format!(
            "nrow must satisfy 1 <= nrow < rank, got nrow={nrow} for rank={rank}"
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
        a: mat_2d.data(),
        q: &mut q_data,
        r: &mut r_data,
        order,
        policy,
    };

    backend.qr(desc)?;

    let q_tensor = backend.make_tensor(q_data, vec![m, k]);
    let r_tensor = backend.make_tensor(r_data, vec![k, n]);

    Ok((q_tensor, r_tensor))
}
