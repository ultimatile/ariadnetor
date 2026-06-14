use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder, QrDescriptor, SvdDescriptor};
use arnet_tensor::{ComputeBackendTensorExt, DenseStorage, DenseTensor, DenseTensorData, OpsFor};
use num_traits::{Float, ToPrimitive, Zero};

use crate::error::LinalgError;
use arnet_tensor::reorder_data;

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

/// Internal kernel form of [`SvdResult`] operating on joined
/// [`DenseTensorData<T>`].
pub(crate) type SvdResultDense<T> = (
    DenseTensorData<T>,
    DenseTensorData<<T as Scalar>::Real>,
    DenseTensorData<T>,
);

/// Result of a truncated SVD decomposition: `(U, S, Vt, trunc_err)`.
///
/// - `U`: Left singular vectors (truncated)
/// - `S`: Singular values (real-valued, descending, truncated)
/// - `Vt`: Right singular vectors transposed (truncated)
/// - `trunc_err`: Truncation error -- Frobenius norm of discarded singular values
pub type TruncSvdResult<T> = (
    DenseTensor<T>,
    DenseTensor<<T as Scalar>::Real>,
    DenseTensor<T>,
    <T as Scalar>::Real,
);

/// Internal kernel form of [`TruncSvdResult`] operating on joined
/// [`DenseTensorData<T>`].
pub(crate) type TruncSvdResultDense<T> = (
    DenseTensorData<T>,
    DenseTensorData<<T as Scalar>::Real>,
    DenseTensorData<T>,
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
pub use lq::{LqResult, lq_with_policy};
pub(crate) use lq::{lq_dense, lq_with_policy_dense};

/// Reshape tensor to 2D (m x n) using row-major axis merge, then convert to target order.
///
/// For rank > 2 tensors, direct column-major flattening merges axes in
/// column-major order, which differs from the standard mathematical reshape.
/// This function ensures row-major merge semantics regardless of input layout.
pub(super) fn reshape_for_backend<T: Scalar>(
    tensor: &DenseTensorData<T>,
    m: usize,
    n: usize,
    order: MemoryOrder,
) -> DenseTensorData<T> {
    // Reorder to RowMajor layout, reshape to 2D, then reorder to backend order
    let rm = reorder_data(tensor, MemoryOrder::RowMajor);
    let mat_2d =
        DenseTensorData::from_raw_parts(rm.data().to_vec(), vec![m, n], MemoryOrder::RowMajor);
    // mat_2d data is in RowMajor; convert to backend's preferred order
    reorder_data(&mat_2d, order)
}

/// Internal kernel for the SVD operation on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn svd_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
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

/// Thin SVD with an explicit backend and caller-specified execution policy.
///
/// Expert-layer counterpart of [`crate::svd_with_backend`]; that entry point
/// consults `backend.par_for_svd`, while this one takes `policy` directly.
/// The backend is supplied at the call site and the tensor's own backend is
/// never consulted.
pub fn svd_with_policy<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<SvdResult<T>, LinalgError> {
    let (u, s, vt) = svd_with_policy_dense(backend, tensor.data(), nrow, policy)?;
    Ok((
        DenseTensor::from_data(u),
        DenseTensor::from_data(s),
        DenseTensor::from_data(vt),
    ))
}

/// Internal kernel for [`svd_with_policy`] on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn svd_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
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
    let s_tensor = DenseTensorData::from_raw_parts(s_data, vec![k], order);
    let vt_tensor = backend.make_tensor(vt_data, vec![k, n]);

    Ok((u_tensor, s_tensor, vt_tensor))
}

/// Internal kernel for the truncated SVD operation on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn trunc_svd_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
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

/// Truncated SVD with an explicit backend and caller-specified execution
/// policy.
///
/// Expert-layer counterpart of [`crate::trunc_svd_with_backend`]; that entry
/// point consults `backend.par_for_svd`, while this one takes `policy`
/// directly. The backend is supplied at the call site and the tensor's own
/// backend is never consulted.
pub fn trunc_svd_with_policy<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
    params: &TruncSvdParams,
    policy: ExecPolicy,
) -> Result<TruncSvdResult<T>, LinalgError> {
    let (u, s, vt, err) =
        trunc_svd_with_policy_dense(backend, tensor.data(), nrow, params, policy)?;
    Ok((
        DenseTensor::from_data(u),
        DenseTensor::from_data(s),
        DenseTensor::from_data(vt),
        err,
    ))
}

/// Internal kernel for [`trunc_svd_with_policy`] on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn trunc_svd_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
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
    let s_tensor = DenseTensorData::from_raw_parts(s_trunc, vec![chi], order);
    let vt_tensor = backend.make_tensor(vt_trunc, vec![chi, n]);

    Ok((u_tensor, s_tensor, vt_tensor, trunc_err))
}

/// Result of a thin QR decomposition: `(Q, R)`.
///
/// - `Q`: Orthogonal/unitary matrix, shape `[m, k]` where `k = min(m, n)`
/// - `R`: Upper triangular matrix, shape `[k, n]`
pub type QrResult<T> = (DenseTensor<T>, DenseTensor<T>);

/// Internal kernel form of [`QrResult`] operating on joined
/// [`DenseTensorData<T>`].
pub(crate) type QrResultDense<T> = (DenseTensorData<T>, DenseTensorData<T>);

/// Internal kernel for the QR operation on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn qr_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
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

/// Thin QR with an explicit backend and caller-specified execution policy.
///
/// Expert-layer counterpart of [`crate::qr_with_backend`]; that entry point
/// consults `backend.par_for_qr`, while this one takes `policy` directly. The
/// backend is supplied at the call site and the tensor's own backend is never
/// consulted.
pub fn qr_with_policy<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<QrResult<T>, LinalgError> {
    let (q, r) = qr_with_policy_dense(backend, tensor.data(), nrow, policy)?;
    Ok((DenseTensor::from_data(q), DenseTensor::from_data(r)))
}

/// Internal kernel for [`qr_with_policy`] on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn qr_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
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
