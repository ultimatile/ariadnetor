use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, EigDescriptor, EighDescriptor, ExecPolicy, MemoryOrder};
use arnet_tensor::{ComputeBackendTensorExt, DenseTensor, DenseTensorData};
use num_traits::Zero;

use crate::error::LinalgError;
use arnet_tensor::reorder_data;

/// Result of a self-adjoint eigenvalue decomposition: `(eigenvalues, eigenvectors)`.
///
/// - Eigenvalues: `DenseTensor<T::Real, B>` with shape `[n]`, sorted ascending
/// - Eigenvectors: `DenseTensor<T, B>` with shape `[n, n]`, columns are eigenvectors
pub type EighResult<T, B> = (DenseTensor<<T as Scalar>::Real, B>, DenseTensor<T, B>);

/// Internal kernel form of [`EighResult`] operating on joined
/// [`DenseTensorData<T>`].
pub(crate) type EighResultDense<T> = (DenseTensorData<<T as Scalar>::Real>, DenseTensorData<T>);

/// Compute self-adjoint eigenvalue decomposition of a tensor reshaped as a square matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// The resulting matrix must be square.
///
/// # Arguments
///
/// * `tensor` - Input tensor (backend flows from the tensor)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// * Eigenvalues: shape `[n]`, real, sorted ascending
/// * Eigenvectors: shape `[n, n]`, columns are eigenvectors
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigh<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<EighResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let (w, v) = eigh_dense(tensor.backend(), tensor.data(), nrow)?;
    Ok((
        DenseTensor::with_backend(w, backend_arc.clone()),
        DenseTensor::with_backend(v, backend_arc),
    ))
}

/// Internal kernel for [`eigh`] on the joined [`DenseTensorData<T>`] form.
pub(crate) fn eigh_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<EighResultDense<T>, LinalgError> {
    let n = if nrow == 0 || nrow >= tensor.rank() {
        0
    } else {
        tensor.shape()[..nrow].iter().product()
    };
    let policy = backend.par_for_eigh(n);
    eigh_with_policy_dense(backend, tensor, nrow, policy)
}

/// Self-adjoint eigenvalue decomposition with caller-specified execution policy.
///
/// Expert-layer counterpart of [`eigh`]; the default wrapper consults
/// `backend.par_for_eigh`, while this entry point takes `policy` directly.
pub fn eigh_with_policy<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<EighResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let (w, v) = eigh_with_policy_dense(tensor.backend(), tensor.data(), nrow, policy)?;
    Ok((
        DenseTensor::with_backend(w, backend_arc.clone()),
        DenseTensor::with_backend(v, backend_arc),
    ))
}

/// Internal kernel for [`eigh_with_policy`] on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn eigh_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<EighResultDense<T>, LinalgError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(LinalgError::InvalidArgument(format!(
            "nrow must satisfy 1 <= nrow < rank, got nrow={nrow} for rank={rank}"
        )));
    }

    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();

    if m != n {
        return Err(LinalgError::InvalidArgument(format!(
            "eigh requires a square matrix, got {m}x{n}"
        )));
    }

    let order = backend.preferred_order();
    // Ensure row-major reshape to 2D, then convert to backend order
    let rm = reorder_data(tensor, MemoryOrder::RowMajor);
    let mat_2d =
        DenseTensorData::from_raw_parts(rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor);
    let contiguous = reorder_data(&mat_2d, order);

    let mut w_data = vec![T::Real::zero(); n];
    let mut v_data = vec![T::zero(); n * n];

    let desc = EighDescriptor {
        n,
        a: contiguous.data(),
        w: &mut w_data,
        v: &mut v_data,
        order,
        policy,
    };

    backend.eigh(desc)?;

    let w_tensor = DenseTensorData::from_raw_parts(w_data, vec![n], order);
    let v_tensor = backend.make_tensor(v_data, vec![n, n]);

    Ok((w_tensor, v_tensor))
}

/// Compute eigenvalues of a self-adjoint tensor reshaped as a square matrix.
///
/// This is a convenience wrapper around [`eigh`] that discards the eigenvectors.
///
/// # Arguments
///
/// * `tensor` - Input tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Eigenvalues: shape `[n]`, real, sorted ascending
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigvalsh<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<DenseTensor<T::Real, B>, LinalgError> {
    let (w, _v) = eigh(tensor, nrow)?;
    Ok(w)
}

/// Result of a general eigenvalue decomposition: `(eigenvalues, eigenvectors)`.
///
/// - Eigenvalues: `DenseTensor<T::Complex, B>` with shape `[n]`, complex
/// - Eigenvectors: `DenseTensor<T::Complex, B>` with shape `[n, n]`, complex, columns are right eigenvectors
pub type EigResult<T, B> = (
    DenseTensor<<T as Scalar>::Complex, B>,
    DenseTensor<<T as Scalar>::Complex, B>,
);

/// Internal kernel form of [`EigResult`] operating on joined
/// [`DenseTensorData<T>`].
pub(crate) type EigResultDense<T> = (
    DenseTensorData<<T as Scalar>::Complex>,
    DenseTensorData<<T as Scalar>::Complex>,
);

/// Compute general eigenvalue decomposition of a tensor reshaped as a square matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// The resulting matrix must be square. Eigenvalues and eigenvectors are always complex.
///
/// # Arguments
///
/// * `tensor` - Input tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// * Eigenvalues: shape `[n]`, complex
/// * Eigenvectors: shape `[n, n]`, complex, columns are right eigenvectors
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eig<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<EigResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let (w, v) = eig_dense(tensor.backend(), tensor.data(), nrow)?;
    Ok((
        DenseTensor::with_backend(w, backend_arc.clone()),
        DenseTensor::with_backend(v, backend_arc),
    ))
}

/// Internal kernel for [`eig`] on the joined [`DenseTensorData<T>`] form.
pub(crate) fn eig_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<EigResultDense<T>, LinalgError> {
    let n = if nrow == 0 || nrow >= tensor.rank() {
        0
    } else {
        tensor.shape()[..nrow].iter().product()
    };
    let policy = backend.par_for_eig(n);
    eig_with_policy_dense(backend, tensor, nrow, policy)
}

/// General eigenvalue decomposition with caller-specified execution policy.
///
/// Expert-layer counterpart of [`eig`]; the default wrapper consults
/// `backend.par_for_eig`, while this entry point takes `policy` directly.
pub fn eig_with_policy<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<EigResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let (w, v) = eig_with_policy_dense(tensor.backend(), tensor.data(), nrow, policy)?;
    Ok((
        DenseTensor::with_backend(w, backend_arc.clone()),
        DenseTensor::with_backend(v, backend_arc),
    ))
}

/// Internal kernel for [`eig_with_policy`] on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn eig_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<EigResultDense<T>, LinalgError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(LinalgError::InvalidArgument(format!(
            "nrow must satisfy 1 <= nrow < rank, got nrow={nrow} for rank={rank}"
        )));
    }

    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();

    if m != n {
        return Err(LinalgError::InvalidArgument(format!(
            "eig requires a square matrix, got {m}x{n}"
        )));
    }

    let order = backend.preferred_order();
    let rm = reorder_data(tensor, MemoryOrder::RowMajor);
    let mat_2d =
        DenseTensorData::from_raw_parts(rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor);
    let contiguous = reorder_data(&mat_2d, order);

    let mut w_data = vec![T::Complex::zero(); n];
    let mut v_data = vec![T::Complex::zero(); n * n];

    let desc = EigDescriptor {
        n,
        a: contiguous.data(),
        w: &mut w_data,
        v: &mut v_data,
        order,
        policy,
    };

    backend.eig(desc)?;

    let w_tensor = DenseTensorData::from_raw_parts(w_data, vec![n], order);
    let v_tensor = backend.make_tensor(v_data, vec![n, n]);

    Ok((w_tensor, v_tensor))
}

/// Compute eigenvalues of a general tensor reshaped as a square matrix.
///
/// This is a convenience wrapper around [`eig`] that discards the eigenvectors.
///
/// # Arguments
///
/// * `tensor` - Input tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Eigenvalues: shape `[n]`, complex
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigvals<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<DenseTensor<T::Complex, B>, LinalgError> {
    let (w, _v) = eig(tensor, nrow)?;
    Ok(w)
}
