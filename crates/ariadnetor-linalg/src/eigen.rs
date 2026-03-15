use arnet_core::backend::{BackendError, ComputeBackend, EigDescriptor, EighDescriptor};
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;
use num_traits::Zero;

/// Result of a self-adjoint eigenvalue decomposition: `(eigenvalues, eigenvectors)`.
///
/// - Eigenvalues: `DenseTensor<T::Real>` with shape `[n]`, sorted ascending
/// - Eigenvectors: `DenseTensor<T>` with shape `[n, n]`, columns are eigenvectors (row-major)
pub type EighResult<T> = (DenseTensor<<T as Scalar>::Real>, DenseTensor<T>);

/// Compute self-adjoint eigenvalue decomposition of a tensor reshaped as a square matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// The resulting matrix must be square.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// * Eigenvalues: shape `[n]`, real, sorted ascending
/// * Eigenvectors: shape `[n, n]`, columns are eigenvectors
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigh<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<EighResult<T>, BackendError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(BackendError::InvalidDimension(format!(
            "nrow must be in 1..rank, got nrow={nrow} for rank={rank}"
        )));
    }

    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();

    if m != n {
        return Err(BackendError::InvalidDimension(format!(
            "eigh requires a square matrix, got {m}×{n}"
        )));
    }

    let mut w_data = vec![T::Real::zero(); n];
    let mut v_data = vec![T::zero(); n * n];

    let desc = EighDescriptor {
        n,
        a: tensor.data(),
        w: &mut w_data,
        v: &mut v_data,
    };

    backend.eigh(desc)?;

    let w_tensor = DenseTensor::from_data(w_data, vec![n]);
    let v_tensor = DenseTensor::from_data(v_data, vec![n, n]);

    Ok((w_tensor, v_tensor))
}

/// Compute eigenvalues of a self-adjoint tensor reshaped as a square matrix.
///
/// This is a convenience wrapper around [`eigh`] that discards the eigenvectors.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Eigenvalues: shape `[n]`, real, sorted ascending
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigvalsh<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T::Real>, BackendError> {
    let (w, _v) = eigh(backend, tensor, nrow)?;
    Ok(w)
}

/// Result of a general eigenvalue decomposition: `(eigenvalues, eigenvectors)`.
///
/// - Eigenvalues: `DenseTensor<T::Complex>` with shape `[n]`, complex
/// - Eigenvectors: `DenseTensor<T::Complex>` with shape `[n, n]`, complex, columns are right eigenvectors (row-major)
pub type EigResult<T> = (DenseTensor<<T as Scalar>::Complex>, DenseTensor<<T as Scalar>::Complex>);

/// Compute general eigenvalue decomposition of a tensor reshaped as a square matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// The resulting matrix must be square. Eigenvalues and eigenvectors are always complex.
///
/// # Arguments
///
/// * `backend` - Compute backend
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
/// Returns `BackendError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eig<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<EigResult<T>, BackendError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(BackendError::InvalidDimension(format!(
            "nrow must be in 1..rank, got nrow={nrow} for rank={rank}"
        )));
    }

    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();

    if m != n {
        return Err(BackendError::InvalidDimension(format!(
            "eig requires a square matrix, got {m}×{n}"
        )));
    }

    let mut w_data = vec![T::Complex::zero(); n];
    let mut v_data = vec![T::Complex::zero(); n * n];

    let desc = EigDescriptor {
        n,
        a: tensor.data(),
        w: &mut w_data,
        v: &mut v_data,
    };

    backend.eig(desc)?;

    let w_tensor = DenseTensor::from_data(w_data, vec![n]);
    let v_tensor = DenseTensor::from_data(v_data, vec![n, n]);

    Ok((w_tensor, v_tensor))
}

/// Compute eigenvalues of a general tensor reshaped as a square matrix.
///
/// This is a convenience wrapper around [`eig`] that discards the eigenvectors.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Eigenvalues: shape `[n]`, complex
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigvals<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T::Complex>, BackendError> {
    let (w, _v) = eig(backend, tensor, nrow)?;
    Ok(w)
}
