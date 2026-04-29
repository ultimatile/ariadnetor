use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, EigDescriptor, EighDescriptor, ExecPolicy, MemoryOrder};
use arnet_tensor::{ComputeBackendTensorExt, Dense};
use num_traits::Zero;

use crate::error::LinalgError;
use arnet_tensor::reorder;

/// Result of a self-adjoint eigenvalue decomposition: `(eigenvalues, eigenvectors)`.
///
/// - Eigenvalues: `Dense<T::Real>` with shape `[n]`, sorted ascending
/// - Eigenvectors: `Dense<T>` with shape `[n, n]`, columns are eigenvectors
pub type EighResult<T> = (Dense<<T as Scalar>::Real>, Dense<T>);

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
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigh<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<EighResult<T>, LinalgError> {
    let n = if nrow == 0 || nrow >= tensor.rank() {
        0
    } else {
        tensor.shape()[..nrow].iter().product()
    };
    let policy = backend.par_for_eigh(n);
    eigh_with_policy(backend, tensor, nrow, policy)
}

/// Self-adjoint eigenvalue decomposition with caller-specified execution policy.
///
/// Expert-layer counterpart of [`eigh`]; the default wrapper consults
/// `backend.par_for_eigh`, while this entry point takes `policy` directly.
pub fn eigh_with_policy<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<EighResult<T>, LinalgError> {
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
    let rm = reorder(tensor, order, MemoryOrder::RowMajor);
    let mat_2d = Dense::new(rm.data().to_vec(), vec![n, n]);
    let contiguous = reorder(&mat_2d, MemoryOrder::RowMajor, order);

    let mut w_data = vec![T::Real::zero(); n];
    let mut v_data = vec![T::zero(); n * n];

    let desc = EighDescriptor {
        n,
        a: contiguous.data(),
        w: &mut w_data,
        v: &mut v_data,
        policy,
    };

    backend.eigh(desc)?;

    let w_tensor = Dense::new(w_data, vec![n]);
    let v_tensor = backend.make_tensor(v_data, vec![n, n]);

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
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigvalsh<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<Dense<T::Real>, LinalgError> {
    let (w, _v) = eigh(backend, tensor, nrow)?;
    Ok(w)
}

/// Result of a general eigenvalue decomposition: `(eigenvalues, eigenvectors)`.
///
/// - Eigenvalues: `Dense<T::Complex>` with shape `[n]`, complex
/// - Eigenvectors: `Dense<T::Complex>` with shape `[n, n]`, complex, columns are right eigenvectors
pub type EigResult<T> = (Dense<<T as Scalar>::Complex>, Dense<<T as Scalar>::Complex>);

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
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eig<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<EigResult<T>, LinalgError> {
    let n = if nrow == 0 || nrow >= tensor.rank() {
        0
    } else {
        tensor.shape()[..nrow].iter().product()
    };
    let policy = backend.par_for_eig(n);
    eig_with_policy(backend, tensor, nrow, policy)
}

/// General eigenvalue decomposition with caller-specified execution policy.
///
/// Expert-layer counterpart of [`eig`]; the default wrapper consults
/// `backend.par_for_eig`, while this entry point takes `policy` directly.
pub fn eig_with_policy<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<EigResult<T>, LinalgError> {
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
    let rm = reorder(tensor, order, MemoryOrder::RowMajor);
    let mat_2d = Dense::new(rm.data().to_vec(), vec![n, n]);
    let contiguous = reorder(&mat_2d, MemoryOrder::RowMajor, order);

    let mut w_data = vec![T::Complex::zero(); n];
    let mut v_data = vec![T::Complex::zero(); n * n];

    let desc = EigDescriptor {
        n,
        a: contiguous.data(),
        w: &mut w_data,
        v: &mut v_data,
        policy,
    };

    backend.eig(desc)?;

    let w_tensor = Dense::new(w_data, vec![n]);
    let v_tensor = backend.make_tensor(v_data, vec![n, n]);

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
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigvals<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<Dense<T::Complex>, LinalgError> {
    let (w, _v) = eig(backend, tensor, nrow)?;
    Ok(w)
}
