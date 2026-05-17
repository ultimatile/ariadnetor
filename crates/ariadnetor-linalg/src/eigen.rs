use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, EigDescriptor, EighDescriptor, ExecPolicy, MemoryOrder};
use arnet_tensor::{ComputeBackendTensorExt, Dense, DenseTensorData};
use num_traits::Zero;

use crate::error::LinalgError;
use arnet_tensor::reorder;

/// Result of a self-adjoint eigenvalue decomposition: `(eigenvalues, eigenvectors)`.
///
/// - Eigenvalues: `DenseTensorData<T::Real>` with shape `[n]`, sorted ascending
/// - Eigenvectors: `DenseTensorData<T>` with shape `[n, n]`, columns are eigenvectors
pub type EighResult<T> = (DenseTensorData<<T as Scalar>::Real>, DenseTensorData<T>);

/// Dense-typed internal counterpart of [`EighResult`].
pub type EighResultDense<T> = (Dense<<T as Scalar>::Real>, Dense<T>);

/// Compute self-adjoint eigenvalue decomposition of a tensor reshaped as a square matrix.
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigh<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<EighResult<T>, LinalgError> {
    let d = Dense::from_tensor_data(tensor.clone());
    let (w, v) = eigh_dense(backend, &d, nrow)?;
    Ok((w.into_tensor_data(), v.into_tensor_data()))
}

/// Dense-typed sister of [`eigh`].
pub fn eigh_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
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
pub fn eigh_with_policy<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<EighResult<T>, LinalgError> {
    let d = Dense::from_tensor_data(tensor.clone());
    let (w, v) = eigh_with_policy_dense(backend, &d, nrow, policy)?;
    Ok((w.into_tensor_data(), v.into_tensor_data()))
}

/// Dense-typed sister of [`eigh_with_policy`].
pub fn eigh_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
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
    let rm = reorder(tensor, tensor.order(), MemoryOrder::RowMajor);
    let mat_2d = Dense::new(rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor);
    let contiguous = reorder(&mat_2d, MemoryOrder::RowMajor, order);

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

    let w_tensor = Dense::new(w_data, vec![n], order);
    let v_tensor = backend.make_tensor(v_data, vec![n, n]);

    Ok((w_tensor, v_tensor))
}

/// Compute eigenvalues of a self-adjoint tensor reshaped as a square matrix.
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigvalsh<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<DenseTensorData<T::Real>, LinalgError> {
    let (w, _v) = eigh(backend, tensor, nrow)?;
    Ok(w)
}

/// Dense-typed sister of [`eigvalsh`].
pub fn eigvalsh_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<Dense<T::Real>, LinalgError> {
    let (w, _v) = eigh_dense(backend, tensor, nrow)?;
    Ok(w)
}

/// Result of a general eigenvalue decomposition: `(eigenvalues, eigenvectors)`.
pub type EigResult<T> = (
    DenseTensorData<<T as Scalar>::Complex>,
    DenseTensorData<<T as Scalar>::Complex>,
);

/// Dense-typed internal counterpart of [`EigResult`].
pub type EigResultDense<T> = (Dense<<T as Scalar>::Complex>, Dense<<T as Scalar>::Complex>);

/// Compute general eigenvalue decomposition of a tensor reshaped as a square matrix.
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eig<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<EigResult<T>, LinalgError> {
    let d = Dense::from_tensor_data(tensor.clone());
    let (w, v) = eig_dense(backend, &d, nrow)?;
    Ok((w.into_tensor_data(), v.into_tensor_data()))
}

/// Dense-typed sister of [`eig`].
pub fn eig_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
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
pub fn eig_with_policy<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<EigResult<T>, LinalgError> {
    let d = Dense::from_tensor_data(tensor.clone());
    let (w, v) = eig_with_policy_dense(backend, &d, nrow, policy)?;
    Ok((w.into_tensor_data(), v.into_tensor_data()))
}

/// Dense-typed sister of [`eig_with_policy`].
pub fn eig_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
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
    let rm = reorder(tensor, tensor.order(), MemoryOrder::RowMajor);
    let mat_2d = Dense::new(rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor);
    let contiguous = reorder(&mat_2d, MemoryOrder::RowMajor, order);

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

    let w_tensor = Dense::new(w_data, vec![n], order);
    let v_tensor = backend.make_tensor(v_data, vec![n, n]);

    Ok((w_tensor, v_tensor))
}

/// Compute eigenvalues of a general tensor reshaped as a square matrix.
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// or the backend fails.
pub fn eigvals<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<DenseTensorData<T::Complex>, LinalgError> {
    let (w, _v) = eig(backend, tensor, nrow)?;
    Ok(w)
}

/// Dense-typed sister of [`eigvals`].
pub fn eigvals_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<Dense<T::Complex>, LinalgError> {
    let (w, _v) = eig_dense(backend, tensor, nrow)?;
    Ok(w)
}
