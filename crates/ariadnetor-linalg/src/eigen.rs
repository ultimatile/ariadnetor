use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{
    ComputeBackend, EigDescriptor, EighDescriptor, ExecPolicy, MemoryOrder,
};
use ariadnetor_tensor::{ComputeBackendTensorExt, DenseTensor, DenseTensorData};
use num_traits::Zero;

use crate::error::LinalgError;
use crate::reorder_route::reorder_via_backend;

/// Result of a self-adjoint eigenvalue decomposition: `(eigenvalues, eigenvectors)`.
///
/// - Eigenvalues: `DenseTensor<T::Real>` with shape `[n]`, sorted ascending
/// - Eigenvectors: `DenseTensor<T>` with shape `[n, n]`, columns are eigenvectors
pub type EighResult<T> = (DenseTensor<<T as Scalar>::Real>, DenseTensor<T>);

/// Internal kernel form of [`EighResult`] operating on joined
/// [`DenseTensorData<T>`].
pub(crate) type EighResultDense<T> = (DenseTensorData<<T as Scalar>::Real>, DenseTensorData<T>);

/// Internal kernel for the self-adjoint eigenvalue decomposition on the
/// joined [`DenseTensorData<T>`] form.
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

/// Internal kernel for [`crate::expert::eigh`] on the joined
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
    let rm = reorder_via_backend(backend, tensor, MemoryOrder::RowMajor)?;
    let mat_2d =
        DenseTensorData::from_raw_parts(rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor);
    let contiguous = reorder_via_backend(backend, &mat_2d, order)?;

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

/// Result of a general eigenvalue decomposition: `(eigenvalues, eigenvectors)`.
///
/// - Eigenvalues: `DenseTensor<T::Complex>` with shape `[n]`, complex
/// - Eigenvectors: `DenseTensor<T::Complex>` with shape `[n, n]`, complex, columns are right eigenvectors
pub type EigResult<T> = (
    DenseTensor<<T as Scalar>::Complex>,
    DenseTensor<<T as Scalar>::Complex>,
);

/// Internal kernel form of [`EigResult`] operating on joined
/// [`DenseTensorData<T>`].
pub(crate) type EigResultDense<T> = (
    DenseTensorData<<T as Scalar>::Complex>,
    DenseTensorData<<T as Scalar>::Complex>,
);

/// Internal kernel for the general eigenvalue decomposition on the joined
/// [`DenseTensorData<T>`] form.
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

/// Internal kernel for [`crate::expert::eig`] on the joined
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
    let rm = reorder_via_backend(backend, tensor, MemoryOrder::RowMajor)?;
    let mat_2d =
        DenseTensorData::from_raw_parts(rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor);
    let contiguous = reorder_via_backend(backend, &mat_2d, order)?;

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
