//! LQ decomposition pub fns + crate-internal `DenseTensorData<T>` kernels.

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, LqDescriptor};
use arnet_tensor::{ComputeBackendTensorExt, DenseTensor, DenseTensorData};

use super::reshape_for_backend;
use crate::error::LinalgError;

/// Result of a thin LQ decomposition: `(L, Q)`.
///
/// - `L`: Lower triangular matrix, shape `[m, k]` where `k = min(m, n)`
/// - `Q`: Orthogonal/unitary matrix, shape `[k, n]`
pub type LqResult<T> = (DenseTensor<T>, DenseTensor<T>);

/// Internal kernel form of [`LqResult`] operating on joined
/// [`DenseTensorData<T>`].
pub(crate) type LqResultDense<T> = (DenseTensorData<T>, DenseTensorData<T>);

/// Internal kernel for the LQ operation on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn lq_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<LqResultDense<T>, LinalgError> {
    let (m, n) = if nrow == 0 || nrow >= tensor.rank() {
        (0, 0)
    } else {
        let m: usize = tensor.shape()[..nrow].iter().product();
        let n: usize = tensor.shape()[nrow..].iter().product();
        (m, n)
    };
    let policy = backend.par_for_lq(m, n);
    lq_with_policy_dense(backend, tensor, nrow, policy)
}

/// Thin LQ with an explicit backend and caller-specified execution policy.
///
/// Expert-layer counterpart of [`crate::lq_with_backend`]; that entry point
/// consults `backend.par_for_lq`, while this one takes `policy` directly. The
/// backend is supplied at the call site and the tensor's own backend is never
/// consulted.
pub fn lq_with_policy<T: Scalar, B: ComputeBackend>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<LqResult<T>, LinalgError> {
    let (l, q) = lq_with_policy_dense(backend, tensor.data(), nrow, policy)?;
    Ok((DenseTensor::from_data(l), DenseTensor::from_data(q)))
}

/// Internal kernel for [`lq_with_policy`] on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn lq_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<LqResultDense<T>, LinalgError> {
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

    let mut l_data = vec![T::zero(); m * k];
    let mut q_data = vec![T::zero(); k * n];

    let desc = LqDescriptor {
        m,
        n,
        a: mat_2d.data(),
        l: &mut l_data,
        q: &mut q_data,
        order,
        policy,
    };

    backend.lq(desc)?;

    let l_tensor = backend.make_tensor(l_data, vec![m, k]);
    let q_tensor = backend.make_tensor(q_data, vec![k, n]);

    Ok((l_tensor, q_tensor))
}
