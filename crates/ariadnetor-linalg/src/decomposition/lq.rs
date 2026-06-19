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

/// Internal kernel for the dense LQ with a caller-specified execution policy,
/// on the joined [`DenseTensorData<T>`] form. The public entry is the
/// layout-keyed [`expert::lq`](crate::expert::lq); the auto-policy entry
/// [`lq`](crate::lq) wraps [`lq_dense`], which consults `backend.par_for_lq`.
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
