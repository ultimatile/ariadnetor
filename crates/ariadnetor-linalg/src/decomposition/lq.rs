//! LQ decomposition pub fns + crate-internal `Dense<T>` kernels.

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, LqDescriptor};
use arnet_tensor::{ComputeBackendTensorExt, Dense, DenseTensor};

use super::reshape_for_backend;
use crate::error::LinalgError;
use crate::tensor_bridge::wrap_dense;

/// Result of a thin LQ decomposition: `(L, Q)`.
///
/// - `L`: Lower triangular matrix, shape `[m, k]` where `k = min(m, n)`
/// - `Q`: Orthogonal/unitary matrix, shape `[k, n]`
pub type LqResult<T, B> = (DenseTensor<T, B>, DenseTensor<T, B>);

/// Internal kernel form of [`LqResult`] operating on legacy `Dense<T>`.
pub(crate) type LqResultDense<T> = (Dense<T>, Dense<T>);

/// Compute thin LQ decomposition of a tensor reshaped as a matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// Returns `(L, Q)` where A = L * Q.
///
/// # Arguments
///
/// * `tensor` - Input tensor (backend flows from the tensor)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// * `L` - Lower triangular matrix, shape `[m, k]` where `m = product(shape[..nrow])`, `k = min(m, n)`
/// * `Q` - Orthogonal matrix, shape `[k, n]` where `n = product(shape[nrow..])`
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range or the backend fails.
pub fn lq<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
) -> Result<LqResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let (l, q) = lq_dense(tensor.backend(), &dense, nrow)?;
    Ok((
        wrap_dense(l, backend_arc.clone()),
        wrap_dense(q, backend_arc),
    ))
}

/// Internal kernel for [`lq`] on legacy `Dense<T>`.
pub(crate) fn lq_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
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

/// Thin LQ with caller-specified execution policy.
///
/// Expert-layer counterpart of [`lq`]; the default wrapper consults
/// `backend.par_for_lq`, while this entry point takes `policy` directly.
pub fn lq_with_policy<T: Scalar, B: ComputeBackend>(
    tensor: &DenseTensor<T, B>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<LqResult<T, B>, LinalgError> {
    let backend_arc = tensor.backend_arc().clone();
    let dense = tensor.data().as_dense();
    let (l, q) = lq_with_policy_dense(tensor.backend(), &dense, nrow, policy)?;
    Ok((
        wrap_dense(l, backend_arc.clone()),
        wrap_dense(q, backend_arc),
    ))
}

/// Internal kernel for [`lq_with_policy`] on legacy `Dense<T>`.
pub(crate) fn lq_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
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
