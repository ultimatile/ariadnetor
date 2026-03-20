use arnet_core::backend::{BackendError, ComputeBackend, MemoryOrder, SolveDescriptor};
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;

use crate::decomposition::make_tensor;

/// Solve the linear system AX = B via LU decomposition.
///
/// The input tensor `a` is reshaped as a square matrix by grouping the first
/// `nrow_a` axes as rows and the remaining axes as columns. The tensor `b`
/// must have compatible leading dimension (same number of rows as A).
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `a` - Coefficient tensor (must reshape to n×n square matrix)
/// * `b` - Right-hand side tensor (must have n rows when reshaped)
/// * `nrow_a` - Number of leading axes to group as rows for A
///
/// # Returns
///
/// Solution tensor X with the same shape as B (reshaped as n×nrhs).
///
/// # Errors
///
/// Returns `BackendError` if `nrow_a` is out of range, the matrix A is non-square,
/// dimensions are incompatible, or the backend fails.
pub fn solve<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &DenseTensor<T>,
    b: &DenseTensor<T>,
    nrow_a: usize,
) -> Result<DenseTensor<T>, BackendError> {
    let a_shape = a.shape();
    let a_rank = a.rank();

    if nrow_a == 0 || nrow_a >= a_rank {
        return Err(BackendError::InvalidDimension(format!(
            "nrow_a must be in 1..rank, got nrow_a={nrow_a} for rank={a_rank}"
        )));
    }

    let m: usize = a_shape[..nrow_a].iter().product();
    let n_a: usize = a_shape[nrow_a..].iter().product();

    if m != n_a {
        return Err(BackendError::InvalidDimension(format!(
            "solve requires a square coefficient matrix, got {m}×{n_a}"
        )));
    }

    let n = m;
    let order = backend.preferred_order();
    // Ensure row-major reshape semantics, then convert to backend order
    let a_rm = a.to_contiguous(MemoryOrder::RowMajor);
    let a_2d = DenseTensor::from_data(a_rm.data().to_vec(), vec![n, n]);
    let a_contiguous = a_2d.to_contiguous(order);

    let b_rm = b.to_contiguous(MemoryOrder::RowMajor);
    let b_total = b_rm.len();

    if !b_total.is_multiple_of(n) {
        return Err(BackendError::InvalidDimension(format!(
            "B total elements ({b_total}) must be divisible by n ({n})"
        )));
    }

    let nrhs = b_total / n;

    let b_2d = DenseTensor::from_data(b_rm.data().to_vec(), vec![n, nrhs]);
    let b_contiguous = b_2d.to_contiguous(order);

    let mut x_data = vec![T::zero(); n * nrhs];

    let desc = SolveDescriptor {
        n,
        nrhs,
        a: a_contiguous.data_contiguous(),
        b: b_contiguous.data_contiguous(),
        x: &mut x_data,
    };

    backend.solve(desc)?;

    // x_data is a column-major n×nrhs 2D buffer.
    // Convert to row-major 2D first, then reshape to b's original shape
    // to preserve standard unflatten semantics for higher-rank RHS.
    let x_2d = make_tensor(x_data, vec![n, nrhs], order);
    let x_rm = x_2d.to_contiguous(MemoryOrder::RowMajor);
    Ok(DenseTensor::from_data(
        x_rm.data().to_vec(),
        b.shape().to_vec(),
    ))
}

/// Compute the inverse of a square matrix via LU decomposition.
///
/// Solves `AX = I` using [`solve`] and returns `X = A⁻¹`.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Inverse matrix with the same shape as the input (n×n).
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range, the matrix is non-square,
/// singular, or the backend fails.
pub fn inverse<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<DenseTensor<T>, BackendError> {
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
            "inverse requires a square matrix, got {m}×{n}"
        )));
    }

    let identity = DenseTensor::<T>::eye(n);

    // Flatten to n×n and solve AX = I → X = A⁻¹
    let a_rm = tensor.to_contiguous(MemoryOrder::RowMajor);
    let a_flat = DenseTensor::from_data(a_rm.data().to_vec(), vec![n, n]);
    let result = solve(backend, &a_flat, &identity, 1)?;

    // Return in original shape, row-major (inverse output matches input convention)
    let result_rm = result.to_contiguous(MemoryOrder::RowMajor);
    Ok(DenseTensor::from_data(
        result_rm.data().to_vec(),
        shape.to_vec(),
    ))
}
