use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder, SolveDescriptor};
use arnet_tensor::{ComputeBackendTensorExt, Dense};

use crate::error::LinalgError;
use arnet_tensor::reorder;

/// Solve the linear system AX = B via LU decomposition.
///
/// The input tensor `a` is reshaped as a square matrix by grouping the first
/// `nrow_a` axes as rows and the remaining axes as columns. The tensor `b`
/// must have compatible leading dimension (same number of rows as A).
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `a` - Coefficient tensor (must reshape to n x n square matrix)
/// * `b` - Right-hand side tensor (must have n rows when reshaped)
/// * `nrow_a` - Number of leading axes to group as rows for A
///
/// # Returns
///
/// Solution tensor X with the same shape as B (reshaped as n x nrhs).
///
/// # Errors
///
/// Returns `LinalgError` if `nrow_a` is out of range, the matrix A is non-square,
/// dimensions are incompatible, or the backend fails.
pub fn solve<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &Dense<T>,
    b: &Dense<T>,
    nrow_a: usize,
) -> Result<Dense<T>, LinalgError> {
    // Extract key dims for par_for_solve; full validation occurs in solve_with_policy.
    // If nrow_a is out of range, use a placeholder key — policy_by_n is defined for any
    // input, and solve_with_policy will return the descriptive error.
    let (m, nrhs) = if nrow_a == 0 || nrow_a >= a.rank() {
        (0, 0)
    } else {
        let m: usize = a.shape()[..nrow_a].iter().product();
        let nrhs = b.len().checked_div(m).unwrap_or(0);
        (m, nrhs)
    };
    let policy = backend.par_for_solve(m, nrhs);
    solve_with_policy(backend, a, b, nrow_a, policy)
}

/// Linear solve with caller-specified execution policy.
///
/// Expert-layer counterpart of [`solve`]; the default wrapper consults
/// `backend.par_for_solve`, while this entry point takes `policy` directly.
pub fn solve_with_policy<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &Dense<T>,
    b: &Dense<T>,
    nrow_a: usize,
    policy: ExecPolicy,
) -> Result<Dense<T>, LinalgError> {
    let a_shape = a.shape();
    let a_rank = a.rank();

    if nrow_a == 0 || nrow_a >= a_rank {
        return Err(LinalgError::InvalidArgument(format!(
            "nrow_a must satisfy 1 <= nrow_a < rank, got nrow_a={nrow_a} for rank={a_rank}"
        )));
    }

    let m: usize = a_shape[..nrow_a].iter().product();
    let n_a: usize = a_shape[nrow_a..].iter().product();

    if m != n_a {
        return Err(LinalgError::InvalidArgument(format!(
            "solve requires a square coefficient matrix, got {m}x{n_a}"
        )));
    }

    let n = m;
    let order = backend.preferred_order();

    // Ensure row-major reshape semantics, then convert to backend order
    let a_rm = reorder(a, a.order(), MemoryOrder::RowMajor);
    let a_2d = Dense::new(a_rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor);
    let a_contiguous = reorder(&a_2d, MemoryOrder::RowMajor, order);

    let b_rm = reorder(b, b.order(), MemoryOrder::RowMajor);
    let b_total = b_rm.len();

    if !b_total.is_multiple_of(n) {
        return Err(LinalgError::InvalidArgument(format!(
            "B total elements ({b_total}) must be divisible by n ({n})"
        )));
    }

    let nrhs = b_total / n;

    let b_2d = Dense::new(b_rm.data().to_vec(), vec![n, nrhs], MemoryOrder::RowMajor);
    let b_contiguous = reorder(&b_2d, MemoryOrder::RowMajor, order);

    let mut x_data = vec![T::zero(); n * nrhs];

    let desc = SolveDescriptor {
        n,
        nrhs,
        a: a_contiguous.data(),
        b: b_contiguous.data(),
        x: &mut x_data,
        order,
        policy,
    };

    backend.solve(desc)?;

    // x_data is in the backend's preferred order for an n x nrhs 2D buffer.
    // Convert to row-major for reshape (correct axis-merge semantics),
    // then reshape to b's original shape, then back to preferred order.
    let x_2d = backend.make_tensor(x_data, vec![n, nrhs]);
    let x_rm = reorder(&x_2d, order, MemoryOrder::RowMajor);
    let x_reshaped = Dense::new(
        x_rm.data().to_vec(),
        b.shape().to_vec(),
        MemoryOrder::RowMajor,
    );
    Ok(reorder(&x_reshaped, MemoryOrder::RowMajor, order))
}

/// Compute the inverse of a square matrix via LU decomposition.
///
/// Solves `AX = I` using [`solve`] and returns `X = A^{-1}`.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor (must reshape to a square matrix)
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// Inverse matrix with the same shape as the input (n x n).
///
/// # Errors
///
/// Returns `LinalgError` if `nrow` is out of range, the matrix is non-square,
/// singular, or the backend fails.
pub fn inverse<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &Dense<T>,
    nrow: usize,
) -> Result<Dense<T>, LinalgError> {
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
            "inverse requires a square matrix, got {m}x{n}"
        )));
    }

    let order = backend.preferred_order();

    // Identity matrix in `preferred_order` for use as RHS. The flat
    // layout of an identity is symmetric, so the choice of order only
    // affects the `order()` field; declaring `preferred_order()` lets
    // `solve` consume it without normalization.
    let identity = backend.eye::<T>(n);

    // Flatten tensor to n x n using RM reshape semantics, then convert to preferred_order.
    let a_rm = reorder(tensor, tensor.order(), MemoryOrder::RowMajor);
    let a_flat_rm = Dense::new(a_rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor);
    let a_flat = reorder(&a_flat_rm, MemoryOrder::RowMajor, order);

    let result = solve(backend, &a_flat, &identity, 1)?;

    // solve() returns preferred_order data. RM intermediate for axis-split,
    // then back to preferred_order.
    let result_rm = reorder(&result, order, MemoryOrder::RowMajor);
    let reshaped = Dense::new(
        result_rm.data().to_vec(),
        shape.to_vec(),
        MemoryOrder::RowMajor,
    );
    Ok(reorder(&reshaped, MemoryOrder::RowMajor, order))
}
