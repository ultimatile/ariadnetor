use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder, SolveDescriptor};
use arnet_tensor::{ComputeBackendTensorExt, DenseTensorData};

use crate::error::LinalgError;
use arnet_tensor::reorder_data;

/// Internal kernel for the linear solve on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn solve_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &DenseTensorData<T>,
    b: &DenseTensorData<T>,
    nrow_a: usize,
) -> Result<DenseTensorData<T>, LinalgError> {
    // Extract key dims for par_for_solve; full validation occurs in solve_with_policy_dense.
    // If nrow_a is out of range, use a placeholder key — policy_by_n is defined for any
    // input, and solve_with_policy_dense will return the descriptive error.
    let (m, nrhs) = if nrow_a == 0 || nrow_a >= a.rank() {
        (0, 0)
    } else {
        let m: usize = a.shape()[..nrow_a].iter().product();
        let nrhs = b.len().checked_div(m).unwrap_or(0);
        (m, nrhs)
    };
    let policy = backend.par_for_solve(m, nrhs);
    solve_with_policy_dense(backend, a, b, nrow_a, policy)
}

/// Internal kernel for [`crate::expert::solve`] on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn solve_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    a: &DenseTensorData<T>,
    b: &DenseTensorData<T>,
    nrow_a: usize,
    policy: ExecPolicy,
) -> Result<DenseTensorData<T>, LinalgError> {
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
    let a_rm = reorder_data(a, MemoryOrder::RowMajor);
    let a_2d =
        DenseTensorData::from_raw_parts(a_rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor);
    let a_contiguous = reorder_data(&a_2d, order);

    let b_rm = reorder_data(b, MemoryOrder::RowMajor);
    let b_total = b_rm.len();

    if !b_total.is_multiple_of(n) {
        return Err(LinalgError::InvalidArgument(format!(
            "B total elements ({b_total}) must be divisible by n ({n})"
        )));
    }

    let nrhs = b_total / n;

    let b_2d =
        DenseTensorData::from_raw_parts(b_rm.data().to_vec(), vec![n, nrhs], MemoryOrder::RowMajor);
    let b_contiguous = reorder_data(&b_2d, order);

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
    let x_rm = reorder_data(&x_2d, MemoryOrder::RowMajor);
    let x_reshaped = DenseTensorData::from_raw_parts(
        x_rm.data().to_vec(),
        b.shape().to_vec(),
        MemoryOrder::RowMajor,
    );
    Ok(reorder_data(&x_reshaped, order))
}

/// Internal kernel for the matrix-inverse operation on the joined
/// [`DenseTensorData<T>`] form. Solves `AX = I` and returns `X = A^{-1}`.
pub(crate) fn inverse_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    nrow: usize,
) -> Result<DenseTensorData<T>, LinalgError> {
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
    let a_rm = reorder_data(tensor, MemoryOrder::RowMajor);
    let a_flat_rm =
        DenseTensorData::from_raw_parts(a_rm.data().to_vec(), vec![n, n], MemoryOrder::RowMajor);
    let a_flat = reorder_data(&a_flat_rm, order);

    let result = solve_dense(backend, &a_flat, &identity, 1)?;

    // solve() returns preferred_order data. RM intermediate for axis-split,
    // then back to preferred_order.
    let result_rm = reorder_data(&result, MemoryOrder::RowMajor);
    let reshaped = DenseTensorData::from_raw_parts(
        result_rm.data().to_vec(),
        shape.to_vec(),
        MemoryOrder::RowMajor,
    );
    Ok(reorder_data(&reshaped, order))
}
