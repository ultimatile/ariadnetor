use arnet_core::backend::{BackendError, ComputeBackend, GemmDescriptor, MemoryOrder};
use arnet_core::einsum::{ContractionPlan, EinsumExpr, compute_permutation};
use arnet_core::scalar::Scalar;
use arnet_tensor::{ComputeBackendTensorExt, DenseTensor};

use crate::transpose::transpose;

/// Contract two tensors using Einstein summation notation with the provided backend.
///
/// Performs a pure tensor contraction: all shared indices between the two inputs
/// must be contracted (summed over). Batch indices (shared but not contracted)
/// are not supported — use [`crate::einsum::einsum`] for expressions with batch
/// or Hadamard patterns.
///
/// Output is returned in `backend.preferred_order()`, consistent with
/// decomposition functions. Callers that need RowMajor (e.g., for reshape
/// with correct axis merge semantics) should call `to_contiguous(RowMajor)`.
///
/// # Arguments
///
/// * `backend` - Compute backend for transpose and GEMM operations
/// * `lhs` - Left-hand side tensor
/// * `rhs` - Right-hand side tensor
/// * `notation` - Einstein summation notation (e.g., "ik,kj->ij")
///
/// # Errors
///
/// Returns `BackendError` if notation is invalid, dimensions mismatch,
/// batch indices are present, or the backend fails to execute transpose/GEMM.
pub fn contract<T: Scalar>(
    backend: &impl ComputeBackend,
    lhs: &DenseTensor<T>,
    rhs: &DenseTensor<T>,
    notation: &str,
) -> Result<DenseTensor<T>, BackendError> {
    let expr = EinsumExpr::parse(notation)
        .map_err(|e| BackendError::ExecutionFailed(format!("Failed to parse einsum: {e}")))?;

    // Validate tensor ranks against notation
    if lhs.rank() != expr.lhs_indices().len() {
        return Err(BackendError::InvalidArgument(format!(
            "LHS tensor rank {} doesn't match notation {}",
            lhs.rank(),
            expr.lhs_indices().len()
        )));
    }
    if rhs.rank() != expr.rhs_indices().len() {
        return Err(BackendError::InvalidArgument(format!(
            "RHS tensor rank {} doesn't match notation {}",
            rhs.rank(),
            expr.rhs_indices().len()
        )));
    }

    let plan = ContractionPlan::from_expr(&expr);

    // Reject batch indices — batch/Hadamard belongs in the einsum layer
    if !plan.batch.is_empty() {
        return Err(BackendError::ExecutionFailed(format!(
            "contract() does not support batch indices {:?}; use einsum() instead",
            plan.batch.iter().map(|&b| b as char).collect::<String>()
        )));
    }

    // Permute operands so indices are in [free, contracted] order
    let lhs_perm = plan.lhs_permutation(expr.lhs_indices(), expr.rhs_indices());
    let rhs_perm = plan.rhs_permutation(expr.rhs_indices());

    let lhs_permuted = if let Some(perm) = lhs_perm {
        transpose(backend, lhs, &perm)?
    } else {
        lhs.clone()
    };

    let rhs_permuted = if let Some(perm) = rhs_perm {
        transpose(backend, rhs, &perm)?
    } else {
        rhs.clone()
    };

    let m: usize = plan
        .free_lhs
        .iter()
        .map(|&idx| dim_of(idx, expr.lhs_indices(), lhs.shape()))
        .product::<usize>()
        .max(1);

    let n: usize = plan
        .free_rhs
        .iter()
        .map(|&idx| dim_of(idx, expr.rhs_indices(), rhs.shape()))
        .product::<usize>()
        .max(1);

    let k: usize = plan
        .contracted
        .iter()
        .map(|&idx| dim_of(idx, expr.lhs_indices(), lhs.shape()))
        .product::<usize>()
        .max(1);

    let order = backend.preferred_order();

    // Prepare operands for GEMM: reshape to 2D and convert to preferred_order.
    // rank <= 2: no axis merge needed, go directly to preferred_order.
    // rank > 2: RowMajor reshape (correct axis merge semantics), then preferred_order.
    let lhs_ready = prepare_for_gemm(&lhs_permuted, m, k, order);
    let rhs_ready = prepare_for_gemm(&rhs_permuted, k, n, order);

    let mut c_data = vec![T::zero(); m * n];

    let desc = GemmDescriptor {
        m,
        n,
        k,
        alpha: T::one(),
        a: lhs_ready.data(),
        b: rhs_ready.data(),
        beta: T::zero(),
        c: &mut c_data,
        trans_a: false,
        trans_b: false,
        order,
    };
    backend.gemm(desc)?;

    // Output in preferred_order. For rank > 2, reconstruct multi-dimensional shape
    // via RowMajor 2D intermediate (correct axis split semantics).
    let output_shape = compute_output_shape(&plan, &expr, lhs.shape(), rhs.shape());
    let result = if output_shape.len() <= 2 {
        backend.make_tensor(c_data, output_shape)
    } else {
        // 2D preferred_order → RowMajor 2D → reshape to multi-dim → preferred_order
        let result_2d = backend.make_tensor(c_data, vec![m, n]);
        let result_rm = result_2d.to_contiguous(MemoryOrder::RowMajor);
        DenseTensor::from_data_with_order(
            result_rm.data().to_vec(),
            output_shape,
            MemoryOrder::RowMajor,
        )
        .to_contiguous(order)
    };

    // Reorder output dimensions to match the requested output index order.
    // GEMM produces [free_lhs, free_rhs]; the notation may request a different order.
    reorder_output(backend, result, &plan, &expr)
}

/// Reshape a permuted operand to 2D and convert to the target memory order.
///
/// For rank <= 2, no axis merge is needed — directly convert to target order.
/// For rank > 2, RowMajor reshape ensures correct axis merge semantics.
fn prepare_for_gemm<T: Scalar>(
    tensor: &DenseTensor<T>,
    rows: usize,
    cols: usize,
    order: MemoryOrder,
) -> DenseTensor<T> {
    if tensor.rank() <= 2 {
        // No axis merge: directly convert to preferred_order (zero-copy if already correct)
        tensor.to_contiguous(order)
    } else {
        // Axis merge requires RowMajor reshape, then convert to preferred_order
        let rm = tensor.to_contiguous(MemoryOrder::RowMajor);
        let reshaped = rm.reshape(vec![rows, cols]);
        reshaped.to_contiguous(order)
    }
}

/// Look up the dimension of an index in the given index list and shape.
fn dim_of(idx: u8, indices: &[u8], shape: &[usize]) -> usize {
    let pos = indices
        .iter()
        .position(|&x| x == idx)
        .expect("Index not found in tensor");
    shape[pos]
}

/// Reorder output dimensions from [free_lhs, free_rhs] to the requested output order.
fn reorder_output<T: Scalar>(
    backend: &impl ComputeBackend,
    result: DenseTensor<T>,
    plan: &ContractionPlan,
    expr: &EinsumExpr,
) -> Result<DenseTensor<T>, BackendError> {
    let out = expr.out_indices();
    if out.is_empty() {
        // Scalar result — no reordering needed
        return Ok(result);
    }

    // Actual output index order produced by GEMM (no batch)
    let mut actual: Vec<u8> = Vec::with_capacity(out.len());
    actual.extend(&plan.free_lhs);
    actual.extend(&plan.free_rhs);

    match compute_permutation(&actual, out) {
        Some(perm) => transpose(backend, &result, &perm),
        None => Ok(result),
    }
}

/// Derive output tensor shape from contraction plan and original input shapes.
fn compute_output_shape(
    plan: &ContractionPlan,
    expr: &EinsumExpr,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Vec<usize> {
    let mut output_shape = Vec::new();

    for &idx in &plan.free_lhs {
        output_shape.push(dim_of(idx, expr.lhs_indices(), lhs_shape));
    }

    for &idx in &plan.free_rhs {
        output_shape.push(dim_of(idx, expr.rhs_indices(), rhs_shape));
    }

    // Scalar result (no free indices) → shape [1]
    if output_shape.is_empty() {
        output_shape.push(1);
    }

    output_shape
}
