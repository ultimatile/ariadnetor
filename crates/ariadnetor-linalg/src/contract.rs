use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, GemmDescriptor, MemoryOrder};
use arnet_core::{ContractionPlan, EinsumExpr, compute_permutation};
use arnet_tensor::{ComputeBackendTensorExt, DenseTensorData};

use crate::contract_spec::validate_contract_notation;
use crate::error::LinalgError;
use crate::transpose::transpose_dense;
use arnet_tensor::{normalize_to_data, reorder_data};

/// Internal kernel for the pure tensor contraction on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn contract_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    lhs: &DenseTensorData<T>,
    rhs: &DenseTensorData<T>,
    notation: &str,
) -> Result<DenseTensorData<T>, LinalgError> {
    // Parse once up-front so we can compute the GEMM key (m, n, k) for
    // par_for_gemm. Parsing is not free, but the result is reused inside
    // contract_with_policy_dense via a re-parse; keeping the two paths
    // independent avoids plumbing parsed state through the expert signature.
    let expr = EinsumExpr::parse(notation)
        .map_err(|e| LinalgError::InvalidArgument(format!("Failed to parse einsum: {e}")))?;
    let plan = ContractionPlan::from_expr(&expr);

    let (m, n, k) = if plan.batch.is_empty()
        && lhs.rank() == expr.lhs_indices().len()
        && rhs.rank() == expr.rhs_indices().len()
    {
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
        (m, n, k)
    } else {
        (0, 0, 0)
    };

    let policy = backend.par_for_gemm(m, n, k);
    contract_with_policy_dense(backend, lhs, rhs, notation, policy)
}

/// Internal kernel for [`crate::expert::contract`] on the joined
/// [`DenseTensorData<T>`] form.
pub(crate) fn contract_with_policy_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    lhs: &DenseTensorData<T>,
    rhs: &DenseTensorData<T>,
    notation: &str,
    policy: ExecPolicy,
) -> Result<DenseTensorData<T>, LinalgError> {
    let expr = EinsumExpr::parse(notation)
        .map_err(|e| LinalgError::InvalidArgument(format!("Failed to parse einsum: {e}")))?;

    // Reject unsupported notation (wrong operand count, partial trace, implicit
    // reduction, batch) up front so it never reaches a downstream panic.
    validate_contract_notation(&expr)?;

    // Validate tensor ranks against notation
    if lhs.rank() != expr.lhs_indices().len() {
        return Err(LinalgError::InvalidArgument(format!(
            "LHS tensor rank {} doesn't match notation {}",
            lhs.rank(),
            expr.lhs_indices().len()
        )));
    }
    if rhs.rank() != expr.rhs_indices().len() {
        return Err(LinalgError::InvalidArgument(format!(
            "RHS tensor rank {} doesn't match notation {}",
            rhs.rank(),
            expr.rhs_indices().len()
        )));
    }

    let plan = ContractionPlan::from_expr(&expr);

    // Permute operands so indices are in [free, contracted] order.
    // These internal transposes self-tune via par_for_transpose (they are
    // preprocessing, not the main kernel).
    let lhs_perm = plan.lhs_permutation(expr.lhs_indices(), expr.rhs_indices());
    let rhs_perm = plan.rhs_permutation(expr.rhs_indices());

    let lhs_permuted = if let Some(perm) = lhs_perm {
        transpose_dense(backend, lhs, &perm)?
    } else {
        lhs.clone()
    };

    let rhs_permuted = if let Some(perm) = rhs_perm {
        transpose_dense(backend, rhs, &perm)?
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

    // Prepare operands for GEMM: reshape to 2D.
    // rank <= 2: no axis merge needed; `prepare_for_gemm` normalizes
    // the operand to `order` via `normalize_to_data` so a caller-supplied
    // tensor tagged in a different order is reordered at the boundary.
    // rank > 2: RowMajor reshape (correct axis merge semantics) is
    // required, then the reshaped tensor is reordered to `order`.
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
        policy,
    };
    backend.gemm(desc)?;

    // Output in preferred_order. For rank > 2, reconstruct multi-dimensional shape
    // via RowMajor 2D intermediate (correct axis split semantics).
    let output_shape = compute_output_shape(&plan, &expr, lhs.shape(), rhs.shape());
    let result = if output_shape.len() <= 2 {
        backend.make_tensor(c_data, output_shape)
    } else {
        // 2D preferred_order -> RowMajor 2D -> reshape to multi-dim -> preferred_order
        let result_2d = DenseTensorData::from_raw_parts(c_data, vec![m, n], order);
        let result_rm = reorder_data(&result_2d, MemoryOrder::RowMajor);
        let multi_dim = DenseTensorData::from_raw_parts(
            result_rm.data().to_vec(),
            output_shape,
            MemoryOrder::RowMajor,
        );
        reorder_data(&multi_dim, order)
    };

    // Reorder output dimensions to match the requested output index order.
    // GEMM produces [free_lhs, free_rhs]; the notation may request a different order.
    reorder_output(backend, result, &plan, &expr)
}

/// Reshape a permuted operand to 2D and convert to the target memory order.
///
/// For rank <= 2, no axis merge is needed — normalize the input to the
/// target order and return.
/// For rank > 2, reorder to RowMajor, reshape (correct axis merge), then back.
fn prepare_for_gemm<T: Scalar>(
    tensor: &DenseTensorData<T>,
    rows: usize,
    cols: usize,
    order: MemoryOrder,
) -> DenseTensorData<T> {
    if tensor.rank() <= 2 {
        // No axis merge needed; ensure data is in the target order before GEMM.
        normalize_to_data(tensor, order).into_owned()
    } else {
        // Axis merge requires RowMajor reshape, then convert to preferred_order
        let rm = reorder_data(tensor, MemoryOrder::RowMajor);
        let reshaped = rm.reshape(vec![rows, cols]);
        reorder_data(&reshaped, order)
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
    result: DenseTensorData<T>,
    plan: &ContractionPlan,
    expr: &EinsumExpr,
) -> Result<DenseTensorData<T>, LinalgError> {
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
        Some(perm) => transpose_dense(backend, &result, &perm),
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

    // Full contraction (no free indices) -> rank-0 tensor (shape []). A rank-0
    // dense tensor holds one element (the product over an empty shape is 1).
    output_shape
}
