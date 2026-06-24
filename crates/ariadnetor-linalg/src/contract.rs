use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, GemmDescriptor, MemoryOrder};
use arnet_core::{ContractionPlan, EinsumExpr, compute_permutation};
use arnet_tensor::{ComputeBackendTensorExt, DenseTensorData};

use crate::contract_spec::{validate_contract_notation, validate_contraction_axes_pair};
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
    // Reject unsupported notation before `ContractionPlan::from_expr`, which
    // assumes two operands (`rhs_indices()` panics otherwise). The policy-explicit
    // `contract_with_policy_dense` re-runs this guard for its own callers.
    validate_contract_notation(&expr)?;
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

    let gemm = DenseGemm {
        lhs_perm,
        rhs_perm,
        m,
        n,
        k,
        output_shape: compute_output_shape(&plan, &expr, lhs.shape(), rhs.shape()),
        policy,
    };
    let result = gemm_reshape_dense(backend, lhs, rhs, gemm)?;

    // Reorder output dimensions to match the requested output index order.
    // GEMM produces [free_lhs, free_rhs]; the notation may request a different order.
    reorder_output(backend, result, &plan, &expr)
}

/// Internal kernel for [`crate::LinalgContract::tensordot`] on the dense
/// [`DenseTensorData<T>`] form. Contracts `axes_lhs` against `axes_rhs` and emits
/// the output legs in their natural order (free left legs, then free right legs,
/// each in input axis order), with the GEMM execution policy auto-selected over
/// the reshaped `(m, n, k)`. A full contraction yields a rank-0 tensor.
///
/// The axis-native counterpart of [`contract_dense`]: it derives the contraction
/// partition directly from the axis pairs instead of from a parsed notation, so
/// no single-letter label budget bounds the operand rank. Both paths share
/// [`gemm_reshape_dense`] for the transpose / GEMM / reshape; this one needs no
/// final reorder because the natural leg order is exactly what the GEMM emits.
pub(crate) fn contract_axes_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    lhs: &DenseTensorData<T>,
    rhs: &DenseTensorData<T>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
) -> Result<DenseTensorData<T>, LinalgError> {
    let lhs_rank = lhs.rank();
    let rhs_rank = rhs.rank();
    validate_contraction_axes_pair(axes_lhs, lhs_rank, axes_rhs, rhs_rank)?;

    let free_lhs: Vec<usize> = (0..lhs_rank).filter(|a| !axes_lhs.contains(a)).collect();
    let free_rhs: Vec<usize> = (0..rhs_rank).filter(|a| !axes_rhs.contains(a)).collect();

    // perm[new_axis] = old_axis. Reorder lhs legs to [free, contracted] and rhs
    // legs to [contracted, free] so the GEMM reads them as (m, k) and (k, n).
    // An identity permutation skips the transpose.
    let lhs_perm = identity_or_perm(free_lhs.iter().chain(axes_lhs).copied().collect());
    let rhs_perm = identity_or_perm(axes_rhs.iter().chain(&free_rhs).copied().collect());

    let m: usize = free_lhs
        .iter()
        .map(|&a| lhs.shape()[a])
        .product::<usize>()
        .max(1);
    let n: usize = free_rhs
        .iter()
        .map(|&a| rhs.shape()[a])
        .product::<usize>()
        .max(1);
    let k: usize = axes_lhs
        .iter()
        .map(|&a| lhs.shape()[a])
        .product::<usize>()
        .max(1);

    // Natural tensordot output shape: free left dims then free right dims, each
    // in input axis order — the order the GEMM already produces, so no reorder.
    let mut output_shape = Vec::with_capacity(free_lhs.len() + free_rhs.len());
    output_shape.extend(free_lhs.iter().map(|&a| lhs.shape()[a]));
    output_shape.extend(free_rhs.iter().map(|&a| rhs.shape()[a]));

    let gemm = DenseGemm {
        lhs_perm,
        rhs_perm,
        m,
        n,
        k,
        output_shape,
        policy: backend.par_for_gemm(m, n, k),
    };
    gemm_reshape_dense(backend, lhs, rhs, gemm)
}

/// `None` when `perm` is the identity (no transpose needed), else `Some(perm)`.
fn identity_or_perm(perm: Vec<usize>) -> Option<Vec<usize>> {
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        None
    } else {
        Some(perm)
    }
}

/// Per-contraction GEMM recipe derived by either dense path (notation or
/// axis-pair) and consumed by [`gemm_reshape_dense`]. `lhs_perm` / `rhs_perm`
/// reorder each operand into GEMM order (`perm[new] = old`; `None` = already
/// ordered); `m` / `n` / `k` are the reshaped GEMM dimensions; `output_shape` is
/// the natural `[free_lhs, free_rhs]` result shape.
struct DenseGemm {
    lhs_perm: Option<Vec<usize>>,
    rhs_perm: Option<Vec<usize>>,
    m: usize,
    n: usize,
    k: usize,
    output_shape: Vec<usize>,
    policy: ExecPolicy,
}

/// Label-agnostic GEMM core shared by the notation and axis-pair dense kernels.
///
/// Permutes each operand into GEMM order, runs the reshaped `(m, k) · (k, n)`
/// GEMM, and reconstructs `output_shape` in the backend's preferred order. The
/// result legs are in natural `[free_lhs, free_rhs]` order; a notation caller
/// that requested a different order applies its own reorder afterward.
fn gemm_reshape_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    lhs: &DenseTensorData<T>,
    rhs: &DenseTensorData<T>,
    gemm: DenseGemm,
) -> Result<DenseTensorData<T>, LinalgError> {
    let DenseGemm {
        lhs_perm,
        rhs_perm,
        m,
        n,
        k,
        output_shape,
        policy,
    } = gemm;

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
    if output_shape.len() <= 2 {
        Ok(backend.make_tensor(c_data, output_shape))
    } else {
        // 2D preferred_order -> RowMajor 2D -> reshape to multi-dim -> preferred_order
        let result_2d = DenseTensorData::from_raw_parts(c_data, vec![m, n], order);
        let result_rm = reorder_data(&result_2d, MemoryOrder::RowMajor);
        let multi_dim = DenseTensorData::from_raw_parts(
            result_rm.data().to_vec(),
            output_shape,
            MemoryOrder::RowMajor,
        );
        Ok(reorder_data(&multi_dim, order))
    }
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
