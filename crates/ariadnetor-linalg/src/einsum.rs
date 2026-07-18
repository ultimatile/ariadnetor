//! General einsum dispatcher for N-input tensor operations.
//!
//! Dispatches to:
//! - [`trace_dense`] + [`transpose_dense`] for single-tensor ops
//! - [`einsum_pair`] for two-tensor operations (pure contraction, Hadamard, or batched)
//! - Sequential pairwise contraction for 3+ tensor chains

use std::collections::{HashMap, HashSet};

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{ComputeBackend, GemmDescriptor, MemoryOrder};
use ariadnetor_core::{ContractionPlan, EinsumExpr, compute_permutation};
use ariadnetor_tensor::{ComputeBackendTensorExt, DenseTensorData, normalize_to_data};

use crate::contract::contract_dense;
use crate::error::LinalgError;
use crate::reorder_route::reorder_via_backend;
use crate::scalar_ops::trace_dense;
use crate::transpose::transpose_dense;

/// Internal kernel for the N-input einsum on joined [`DenseTensorData<T>`]
/// slices. The public entry point is [`crate::einsum_with_backend`].
pub(crate) fn einsum_dense<T: Scalar>(
    backend: &impl ComputeBackend,
    tensors: &[&DenseTensorData<T>],
    notation: &str,
) -> Result<DenseTensorData<T>, LinalgError> {
    let expr = EinsumExpr::parse(notation)
        .map_err(|e| LinalgError::InvalidArgument(format!("Failed to parse einsum: {e}")))?;

    if tensors.len() != expr.num_inputs() {
        return Err(LinalgError::InvalidArgument(format!(
            "Notation requires {} input(s), got {}",
            expr.num_inputs(),
            tensors.len()
        )));
    }

    match expr.num_inputs() {
        0 => Err(LinalgError::InvalidArgument(
            "einsum requires at least 1 input".to_string(),
        )),
        1 => einsum_single(backend, tensors[0], &expr),
        2 => einsum_pair(backend, tensors[0], tensors[1], notation),
        _ => einsum_multi(backend, tensors, &expr),
    }
}

/// Dispatch a 2-input einsum to the appropriate path.
///
/// - Pure contraction (no batch indices) -> [`contract_dense`]
/// - Hadamard product (all batch, no free/contracted) -> element-wise multiply
/// - Batched contraction -> one GEMM per batch slice
fn einsum_pair<T: Scalar>(
    backend: &impl ComputeBackend,
    lhs: &DenseTensorData<T>,
    rhs: &DenseTensorData<T>,
    notation: &str,
) -> Result<DenseTensorData<T>, LinalgError> {
    let expr = EinsumExpr::parse(notation)
        .map_err(|e| LinalgError::InvalidArgument(format!("Failed to parse einsum: {e}")))?;

    let plan = ContractionPlan::from_expr(&expr);

    if plan.batch.is_empty() {
        // Pure contraction -- delegate to contract() (GEMM path)
        return contract_dense(backend, lhs, rhs, notation);
    }

    // Compute dimension sizes for each category
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

    if m == 1 && n == 1 && k == 1 {
        // Hadamard product: all indices are batch, no free or contracted
        hadamard(backend, lhs, rhs, &expr, &plan)
    } else {
        // Batched contraction: one GEMM per batch slice
        batched_contract(backend, lhs, rhs, &expr, &plan)
    }
}

/// Element-wise (Hadamard) product: all indices are batch.
fn hadamard<T: Scalar>(
    backend: &impl ComputeBackend,
    lhs: &DenseTensorData<T>,
    rhs: &DenseTensorData<T>,
    expr: &EinsumExpr,
    plan: &ContractionPlan,
) -> Result<DenseTensorData<T>, LinalgError> {
    // Permute both operands to [batch...] order (= output order for Hadamard)
    let lhs_perm = plan.lhs_permutation(expr.lhs_indices(), expr.rhs_indices());
    let rhs_perm = plan.rhs_permutation(expr.rhs_indices());

    let lhs_ordered = if let Some(perm) = lhs_perm {
        transpose_dense(backend, lhs, &perm)?
    } else {
        lhs.clone()
    };

    let rhs_ordered = if let Some(perm) = rhs_perm {
        transpose_dense(backend, rhs, &perm)?
    } else {
        rhs.clone()
    };

    // Both operands must share a common layout before element-wise zip:
    // when no permutation was applied, `*_ordered` carries the caller's
    // original `order()`, which may differ from the backend's preferred
    // order; normalize to `backend.preferred_order()` so the result we
    // tag below matches its data.
    let order = backend.preferred_order();
    let lhs_normalized = normalize_to_data(&lhs_ordered, order);
    let rhs_normalized = normalize_to_data(&rhs_ordered, order);

    let c_data: Vec<T> = lhs_normalized
        .data()
        .iter()
        .zip(rhs_normalized.data().iter())
        .map(|(&a, &b)| a * b)
        .collect();

    // Output shape in canonical [batch...] order, same layout as operands
    let output_shape: Vec<usize> = plan
        .batch
        .iter()
        .map(|&idx| dim_of(idx, expr.lhs_indices(), lhs.shape()))
        .collect();
    let result = backend.make_tensor(c_data, output_shape);

    // Reorder to requested output index order
    reorder_batched_output(backend, result, &plan.batch, &[], &[], expr.out_indices())
}

/// Batched contraction: one bare GEMM per batch slice.
///
/// Both operands are permuted and reordered once so the batch axes are the
/// slowest-varying in the backend's preferred order; every per-slice
/// `(m, k)` / `(k, n)` block is then a contiguous range fed straight to
/// [`ComputeBackend::gemm`], with the shared `(m, n, k)`, execution policy,
/// and output buffer computed once — so the per-slice loop itself performs no
/// copy, reorder, or notation re-parse. Ranks and shared batch/contracted
/// extents are validated up front, so the slice offsets are always in bounds.
fn batched_contract<T: Scalar>(
    backend: &impl ComputeBackend,
    lhs: &DenseTensorData<T>,
    rhs: &DenseTensorData<T>,
    expr: &EinsumExpr,
    plan: &ContractionPlan,
) -> Result<DenseTensorData<T>, LinalgError> {
    let order = backend.preferred_order();

    // Validate before slicing: `dim_of` would panic on a rank mismatch, and
    // an unchecked contracted/batch extent disagreement would let the
    // LHS-derived block boundaries index the wrong RHS region. Both are
    // reported as `InvalidArgument` rather than reaching a panic.
    if lhs.rank() != expr.lhs_indices().len() || rhs.rank() != expr.rhs_indices().len() {
        return Err(LinalgError::InvalidArgument(format!(
            "batched contraction: operand ranks {}/{} do not match notation arities {}/{}",
            lhs.rank(),
            rhs.rank(),
            expr.lhs_indices().len(),
            expr.rhs_indices().len(),
        )));
    }
    for &idx in plan.batch.iter().chain(&plan.contracted) {
        let lhs_dim = dim_of(idx, expr.lhs_indices(), lhs.shape());
        let rhs_dim = dim_of(idx, expr.rhs_indices(), rhs.shape());
        if lhs_dim != rhs_dim {
            return Err(LinalgError::InvalidArgument(format!(
                "batched contraction: shared index '{}' has mismatched extents {lhs_dim} != {rhs_dim}",
                idx as char
            )));
        }
    }

    // GEMM extents, unclamped: an empty free/contracted group is a scalar axis
    // (extent 1), while a genuine zero extent stays zero and drives the
    // degenerate branch below.
    let batch_dims = group_dims(&plan.batch, expr.lhs_indices(), lhs.shape());
    let batch_size: usize = batch_dims.iter().product();
    let m = group_extent(&plan.free_lhs, expr.lhs_indices(), lhs.shape());
    let n = group_extent(&plan.free_rhs, expr.rhs_indices(), rhs.shape());
    let k = group_extent(&plan.contracted, expr.lhs_indices(), lhs.shape());

    // Canonical output shape [batch..., free_lhs..., free_rhs...].
    let mut output_shape = batch_dims;
    output_shape.extend(group_dims(&plan.free_lhs, expr.lhs_indices(), lhs.shape()));
    output_shape.extend(group_dims(&plan.free_rhs, expr.rhs_indices(), rhs.shape()));

    // A zero extent on any GEMM axis (or an empty batch) is an empty sum: the
    // zero tensor of the output shape, reordered to the requested axes. Short-
    // circuit here rather than issuing degenerate GEMMs, mirroring
    // `gemm_reshape_dense`.
    if m == 0 || n == 0 || k == 0 || batch_size == 0 {
        let total: usize = output_shape.iter().product();
        let zeros = backend.make_tensor(vec![T::zero(); total], output_shape);
        return reorder_batched_output(
            backend,
            zeros,
            &plan.batch,
            &plan.free_lhs,
            &plan.free_rhs,
            expr.out_indices(),
        );
    }

    // Permute each operand to its GEMM order with batch axes first
    // (LHS `[batch, free_lhs, contracted]`, RHS `[batch, contracted, free_rhs]`),
    // then lay it out as contiguous per-slice blocks in `order`.
    let lhs_permuted = match plan.lhs_permutation(expr.lhs_indices(), expr.rhs_indices()) {
        Some(perm) => transpose_dense(backend, lhs, &perm)?,
        None => lhs.clone(),
    };
    let rhs_permuted = match plan.rhs_permutation(expr.rhs_indices()) {
        Some(perm) => transpose_dense(backend, rhs, &perm)?,
        None => rhs.clone(),
    };
    let lhs_slices = prepare_batched_operand(backend, &lhs_permuted, batch_size, m, k, order)?;
    let rhs_slices = prepare_batched_operand(backend, &rhs_permuted, batch_size, k, n, order)?;

    // One GEMM per slice into a preallocated buffer. `(m, n, k)` and the
    // execution policy are shared across slices by construction, so they are
    // computed once. `chunks_exact` yields exactly `batch_size` blocks because
    // each buffer is `batch_size` times the per-slice length.
    let policy = backend.par_for_gemm(m, n, k);
    let mut c_data = vec![T::zero(); batch_size * m * n];
    let lhs_data = lhs_slices.data();
    let rhs_data = rhs_slices.data();
    debug_assert_eq!(lhs_data.len(), batch_size * m * k);
    debug_assert_eq!(rhs_data.len(), batch_size * k * n);
    for ((a, b), c) in lhs_data
        .chunks_exact(m * k)
        .zip(rhs_data.chunks_exact(k * n))
        .zip(c_data.chunks_exact_mut(m * n))
    {
        backend.gemm(GemmDescriptor {
            m,
            n,
            k,
            alpha: T::one(),
            a,
            b,
            beta: T::zero(),
            c,
            trans_a: false,
            trans_b: false,
            order,
            policy,
        })?;
    }

    // `c_data` holds `batch_size` contiguous `(m, n)` blocks in `order`.
    // Reinterpret it as the canonical [batch, m, n] tensor (row-major keeps the
    // batch axis outermost; column-major put it trailing), split the merged
    // GEMM axes back out via a row-major reshape, and reorder to the requested
    // output index order.
    let c_batched = match order {
        MemoryOrder::RowMajor => {
            DenseTensorData::from_raw_parts(c_data, vec![batch_size, m, n], MemoryOrder::RowMajor)
        }
        MemoryOrder::ColumnMajor => {
            let laid = DenseTensorData::from_raw_parts(
                c_data,
                vec![m, n, batch_size],
                MemoryOrder::ColumnMajor,
            );
            transpose_dense(backend, &laid, &[2, 0, 1])?
        }
    };
    let c_rm = reorder_via_backend(backend, &c_batched, MemoryOrder::RowMajor)?;
    let c_full = reorder_via_backend(backend, &c_rm.reshape(output_shape), order)?;
    reorder_batched_output(
        backend,
        c_full,
        &plan.batch,
        &plan.free_lhs,
        &plan.free_rhs,
        expr.out_indices(),
    )
}

/// Reorder a permuted operand into contiguous per-slice GEMM blocks.
///
/// `permuted` is logically `[batch..., rows..., cols...]`. The result lays the
/// `batch_size` blocks of shape `(rows, cols)` back to back, each contiguous in
/// `order`, so slice `s` is exactly `data[s * rows * cols .. (s + 1) * rows *
/// cols]`. Row-major keeps the batch axis outermost; column-major rolls it to
/// the trailing (slowest) axis, since only there is each slice contiguous.
fn prepare_batched_operand<T: Scalar>(
    backend: &impl ComputeBackend,
    permuted: &DenseTensorData<T>,
    batch_size: usize,
    rows: usize,
    cols: usize,
    order: MemoryOrder,
) -> Result<DenseTensorData<T>, LinalgError> {
    // Row-major reshape merges the free / contracted axis groups correctly.
    let rm = reorder_via_backend(backend, permuted, MemoryOrder::RowMajor)?;
    let three_d = rm.reshape(vec![batch_size, rows, cols]);
    match order {
        MemoryOrder::RowMajor => Ok(three_d),
        MemoryOrder::ColumnMajor => {
            let rolled = transpose_dense(backend, &three_d, &[1, 2, 0])?;
            reorder_via_backend(backend, &rolled, MemoryOrder::ColumnMajor)
        }
    }
}

/// Reorder output from canonical [batch..., free_lhs..., free_rhs...] to requested order.
fn reorder_batched_output<T: Scalar>(
    backend: &impl ComputeBackend,
    result: DenseTensorData<T>,
    batch: &[u8],
    free_lhs: &[u8],
    free_rhs: &[u8],
    out: &[u8],
) -> Result<DenseTensorData<T>, LinalgError> {
    if out.is_empty() {
        return Ok(result);
    }

    let mut actual: Vec<u8> = Vec::with_capacity(out.len());
    actual.extend(batch);
    actual.extend(free_lhs);
    actual.extend(free_rhs);

    match compute_permutation(&actual, out) {
        Some(perm) => transpose_dense(backend, &result, &perm),
        None => Ok(result),
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

/// Extents of `group`'s indices, each looked up in `indices` / `shape`.
fn group_dims(group: &[u8], indices: &[u8], shape: &[usize]) -> Vec<usize> {
    group
        .iter()
        .map(|&idx| dim_of(idx, indices, shape))
        .collect()
}

/// Product of the extents of `group`'s indices. An empty group is the empty
/// product `1` (a scalar axis); a zero-extent index yields `0`. Kept separate
/// from [`group_dims`] so the hot `m` / `n` / `k` path allocates no `Vec`.
fn group_extent(group: &[u8], indices: &[u8], shape: &[usize]) -> usize {
    group
        .iter()
        .map(|&idx| dim_of(idx, indices, shape))
        .product()
}

/// Sequential left-to-right pairwise contraction for 3+ tensors.
///
/// For each step, computes intermediate output indices as the intersection of
/// (accumulated union next tensor indices) with (final output union all future input indices).
fn einsum_multi<T: Scalar>(
    backend: &impl ComputeBackend,
    tensors: &[&DenseTensorData<T>],
    expr: &EinsumExpr,
) -> Result<DenseTensorData<T>, LinalgError> {
    let inputs = expr.inputs();
    let final_output = expr.out_indices();
    let n = inputs.len();

    // First contraction: tensors[0] with tensors[1]
    let intermediate_out =
        compute_intermediate_output(&inputs[0], &inputs[1], &inputs[2..], final_output);
    let notation = build_notation(&inputs[0], &inputs[1], &intermediate_out);
    let mut accumulated = einsum_pair(backend, tensors[0], tensors[1], &notation)?;
    let mut acc_indices = intermediate_out;

    // Subsequent contractions: accumulated with tensors[i]
    for i in 2..n {
        let remaining = &inputs[i + 1..];
        let intermediate_out =
            compute_intermediate_output(&acc_indices, &inputs[i], remaining, final_output);
        let notation = build_notation(&acc_indices, &inputs[i], &intermediate_out);
        accumulated = einsum_pair(backend, &accumulated, tensors[i], &notation)?;
        acc_indices = intermediate_out;
    }

    Ok(accumulated)
}

/// Compute intermediate output indices for a pairwise contraction step.
///
/// Keeps indices from (lhs union rhs) that appear in the final output or in any
/// future input tensor, preserving the order of first appearance. The LAST
/// step (no future inputs) instead emits the requested output order itself:
/// first-appearance order is only an internal labeling, and no reorder runs
/// after the pairwise loop, so the final step's notation must carry the
/// caller's order. The index sets coincide either way — every surviving
/// index at the last step is a final-output index and vice versa (parse
/// validation rejects output indices absent from the inputs) — and
/// `einsum_pair` handles arbitrary output orders.
fn compute_intermediate_output(
    lhs: &[u8],
    rhs: &[u8],
    future_inputs: &[Vec<u8>],
    final_output: &[u8],
) -> Vec<u8> {
    if future_inputs.is_empty() {
        return final_output.to_vec();
    }

    // Indices that must survive this contraction step
    let mut needed: HashSet<u8> = final_output.iter().copied().collect();
    for future in future_inputs {
        needed.extend(future.iter().copied());
    }

    // Keep indices from lhs union rhs that are needed, in order of first appearance
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &idx in lhs.iter().chain(rhs.iter()) {
        if needed.contains(&idx) && seen.insert(idx) {
            out.push(idx);
        }
    }
    out
}

/// Build a 2-input einsum notation string from index slices.
fn build_notation(lhs: &[u8], rhs: &[u8], output: &[u8]) -> String {
    let lhs_s: String = lhs.iter().map(|&b| b as char).collect();
    let rhs_s: String = rhs.iter().map(|&b| b as char).collect();
    let out_s: String = output.iter().map(|&b| b as char).collect();
    format!("{lhs_s},{rhs_s}->{out_s}")
}

/// Dispatch a single-tensor einsum to trace and/or transpose.
fn einsum_single<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    expr: &EinsumExpr,
) -> Result<DenseTensorData<T>, LinalgError> {
    let input = expr.lhs_indices();
    let output = expr.out_indices();

    if tensor.rank() != input.len() {
        return Err(LinalgError::InvalidArgument(format!(
            "Tensor rank {} doesn't match notation {}",
            tensor.rank(),
            input.len()
        )));
    }

    // Group each index by its positions in the input
    let mut positions: HashMap<u8, Vec<usize>> = HashMap::new();
    for (i, &idx) in input.iter().enumerate() {
        positions.entry(idx).or_default().push(i);
    }

    let output_set: HashSet<u8> = output.iter().copied().collect();
    let mut trace_pairs = Vec::new();
    let mut free_positions: Vec<(usize, u8)> = Vec::new(); // (original axis, index)

    for (&idx, pos) in &positions {
        match pos.len() {
            1 => {
                if output_set.contains(&idx) {
                    free_positions.push((pos[0], idx));
                } else {
                    return Err(LinalgError::InvalidArgument(format!(
                        "Index '{}' appears once in input but not in output \
                         (general reduction not supported, use explicit trace pairs)",
                        idx as char
                    )));
                }
            }
            2 => {
                if output_set.contains(&idx) {
                    return Err(LinalgError::InvalidArgument(format!(
                        "Index '{}' appears twice in input and in output \
                         (diagonal extraction not yet supported)",
                        idx as char
                    )));
                }
                trace_pairs.push((pos[0], pos[1]));
            }
            n => {
                return Err(LinalgError::InvalidArgument(format!(
                    "Index '{}' appears {} times in input (max 2 allowed)",
                    idx as char, n
                )));
            }
        }
    }

    // Step 1: trace if needed.
    // `trace` normalizes its input to RowMajor internally and tags the
    // result RowMajor; convert the result to `backend.preferred_order()`
    // so downstream `transpose` operates on the layout the backend
    // kernels expect.
    let order = backend.preferred_order();
    let traced = if trace_pairs.is_empty() {
        tensor.clone()
    } else {
        let result_rm = trace_dense(tensor, &trace_pairs)?;
        reorder_via_backend(backend, &result_rm, order)?
    };

    // Scalar result -> done
    if free_positions.is_empty() {
        return Ok(traced);
    }

    // Step 2: compute permutation from trace-output order to desired output order
    // After trace, the result axes correspond to the free positions sorted by
    // their original axis index.
    free_positions.sort_by_key(|&(pos, _)| pos);
    let trace_order: Vec<u8> = free_positions.iter().map(|&(_, idx)| idx).collect();

    let perm: Vec<usize> = output
        .iter()
        .map(|&idx| {
            trace_order
                .iter()
                .position(|&x| x == idx)
                .expect("Output index not in free indices")
        })
        .collect();

    let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);

    if is_identity {
        Ok(traced)
    } else {
        transpose_dense(backend, &traced, &perm)
    }
}

#[cfg(test)]
mod tests;
