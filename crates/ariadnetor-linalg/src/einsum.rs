//! General einsum dispatcher for N-input tensor operations.
//!
//! Dispatches to:
//! - [`crate::scalar_ops::trace`] + [`crate::transpose::transpose`] for single-tensor ops
//! - [`einsum_pair`] for two-tensor operations (pure contraction, Hadamard, or batched)
//! - Sequential pairwise contraction for 3+ tensor chains

use std::collections::{HashMap, HashSet};

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_core::{ContractionPlan, EinsumExpr, compute_permutation};
use arnet_tensor::{ComputeBackendTensorExt, DenseTensor, DenseTensorData, normalize_to_data};

use crate::contract::contract_dense;
use crate::error::LinalgError;
use crate::scalar_ops::trace_dense;
use crate::transpose::transpose_dense;
use arnet_tensor::reorder_data;

/// Execute an Einstein summation over N tensors.
///
/// Parses the notation and dispatches:
/// - **1 input**: trace (repeated indices not in output) + transpose (reorder)
/// - **2 inputs**: delegates to [`einsum_pair`] (handles pure contraction, Hadamard, batched)
/// - **N inputs** (N > 2): sequential left-to-right pairwise contraction via [`einsum_pair`]
///
/// The backend is taken from `tensors[0]` and the result is wrapped against
/// `tensors[0]`'s backend Arc. Callers must ensure all inputs share the same
/// backend Arc; a mismatch silently runs on `tensors[0]`'s backend and labels
/// the output with `tensors[0]`'s backend, which is wrong for backends
/// carrying state.
///
/// # Examples
///
/// ```rust,ignore
/// use arnet_linalg::einsum;
///
/// // Matrix multiplication
/// let c = einsum(&[&a, &b], "ij,jk->ik")?;
/// // 3-tensor chain
/// let d = einsum(&[&a, &b, &c], "ij,jk,kl->il")?;
/// // Trace
/// let t = einsum(&[&m], "ii->")?;
/// ```
///
/// # Errors
///
/// Returns `LinalgError` if notation is invalid, tensor count mismatches,
/// or any sub-operation fails.
pub fn einsum<T: Scalar, B: ComputeBackend>(
    tensors: &[&DenseTensor<T, B>],
    notation: &str,
) -> Result<DenseTensor<T, B>, LinalgError> {
    if tensors.is_empty() {
        return Err(LinalgError::InvalidArgument(
            "einsum requires at least 1 input".to_string(),
        ));
    }
    let backend_arc = tensors[0].backend_arc().clone();
    let backend = tensors[0].backend();
    let data_refs: Vec<&DenseTensorData<T>> = tensors.iter().map(|t| t.data()).collect();
    let result = einsum_dense(backend, &data_refs, notation)?;
    Ok(DenseTensor::with_backend(result, backend_arc))
}

/// Internal kernel for [`einsum`] on joined [`DenseTensorData<T>`] slices.
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
/// - Pure contraction (no batch indices) -> [`contract`]
/// - Hadamard product (all batch, no free/contracted) -> element-wise multiply
/// - Batched contraction -> slice-and-loop over [`contract`]
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
        // Batched contraction: slice over batch dims, contract per slice
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

/// Batched contraction: loop over batch dimensions, call contract() per slice.
fn batched_contract<T: Scalar>(
    backend: &impl ComputeBackend,
    lhs: &DenseTensorData<T>,
    rhs: &DenseTensorData<T>,
    expr: &EinsumExpr,
    plan: &ContractionPlan,
) -> Result<DenseTensorData<T>, LinalgError> {
    let order = backend.preferred_order();

    // Compute batch dimensions
    let batch_dims: Vec<usize> = plan
        .batch
        .iter()
        .map(|&idx| dim_of(idx, expr.lhs_indices(), lhs.shape()))
        .collect();
    let batch_size: usize = batch_dims.iter().product::<usize>().max(1);

    // Build permutations to [batch..., free/contracted...] order
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

    // The no-permutation branch carries the caller's order tag, which
    // may differ from `order`. Normalize so the subsequent
    // `reorder_data(.., RowMajor)` reads the bytes under the
    // declared layout.
    let lhs_permuted = normalize_to_data(&lhs_permuted, order).into_owned();
    let rhs_permuted = normalize_to_data(&rhs_permuted, order).into_owned();

    // Reorder to RowMajor so batch slices are contiguous memory ranges
    let lhs_rm = reorder_data(&lhs_permuted, MemoryOrder::RowMajor);
    let rhs_rm = reorder_data(&rhs_permuted, MemoryOrder::RowMajor);

    // Per-slice sizes (after removing batch dimensions)
    let lhs_slice_size = lhs_rm.len() / batch_size;
    let rhs_slice_size = rhs_rm.len() / batch_size;

    // Slice shape: drop batch dims from the permuted shape
    let n_batch = plan.batch.len();
    let lhs_slice_shape: Vec<usize> = lhs_rm.shape()[n_batch..].to_vec();
    let rhs_slice_shape: Vec<usize> = rhs_rm.shape()[n_batch..].to_vec();

    // Contracted indices in RHS occurrence order (matching permutation order)
    let contracted_set: HashSet<u8> = plan.contracted.iter().copied().collect();
    let contracted_rhs_order: Vec<u8> = expr
        .rhs_indices()
        .iter()
        .filter(|idx| contracted_set.contains(idx))
        .copied()
        .collect();

    // Build batch-free notation with canonical [free_lhs, free_rhs] output order
    let batch_free_notation = build_batch_free_notation(plan, &contracted_rhs_order);

    let lhs_data = lhs_rm.data();
    let rhs_data = rhs_rm.data();

    // Contract each batch slice
    let mut result_slices: Vec<DenseTensorData<T>> = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let lhs_slice_data = &lhs_data[b * lhs_slice_size..(b + 1) * lhs_slice_size];
        let rhs_slice_data = &rhs_data[b * rhs_slice_size..(b + 1) * rhs_slice_size];

        // Slice data is in RowMajor layout; contract() expects data in
        // preferred_order, so reorder each slice before contracting.
        let lhs_slice_preferred = reorder_data(
            &DenseTensorData::from_raw_parts(
                lhs_slice_data.to_vec(),
                lhs_slice_shape.clone(),
                MemoryOrder::RowMajor,
            ),
            order,
        );
        let rhs_slice_preferred = reorder_data(
            &DenseTensorData::from_raw_parts(
                rhs_slice_data.to_vec(),
                rhs_slice_shape.clone(),
                MemoryOrder::RowMajor,
            ),
            order,
        );

        let slice_result = contract_dense(
            backend,
            &lhs_slice_preferred,
            &rhs_slice_preferred,
            &batch_free_notation,
        )?;
        result_slices.push(slice_result);
    }

    // Stack results into canonical shape [batch..., free_lhs..., free_rhs...]
    // Compute output shape from plan, not from contract() output (which adds a
    // dummy [1] for scalar results).
    let mut output_shape = batch_dims;
    for &idx in &plan.free_lhs {
        output_shape.push(dim_of(idx, expr.lhs_indices(), lhs.shape()));
    }
    for &idx in &plan.free_rhs {
        output_shape.push(dim_of(idx, expr.rhs_indices(), rhs.shape()));
    }
    let total_size: usize = output_shape.iter().product();

    let mut stacked_data: Vec<T> = Vec::with_capacity(total_size);
    for slice in &result_slices {
        // Reorder each slice result to RowMajor for consistent stacking
        let rm = reorder_data(slice, MemoryOrder::RowMajor);
        stacked_data.extend_from_slice(rm.data());
    }

    // Stacked data is in RowMajor; construct in RowMajor, then reorder to
    // preferred_order for the final result.
    let stacked_rm =
        DenseTensorData::from_raw_parts(stacked_data, output_shape, MemoryOrder::RowMajor);
    let stacked = reorder_data(&stacked_rm, order);

    // Reorder to requested output index order
    reorder_batched_output(
        backend,
        stacked,
        &plan.batch,
        &plan.free_lhs,
        &plan.free_rhs,
        expr.out_indices(),
    )
}

/// Build notation string with batch indices removed.
///
/// Uses `contracted_rhs_order` (contracted indices in RHS occurrence order) to
/// match the axis order produced by `lhs_permutation`/`rhs_permutation`, which
/// both place contracted indices in RHS occurrence order.
///
/// Output is always in canonical [free_lhs, free_rhs] order so that per-slice
/// contract() results stack consistently. The final reorder to the user's
/// requested output order happens after stacking.
fn build_batch_free_notation(plan: &ContractionPlan, contracted_rhs_order: &[u8]) -> String {
    let lhs_s: String = plan
        .free_lhs
        .iter()
        .chain(contracted_rhs_order.iter())
        .map(|&b| b as char)
        .collect();
    let rhs_s: String = contracted_rhs_order
        .iter()
        .chain(plan.free_rhs.iter())
        .map(|&b| b as char)
        .collect();
    let out_s: String = plan
        .free_lhs
        .iter()
        .chain(plan.free_rhs.iter())
        .map(|&b| b as char)
        .collect();
    format!("{lhs_s},{rhs_s}->{out_s}")
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
/// future input tensor, preserving the order of first appearance.
fn compute_intermediate_output(
    lhs: &[u8],
    rhs: &[u8],
    future_inputs: &[Vec<u8>],
    final_output: &[u8],
) -> Vec<u8> {
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
        reorder_data(&result_rm, order)
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
