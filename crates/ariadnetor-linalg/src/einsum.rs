//! General einsum dispatcher for N-input tensor operations.
//!
//! Dispatches to:
//! - [`crate::scalar_ops::trace`] + [`crate::transpose::transpose`] for single-tensor ops
//! - [`crate::contract::contract`] for two-tensor contractions
//! - Sequential pairwise contraction for 3+ tensor chains

use std::collections::{HashMap, HashSet};

use arnet_core::backend::{BackendError, ComputeBackend};
use arnet_core::einsum::EinsumExpr;
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;

use crate::contract::contract;
use crate::scalar_ops::trace;
use crate::transpose::transpose;

/// Execute an Einstein summation over N tensors.
///
/// Parses the notation and dispatches:
/// - **1 input**: trace (repeated indices not in output) + transpose (reorder)
/// - **2 inputs**: delegates to [`contract`] (supports Hadamard, batched GEMM)
/// - **N inputs** (N > 2): sequential left-to-right pairwise contraction
///
/// # Examples
///
/// ```rust,ignore
/// use arnet_linalg::einsum;
/// use arnet_native::NativeBackend;
///
/// let backend = NativeBackend::new();
/// // Matrix multiplication
/// let c = einsum(&backend, &[&a, &b], "ij,jk->ik")?;
/// // 3-tensor chain
/// let d = einsum(&backend, &[&a, &b, &c], "ij,jk,kl->il")?;
/// // Trace
/// let t = einsum(&backend, &[&m], "ii->")?;
/// ```
///
/// # Errors
///
/// Returns `BackendError` if notation is invalid, tensor count mismatches,
/// or any sub-operation fails.
pub fn einsum<T: Scalar>(
    backend: &impl ComputeBackend,
    tensors: &[&DenseTensor<T>],
    notation: &str,
) -> Result<DenseTensor<T>, BackendError> {
    let expr = EinsumExpr::parse(notation)
        .map_err(|e| BackendError::ExecutionFailed(format!("Failed to parse einsum: {e}")))?;

    if tensors.len() != expr.num_inputs() {
        return Err(BackendError::InvalidDimension(format!(
            "Notation requires {} input(s), got {}",
            expr.num_inputs(),
            tensors.len()
        )));
    }

    match expr.num_inputs() {
        0 => Err(BackendError::ExecutionFailed(
            "einsum requires at least 1 input".to_string(),
        )),
        1 => einsum_single(backend, tensors[0], &expr),
        2 => contract(backend, tensors[0], tensors[1], notation),
        _ => einsum_multi(backend, tensors, &expr),
    }
}

/// Sequential left-to-right pairwise contraction for 3+ tensors.
///
/// For each step, computes intermediate output indices as the intersection of
/// (accumulated ∪ next tensor indices) with (final output ∪ all future input indices).
fn einsum_multi<T: Scalar>(
    backend: &impl ComputeBackend,
    tensors: &[&DenseTensor<T>],
    expr: &EinsumExpr,
) -> Result<DenseTensor<T>, BackendError> {
    let inputs = expr.inputs();
    let final_output = expr.out_indices();
    let n = inputs.len();

    // First contraction: tensors[0] with tensors[1]
    let intermediate_out = compute_intermediate_output(&inputs[0], &inputs[1], &inputs[2..], final_output);
    let notation = build_notation(&inputs[0], &inputs[1], &intermediate_out);
    let mut accumulated = contract(backend, tensors[0], tensors[1], &notation)?;
    let mut acc_indices = intermediate_out;

    // Subsequent contractions: accumulated with tensors[i]
    for i in 2..n {
        let remaining = &inputs[i + 1..];
        let intermediate_out = compute_intermediate_output(&acc_indices, &inputs[i], remaining, final_output);
        let notation = build_notation(&acc_indices, &inputs[i], &intermediate_out);
        accumulated = contract(backend, &accumulated, tensors[i], &notation)?;
        acc_indices = intermediate_out;
    }

    Ok(accumulated)
}

/// Compute intermediate output indices for a pairwise contraction step.
///
/// Keeps indices from (lhs ∪ rhs) that appear in the final output or in any
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

    // Keep indices from lhs ∪ rhs that are needed, in order of first appearance
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
    tensor: &DenseTensor<T>,
    expr: &EinsumExpr,
) -> Result<DenseTensor<T>, BackendError> {
    let input = expr.lhs_indices();
    let output = expr.out_indices();

    if tensor.rank() != input.len() {
        return Err(BackendError::InvalidDimension(format!(
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
                    return Err(BackendError::ExecutionFailed(format!(
                        "Index '{}' appears once in input but not in output \
                         (general reduction not supported, use explicit trace pairs)",
                        idx as char
                    )));
                }
            }
            2 => {
                if output_set.contains(&idx) {
                    return Err(BackendError::ExecutionFailed(format!(
                        "Index '{}' appears twice in input and in output \
                         (diagonal extraction not yet supported)",
                        idx as char
                    )));
                }
                trace_pairs.push((pos[0], pos[1]));
            }
            n => {
                return Err(BackendError::ExecutionFailed(format!(
                    "Index '{}' appears {} times in input (max 2 allowed)",
                    idx as char, n
                )));
            }
        }
    }

    // Step 1: trace if needed
    let traced = if trace_pairs.is_empty() {
        tensor.clone()
    } else {
        trace(tensor, &trace_pairs)
            .map_err(|e| BackendError::ExecutionFailed(format!("Trace failed: {e}")))?
    };

    // Scalar result → done
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
        transpose(backend, &traced, &perm)
    }
}
