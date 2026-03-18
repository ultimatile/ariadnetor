//! General einsum dispatcher for 1- and 2-input tensor operations.
//!
//! Dispatches to:
//! - [`crate::scalar_ops::trace`] + [`crate::transpose::transpose`] for single-tensor ops
//! - [`crate::contract::contract`] for two-tensor contractions

use std::collections::{HashMap, HashSet};

use arnet_core::backend::{BackendError, ComputeBackend};
use arnet_core::einsum::EinsumExpr;
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;

use crate::contract::contract;
use crate::scalar_ops::trace;
use crate::transpose::transpose;

/// Execute an Einstein summation over one or two tensors.
///
/// Parses the notation and dispatches:
/// - **1 input**: trace (repeated indices not in output) + transpose (reorder)
/// - **2 inputs**: delegates to [`contract`]
///
/// # Supported single-tensor patterns
///
/// | Notation | Operation |
/// |----------|-----------|
/// | `"ii->"` | Full trace |
/// | `"iij->j"` | Partial trace |
/// | `"ij->ji"` | Transpose |
/// | `"ijk->kji"` | Axis permutation |
/// | `"ijki->kj"` | Trace + transpose |
///
/// # Errors
///
/// Returns `BackendError` if:
/// - Notation is invalid
/// - Number of tensors doesn't match notation
/// - An index appears once in input but not in output (general reduction)
/// - An index appears twice in input and also in output (diagonal extraction)
/// - An index appears more than twice in a single input
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
        1 => einsum_single(backend, tensors[0], &expr),
        2 => contract(backend, tensors[0], tensors[1], notation),
        n => Err(BackendError::ExecutionFailed(format!(
            "einsum supports 1 or 2 inputs, got {n}"
        ))),
    }
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
