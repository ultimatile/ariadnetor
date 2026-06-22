//! Block-sparse partial trace.
//!
//! Traces a `BlockSparseTensor<T, S>` over matched bond index pairs. Each pair
//! ties two mutually-dual legs on their shared diagonal; the result keeps the
//! non-paired legs in their original order and the input flux.
//!
//! # Why the traced legs must be mutually dual
//!
//! A leg pair `(a, b)` is traceable only when the two legs have identical
//! sector → dimension maps and opposite directions:
//!
//! - **Identical block structure** is what makes the diagonal well-defined. The
//!   global diagonal index runs over `[0, total_dim)`; at each step it must land
//!   on the same `(sector, intra-block offset)` on both legs. If the maps
//!   differ, the diagonal crosses sector boundaries and has no meaning. This is
//!   the block-sparse strengthening of the dense rule `shape[a] == shape[b]`.
//! - **Opposite directions** keep the pair's flux contribution at identity
//!   (`q` fused with `q.dual()`), so the trace preserves the tensor flux. With
//!   equal directions the contribution would be sector-dependent and the output
//!   could not carry a single flux.
//!
//! Only blocks whose paired legs sit at the same sector block carry diagonal
//! support; every such block is reduced by the dense per-block trace and
//! accumulated into its free-leg sub-coordinate, so several traced sectors fold
//! into one output block.

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_tensor::{BlockCoord, BlockSparseTensorData, DenseTensorData, Sector, reorder_data};

use crate::error::LinalgError;
use crate::scalar_ops::{trace_dense, validate_trace_pairs};

/// Internal kernel for the block-sparse partial trace on joined-form
/// [`BlockSparseTensorData<T, S>`]. The public entry point is
/// [`crate::trace_block_sparse_with_backend`].
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if a pair has an out-of-range,
/// self-paired, or reused bond index, or if the two paired legs do not have
/// identical sector blocks and opposite directions.
pub(crate) fn trace_block_sparse_dense<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensorData<T, S>,
    pairs: &[(usize, usize)],
) -> Result<BlockSparseTensorData<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    // Empty pairs: identity, matching the dense `trace` contract.
    if pairs.is_empty() {
        return Ok(tensor.clone());
    }

    let rank = tensor.layout().rank();
    let indices = tensor.layout().indices();

    // Shared range / self / reuse checks; the block-sparse rules — identical
    // sector blocks and opposite directions — are layout-specific and checked
    // here.
    let used = validate_trace_pairs(pairs, rank)?;
    for &(a, b) in pairs {
        if indices[a].blocks() != indices[b].blocks() {
            return Err(LinalgError::InvalidArgument(format!(
                "Traced pair ({a}, {b}) must have identical QN block structure"
            )));
        }
        if indices[a].direction() == indices[b].direction() {
            return Err(LinalgError::InvalidArgument(format!(
                "Traced pair ({a}, {b}) must have opposite leg directions"
            )));
        }
    }

    // Free legs: non-paired, original order. Output keeps the input flux.
    let free_axes: Vec<usize> = (0..rank).filter(|&i| !used[i]).collect();
    let free_indices: Vec<_> = free_axes.iter().map(|&i| indices[i].clone()).collect();
    let order = backend.preferred_order();

    let mut output =
        BlockSparseTensorData::zeros(free_indices, tensor.layout().flux().clone(), order);

    for meta in tensor.layout().block_metas() {
        // Diagonal support exists only where both legs of every pair sit at the
        // same sector block; other blocks contribute nothing to the trace.
        if !pairs
            .iter()
            .all(|&(a, b)| meta.coord.0[a] == meta.coord.0[b])
        {
            continue;
        }

        let block_shape: Vec<usize> = (0..rank)
            .map(|axis| indices[axis].block_dim(meta.coord.0[axis]))
            .collect();
        let block_data = tensor.block_data(&meta.coord).unwrap();
        let block = DenseTensorData::from_raw_parts(block_data.to_vec(), block_shape, order);

        // Reuse the dense per-block trace; it returns RowMajor data, so realign
        // to the storage order before accumulating into the packed buffer.
        let traced = trace_dense(&block, pairs)?;
        let traced = reorder_data(&traced, order);

        let free_coord = BlockCoord(free_axes.iter().map(|&i| meta.coord.0[i]).collect());
        let dst = output
            .block_data_mut(&free_coord)
            .expect("traced free-leg block must exist in output");
        for (d, s) in dst.iter_mut().zip(traced.data()) {
            *d = *d + *s;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests;
