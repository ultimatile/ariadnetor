//! Block-sparse tensor axis permutation.
//!
//! Permutes the leg order of a block-sparse tensor by reordering
//! indices, block coordinates, and transposing each block's data.

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_tensor::{BlockCoord, BlockSparseTensorData, Sector};

use crate::block_sparse_contract::transpose_block_data;
use crate::error::LinalgError;

/// Permute the axes of a block-sparse tensor.
///
/// `perm` maps new axis positions to old axis positions:
/// `new_axis[i] = old_axis[perm[i]]`.
///
/// The output tensor has the same flux and the same set of blocks, but with
/// reordered indices, coordinates, and transposed block data. The output
/// memory order equals the input memory order (`tensor.layout().order()`);
/// the operation does not consult `backend.preferred_order()`.
///
/// # Errors
///
/// Returns `LinalgError::InvalidArgument` if `perm` is not a valid
/// permutation of `0..tensor.rank()`.
pub fn permute_block_sparse<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensorData<T, S>,
    perm: &[usize],
) -> Result<BlockSparseTensorData<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let _ = backend;
    let order = tensor.layout().order();
    let rank = tensor.rank();

    // Validate permutation
    if perm.len() != rank {
        return Err(LinalgError::InvalidArgument(format!(
            "perm length {} != tensor rank {rank}",
            perm.len()
        )));
    }
    let mut seen = vec![false; rank];
    for (i, &p) in perm.iter().enumerate() {
        if p >= rank {
            return Err(LinalgError::InvalidArgument(format!(
                "perm[{i}] = {p} out of range for rank {rank}"
            )));
        }
        if seen[p] {
            return Err(LinalgError::InvalidArgument(format!(
                "perm contains duplicate axis {p}"
            )));
        }
        seen[p] = true;
    }

    // Identity permutation → clone
    if perm.iter().enumerate().all(|(i, &p)| p == i) {
        return Ok(tensor.clone());
    }

    let old_indices = tensor.indices().to_vec();

    // Permuted indices
    let new_indices = perm.iter().map(|&p| old_indices[p].clone()).collect();

    // Build output with zeros (establishes correct block structure)
    let mut output = BlockSparseTensorData::zeros(new_indices, tensor.flux().clone(), order);

    // Fill each block by transposing the corresponding input block's data
    for meta in tensor.block_metas() {
        // Permute block coordinate
        let new_coord_vec: Vec<usize> = perm.iter().map(|&p| meta.coord.0[p]).collect();
        let new_coord = BlockCoord(new_coord_vec);

        let src_data = tensor.block_data(&meta.coord).unwrap();
        let src_shape: Vec<usize> = (0..rank)
            .map(|a| old_indices[a].block_dim(meta.coord.0[a]))
            .collect();

        let transposed = transpose_block_data(src_data, &src_shape, perm, order);

        let dst_data = output
            .block_data_mut(&new_coord)
            .expect("permuted block must exist in output");
        dst_data.copy_from_slice(&transposed);
    }

    Ok(output)
}

#[cfg(test)]
mod tests;
