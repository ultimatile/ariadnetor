//! Block-sparse tensor axis permutation.
//!
//! Permutes the leg order of a `BlockSparseTensor<T, S>` tensor by
//! reordering indices, block coordinates, and transposing each block's data.

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::ComputeBackend;
use ariadnetor_tensor::{BlockCoord, BlockSparseTensorData, Sector};

use crate::block_sparse_contract::transpose_block_data;
use crate::error::LinalgError;
use crate::perm::validate_perm;

/// Permute the axes of a block-sparse tensor.
///
/// `perm` maps new axis positions to old axis positions:
/// `new_axis[i] = old_axis[perm[i]]`.
///
/// The output tensor has the same flux and the same set of blocks, but with
/// reordered indices, coordinates, and transposed block data.
///
/// # Errors
///
/// Returns `LinalgError::InvalidArgument` if `perm` is not a valid
/// permutation of `0..tensor.rank()`.
///
/// Internal kernel for the block-sparse permutation on joined-form
/// [`BlockSparseTensorData<T, S>`]. The public entry point is
/// [`crate::permute_block_sparse_with_backend`].
pub(crate) fn permute_block_sparse_dense<T, S, B>(
    backend: &B,
    tensor: &BlockSparseTensorData<T, S>,
    perm: &[usize],
) -> Result<BlockSparseTensorData<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let rank = tensor.layout().rank();

    validate_perm(perm, rank)?;

    // Identity permutation → clone
    if perm.iter().enumerate().all(|(i, &p)| p == i) {
        return Ok(tensor.clone());
    }

    let order = backend.preferred_order();
    let old_indices = tensor.layout().indices();

    // Permuted indices
    let new_indices = perm.iter().map(|&p| old_indices[p].clone()).collect();

    // Build output with zeros (establishes correct block structure)
    let mut output =
        BlockSparseTensorData::zeros(new_indices, tensor.layout().flux().clone(), order);

    // Fill each block by transposing the corresponding input block's data
    for meta in tensor.layout().block_metas() {
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
