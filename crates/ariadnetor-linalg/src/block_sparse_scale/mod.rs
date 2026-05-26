//! Diagonal scaling for block-sparse tensors.
//!
//! Scales each slice along a specified axis by per-sector weights from
//! [`BlockSingularValues`], typically used to absorb singular values
//! after block-sparse SVD (e.g., S·Vt or U·S in MPS truncation).

use std::collections::HashMap;

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_tensor::BlockSparseTensor;
use arnet_tensor::BlockSparseTensorData;
use arnet_tensor::Sector;

use crate::block_sparse_decomp::BlockSingularValues;
use crate::error::LinalgError;

/// Scale each slice along `axis` by per-sector diagonal weights.
///
/// For each block, the sector at `axis` determines which weight vector
/// applies. Element `i` along the scaled axis is multiplied by `weights[i]`
/// for that sector.
///
/// Memory layout is determined by the backend's `preferred_order()`.
///
/// # Errors
///
/// Returns an error if `axis` is out of range, a block's sector is missing
/// from `weights`, or the weight vector length doesn't match the block
/// dimension at `axis`.
pub fn diagonal_scale_block_sparse<T, S, B>(
    tensor: &BlockSparseTensor<T, S, B>,
    weights: &BlockSingularValues<T::Real, S>,
    axis: usize,
) -> Result<BlockSparseTensor<T, S, B>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    crate::tensor_bridge::assert_bsp_layout_order_matches_backend(
        tensor,
        "diagonal_scale_block_sparse",
    );
    let backend_arc = tensor.backend_arc().clone();
    let result = diagonal_scale_block_sparse_dense(tensor.backend(), tensor.data(), weights, axis)?;
    Ok(BlockSparseTensor::with_backend(result, backend_arc))
}

/// Internal kernel for [`diagonal_scale_block_sparse`] on joined-form
/// [`BlockSparseTensorData<T, S>`].
pub(crate) fn diagonal_scale_block_sparse_dense<T, S>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    weights: &BlockSingularValues<T::Real, S>,
    axis: usize,
) -> Result<BlockSparseTensorData<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
{
    let rank = tensor.layout().rank();
    if axis >= rank {
        return Err(LinalgError::InvalidArgument(format!(
            "axis {axis} out of range for rank {rank}"
        )));
    }

    // Build sector → weight vector lookup.
    let weight_map: HashMap<&S, &Vec<T::Real>> =
        weights.values.iter().map(|(s, vs)| (s, vs)).collect();

    // Snapshot the block metadata we need before cloning the tensor.
    // The clone shares storage via Arc; mutating `result` triggers CoW
    // on first `block_data_mut`, leaving the original `tensor` intact
    // for the immutable inspection inside the loop.
    let block_metas = tensor.layout().block_metas().to_vec();
    let indices = tensor.layout().indices().to_vec();
    let mut result = tensor.clone();

    for meta in &block_metas {
        let block_idx_at_axis = meta.coord.0[axis];
        let sector = indices[axis].sector(block_idx_at_axis);

        let w = weight_map.get(sector).ok_or_else(|| {
            LinalgError::InvalidArgument(format!("no weights for sector {sector:?} at axis {axis}"))
        })?;

        let block_shape: Vec<usize> = meta
            .coord
            .0
            .iter()
            .enumerate()
            .map(|(a, &bi)| indices[a].block_dim(bi))
            .collect();

        let d_axis = block_shape[axis];
        if w.len() != d_axis {
            return Err(LinalgError::InvalidArgument(format!(
                "weight length {} doesn't match block dimension {} at axis {axis} for sector {sector:?}",
                w.len(),
                d_axis
            )));
        }

        // Stride for the scaled axis: product of trailing dims (RowMajor)
        // or product of preceding dims (ColumnMajor).
        let inner_stride: usize = match backend.preferred_order() {
            MemoryOrder::RowMajor => block_shape[axis + 1..].iter().product(),
            MemoryOrder::ColumnMajor => block_shape[..axis].iter().product(),
        };

        let data = result
            .block_data_mut(&meta.coord)
            .expect("block must exist in cloned tensor");

        for (idx, elem) in data.iter_mut().enumerate() {
            let i_axis = (idx / inner_stride) % d_axis;
            *elem = elem.scale_real(w[i_axis]);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests;
