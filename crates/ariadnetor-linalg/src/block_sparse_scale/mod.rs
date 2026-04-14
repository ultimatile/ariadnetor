//! Diagonal scaling for block-sparse tensors.
//!
//! Scales each slice along a specified axis by per-sector weights from
//! [`BlockSingularValues`], typically used to absorb singular values
//! after block-sparse SVD (e.g., S·Vt or U·S in MPS truncation).

use std::collections::HashMap;

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_tensor::block_sparse::BlockSparse;
use arnet_tensor::sector::Sector;

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
pub fn diagonal_scale_block_sparse<T, S>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparse<T, S>,
    weights: &BlockSingularValues<T::Real, S>,
    axis: usize,
) -> Result<BlockSparse<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
{
    if axis >= tensor.rank() {
        return Err(LinalgError::InvalidArgument(format!(
            "axis {axis} out of range for rank {}",
            tensor.rank()
        )));
    }

    // Build sector → weight vector lookup.
    let weight_map: HashMap<&S, &Vec<T::Real>> =
        weights.values.iter().map(|(s, vs)| (s, vs)).collect();

    // Clone tensor; first block_data_mut triggers one CoW copy.
    let mut result = tensor.clone();

    // Iterate original block metadata (immutable) while mutating the clone.
    for meta in tensor.block_metas() {
        let block_idx_at_axis = meta.coord.0[axis];
        let sector = tensor.indices()[axis].sector(block_idx_at_axis);

        let w = weight_map.get(sector).ok_or_else(|| {
            LinalgError::InvalidArgument(format!("no weights for sector {sector:?} at axis {axis}"))
        })?;

        let block_shape: Vec<usize> = meta
            .coord
            .0
            .iter()
            .enumerate()
            .map(|(a, &bi)| tensor.indices()[a].block_dim(bi))
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
