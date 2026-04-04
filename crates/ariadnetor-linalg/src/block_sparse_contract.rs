//! Block-sparse tensor contraction.
//!
//! Contracts two [`BlockSparse<T, S>`] tensors over specified axis pairs,
//! performing QN compatibility validation, block pairing, and dense GEMM
//! per block pair.

use std::collections::HashMap;

use arnet_core::backend::{ComputeBackend, GemmDescriptor, MemoryOrder};
use arnet_core::scalar::Scalar;
use arnet_tensor::block_sparse::{BlockCoord, BlockSparse, QNIndex};
use arnet_tensor::sector::Sector;

use crate::error::LinalgError;

/// Result of a block-sparse tensor contraction.
pub enum BlockSparseContractResult<T, S: Sector> {
    /// Partial contraction produced a block-sparse tensor.
    Tensor(BlockSparse<T, S>),
    /// Full contraction produced a scalar.
    Scalar(T),
}

/// Contract two block-sparse tensors over specified axis pairs.
///
/// # Partial contraction (free axes remain)
///
/// Returns `BlockSparse` with indices `[lhs_free..., rhs_free...]` and
/// `flux = lhs.flux().fuse(rhs.flux())`.
///
/// # Full contraction (all axes contracted)
///
/// Returns a scalar. If `lhs.flux().fuse(rhs.flux()) != identity`, the
/// result is `T::zero()` (symmetry selection rule).
///
/// # Errors
///
/// Returns `LinalgError::InvalidArgument` if axis pairs are invalid:
/// length mismatch, out-of-range, duplicates, sector/dimension mismatch,
/// or non-opposite directions.
pub fn contract_block_sparse<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    lhs: &BlockSparse<T, S>,
    rhs: &BlockSparse<T, S>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
) -> Result<BlockSparseContractResult<T, S>, LinalgError> {
    validate_contraction_axes(lhs, rhs, axes_lhs, axes_rhs)?;

    let num_contracted = axes_lhs.len();
    let output_rank = lhs.rank() + rhs.rank() - 2 * num_contracted;

    let free_lhs: Vec<usize> = (0..lhs.rank()).filter(|a| !axes_lhs.contains(a)).collect();
    let free_rhs: Vec<usize> = (0..rhs.rank()).filter(|a| !axes_rhs.contains(a)).collect();

    let rhs_groups = group_by_contracted_key(rhs, axes_rhs);

    if output_rank == 0 {
        return contract_to_scalar(lhs, rhs, axes_lhs, axes_rhs, &rhs_groups);
    }

    contract_to_tensor(
        backend,
        lhs,
        rhs,
        axes_lhs,
        axes_rhs,
        &free_lhs,
        &free_rhs,
        &rhs_groups,
    )
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_contraction_axes<T, S: Sector>(
    lhs: &BlockSparse<T, S>,
    rhs: &BlockSparse<T, S>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
) -> Result<(), LinalgError> {
    if axes_lhs.len() != axes_rhs.len() {
        return Err(LinalgError::InvalidArgument(format!(
            "axes_lhs length {} != axes_rhs length {}",
            axes_lhs.len(),
            axes_rhs.len()
        )));
    }

    for (i, &a) in axes_lhs.iter().enumerate() {
        if a >= lhs.rank() {
            return Err(LinalgError::InvalidArgument(format!(
                "axes_lhs[{i}] = {a} out of range for lhs rank {}",
                lhs.rank()
            )));
        }
        if axes_lhs[..i].contains(&a) {
            return Err(LinalgError::InvalidArgument(format!(
                "axes_lhs contains duplicate axis {a}"
            )));
        }
    }
    for (i, &a) in axes_rhs.iter().enumerate() {
        if a >= rhs.rank() {
            return Err(LinalgError::InvalidArgument(format!(
                "axes_rhs[{i}] = {a} out of range for rhs rank {}",
                rhs.rank()
            )));
        }
        if axes_rhs[..i].contains(&a) {
            return Err(LinalgError::InvalidArgument(format!(
                "axes_rhs contains duplicate axis {a}"
            )));
        }
    }

    for (i, (&al, &ar)) in axes_lhs.iter().zip(axes_rhs.iter()).enumerate() {
        let il = &lhs.indices()[al];
        let ir = &rhs.indices()[ar];

        if il.direction() == ir.direction() {
            return Err(LinalgError::InvalidArgument(format!(
                "Contracted pair {i}: axes ({al}, {ar}) have same direction {:?}",
                il.direction()
            )));
        }
        if il.num_blocks() != ir.num_blocks() {
            return Err(LinalgError::InvalidArgument(format!(
                "Contracted pair {i}: sector count mismatch {} vs {}",
                il.num_blocks(),
                ir.num_blocks()
            )));
        }
        for j in 0..il.num_blocks() {
            if il.sector(j) != ir.sector(j) {
                return Err(LinalgError::InvalidArgument(format!(
                    "Contracted pair {i}: sector mismatch at block {j}: {:?} vs {:?}",
                    il.sector(j),
                    ir.sector(j)
                )));
            }
            if il.block_dim(j) != ir.block_dim(j) {
                return Err(LinalgError::InvalidArgument(format!(
                    "Contracted pair {i}: dim mismatch at block {j}: {} vs {}",
                    il.block_dim(j),
                    ir.block_dim(j)
                )));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Block pairing
// ---------------------------------------------------------------------------

/// Group blocks by their contracted-axis block indices for O(1) lookup.
fn group_by_contracted_key<T, S: Sector>(
    tensor: &BlockSparse<T, S>,
    contracted_axes: &[usize],
) -> HashMap<Vec<usize>, Vec<usize>> {
    let mut groups: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for (idx, meta) in tensor.block_metas().iter().enumerate() {
        let key: Vec<usize> = contracted_axes.iter().map(|&a| meta.coord.0[a]).collect();
        groups.entry(key).or_default().push(idx);
    }
    groups
}

// ---------------------------------------------------------------------------
// Full contraction → scalar
// ---------------------------------------------------------------------------

fn contract_to_scalar<T: Scalar, S: Sector>(
    lhs: &BlockSparse<T, S>,
    rhs: &BlockSparse<T, S>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
    rhs_groups: &HashMap<Vec<usize>, Vec<usize>>,
) -> Result<BlockSparseContractResult<T, S>, LinalgError> {
    if lhs.flux().fuse(rhs.flux()) != S::identity() {
        return Ok(BlockSparseContractResult::Scalar(T::zero()));
    }

    // Permute rhs so its axes align with lhs axis order for dot product
    let rhs_perm: Vec<usize> = (0..lhs.rank())
        .map(|l| {
            let pos = axes_lhs.iter().position(|&a| a == l).unwrap();
            axes_rhs[pos]
        })
        .collect();
    let needs_transpose = !is_identity_perm(&rhs_perm);

    let rhs_metas = rhs.block_metas();
    let mut sum = T::zero();

    for lhs_meta in lhs.block_metas() {
        let key: Vec<usize> = axes_lhs.iter().map(|&a| lhs_meta.coord.0[a]).collect();
        let Some(rhs_indices) = rhs_groups.get(&key) else {
            continue;
        };

        let lhs_data = lhs.block_data(&lhs_meta.coord).unwrap();

        for &ri in rhs_indices {
            let rhs_meta = &rhs_metas[ri];
            let rhs_data = rhs.block_data(&rhs_meta.coord).unwrap();

            if needs_transpose {
                let rhs_shape: Vec<usize> = (0..rhs.rank())
                    .map(|a| rhs.indices()[a].block_dim(rhs_meta.coord.0[a]))
                    .collect();
                let rhs_t = transpose_block_data(rhs_data, &rhs_shape, &rhs_perm);
                for (&a, &b) in lhs_data.iter().zip(rhs_t.iter()) {
                    sum = sum + a * b;
                }
            } else {
                for (&a, &b) in lhs_data.iter().zip(rhs_data.iter()) {
                    sum = sum + a * b;
                }
            }
        }
    }

    Ok(BlockSparseContractResult::Scalar(sum))
}

// ---------------------------------------------------------------------------
// Partial contraction → tensor
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn contract_to_tensor<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    lhs: &BlockSparse<T, S>,
    rhs: &BlockSparse<T, S>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
    free_lhs: &[usize],
    free_rhs: &[usize],
    rhs_groups: &HashMap<Vec<usize>, Vec<usize>>,
) -> Result<BlockSparseContractResult<T, S>, LinalgError> {
    let mut output_indices: Vec<QNIndex<S>> = Vec::with_capacity(free_lhs.len() + free_rhs.len());
    for &a in free_lhs {
        output_indices.push(lhs.indices()[a].clone());
    }
    for &a in free_rhs {
        output_indices.push(rhs.indices()[a].clone());
    }
    let output_flux = lhs.flux().fuse(rhs.flux());
    let mut output = BlockSparse::zeros(output_indices, output_flux);

    // lhs → [free..., contracted...], rhs → [contracted..., free...]
    let lhs_perm: Vec<usize> = free_lhs.iter().chain(axes_lhs.iter()).copied().collect();
    let rhs_perm: Vec<usize> = axes_rhs.iter().chain(free_rhs.iter()).copied().collect();
    let lhs_needs_t = !is_identity_perm(&lhs_perm);
    let rhs_needs_t = !is_identity_perm(&rhs_perm);

    let rhs_metas = rhs.block_metas();

    for lhs_meta in lhs.block_metas() {
        let key: Vec<usize> = axes_lhs.iter().map(|&a| lhs_meta.coord.0[a]).collect();
        let Some(rhs_indices) = rhs_groups.get(&key) else {
            continue;
        };

        let lhs_block_shape: Vec<usize> = (0..lhs.rank())
            .map(|a| lhs.indices()[a].block_dim(lhs_meta.coord.0[a]))
            .collect();
        let m: usize = free_lhs
            .iter()
            .map(|&a| lhs_block_shape[a])
            .product::<usize>()
            .max(1);
        let k: usize = axes_lhs
            .iter()
            .map(|&a| lhs_block_shape[a])
            .product::<usize>()
            .max(1);

        let lhs_data = lhs.block_data(&lhs_meta.coord).unwrap();
        let lhs_buf;
        let lhs_slice: &[T] = if lhs_needs_t {
            lhs_buf = transpose_block_data(lhs_data, &lhs_block_shape, &lhs_perm);
            &lhs_buf
        } else {
            lhs_data
        };

        for &ri in rhs_indices {
            let rhs_meta = &rhs_metas[ri];

            let mut out_coord_vec = Vec::with_capacity(free_lhs.len() + free_rhs.len());
            for &a in free_lhs {
                out_coord_vec.push(lhs_meta.coord.0[a]);
            }
            for &a in free_rhs {
                out_coord_vec.push(rhs_meta.coord.0[a]);
            }
            let out_coord = BlockCoord(out_coord_vec);

            let Some(out_data) = output.block_data_mut(&out_coord) else {
                continue;
            };

            let rhs_block_shape: Vec<usize> = (0..rhs.rank())
                .map(|a| rhs.indices()[a].block_dim(rhs_meta.coord.0[a]))
                .collect();
            let n: usize = free_rhs
                .iter()
                .map(|&a| rhs_block_shape[a])
                .product::<usize>()
                .max(1);

            let rhs_data = rhs.block_data(&rhs_meta.coord).unwrap();
            let rhs_buf;
            let rhs_slice: &[T] = if rhs_needs_t {
                rhs_buf = transpose_block_data(rhs_data, &rhs_block_shape, &rhs_perm);
                &rhs_buf
            } else {
                rhs_data
            };

            backend.gemm(GemmDescriptor {
                m,
                n,
                k,
                alpha: T::one(),
                a: lhs_slice,
                b: rhs_slice,
                beta: T::one(),
                c: out_data,
                trans_a: false,
                trans_b: false,
                order: MemoryOrder::RowMajor,
            })?;
        }
    }

    Ok(BlockSparseContractResult::Tensor(output))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Transpose block data stored in row-major layout.
///
/// Convention: `perm[new_axis] = old_axis`.
fn transpose_block_data<T: Copy>(data: &[T], shape: &[usize], perm: &[usize]) -> Vec<T> {
    let rank = shape.len();
    if rank <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let total = data.len();
    let old_strides = row_major_strides(shape);
    let perm_strides: Vec<usize> = perm.iter().map(|&p| old_strides[p]).collect();

    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let new_strides = row_major_strides(&new_shape);

    let mut result = vec![data[0]; total];
    let mut idx = vec![0usize; rank];

    for (flat, out) in result.iter_mut().enumerate() {
        let mut rem = flat;
        for i in 0..rank {
            idx[i] = rem / new_strides[i];
            rem %= new_strides[i];
        }
        let old_flat: usize = idx
            .iter()
            .zip(perm_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();
        *out = data[old_flat];
    }

    result
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| p == i)
}

#[cfg(test)]
mod tests;
