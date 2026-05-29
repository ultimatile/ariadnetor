//! Block-sparse tensor contraction.
//!
//! Contracts two [`BlockSparseTensor<T, S, B>`] tensors over specified axis
//! pairs, performing QN compatibility validation, block pairing, and dense
//! GEMM per block pair.

use std::collections::HashMap;

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, GemmDescriptor, MemoryOrder};
use arnet_tensor::Sector;
use arnet_tensor::{BlockCoord, BlockSparseTensor, BlockSparseTensorData, QNIndex};

use crate::error::LinalgError;

/// Result of a block-sparse tensor contraction.
pub enum BlockSparseContractResult<T, S: Sector, B: ComputeBackend> {
    /// Partial contraction produced a block-sparse tensor.
    Tensor(BlockSparseTensor<T, S, B>),
    /// Full contraction produced a scalar.
    Scalar(T),
}

/// Internal kernel form of [`BlockSparseContractResult`] operating on
/// joined-form [`BlockSparseTensorData<T, S>`].
pub(crate) enum BlockSparseContractResultBsp<T, S: Sector> {
    Tensor(BlockSparseTensorData<T, S>),
    Scalar(T),
}

/// Contract two block-sparse tensors over specified axis pairs.
///
/// The backend is taken from `lhs` and a tensor result is wrapped against
/// `lhs`'s backend Arc. Callers must ensure `lhs` and `rhs` share the same
/// backend Arc; a mismatch silently runs on `lhs`'s backend and labels the
/// output with `lhs`'s backend, which is wrong for backends carrying state.
///
/// # Partial contraction (free axes remain)
///
/// Returns `BlockSparseTensor` with indices `[lhs_free..., rhs_free...]` and
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
pub fn contract_block_sparse<T: Scalar, S: Sector, B: ComputeBackend>(
    lhs: &BlockSparseTensor<T, S, B>,
    rhs: &BlockSparseTensor<T, S, B>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
) -> Result<BlockSparseContractResult<T, S, B>, LinalgError> {
    contract_block_sparse_with_policy(lhs, rhs, axes_lhs, axes_rhs, ExecPolicy::Sequential)
}

/// Block-sparse tensor contraction with caller-specified execution policy
/// for per-sector GEMM.
///
/// Expert-layer counterpart of [`contract_block_sparse`]. The default wrapper
/// hardcodes `ExecPolicy::Sequential` (conservative for typical small-sector
/// cases and compatible with future outer parallelism); this entry point lets
/// a caller opt a large-sector case into `Parallel`. The policy is forwarded
/// to every per-sector GEMM descriptor.
///
/// The backend is taken from `lhs` and the result is wrapped against `lhs`'s
/// backend Arc. Callers must ensure `lhs` and `rhs` share the same backend
/// Arc; a mismatch silently runs on `lhs`'s backend and labels the output
/// with `lhs`'s backend, which is wrong for backends carrying state.
pub(crate) fn contract_block_sparse_with_policy<T: Scalar, S: Sector, B: ComputeBackend>(
    lhs: &BlockSparseTensor<T, S, B>,
    rhs: &BlockSparseTensor<T, S, B>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
    policy: ExecPolicy,
) -> Result<BlockSparseContractResult<T, S, B>, LinalgError> {
    crate::tensor_bridge::assert_bsp_layout_order_matches_backend(
        lhs,
        "contract_block_sparse: lhs",
    );
    crate::tensor_bridge::assert_bsp_layout_order_matches_backend(
        rhs,
        "contract_block_sparse: rhs",
    );
    let backend_arc = lhs.backend_arc().clone();
    let result = contract_block_sparse_with_policy_dense(
        lhs.backend(),
        lhs.data(),
        rhs.data(),
        axes_lhs,
        axes_rhs,
        policy,
    )?;
    match result {
        BlockSparseContractResultBsp::Tensor(t) => Ok(BlockSparseContractResult::Tensor(
            BlockSparseTensor::with_backend(t, backend_arc),
        )),
        BlockSparseContractResultBsp::Scalar(s) => Ok(BlockSparseContractResult::Scalar(s)),
    }
}

/// Internal kernel for [`contract_block_sparse_with_policy`] on joined-form
/// [`BlockSparseTensorData<T, S>`].
pub(crate) fn contract_block_sparse_with_policy_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    lhs: &BlockSparseTensorData<T, S>,
    rhs: &BlockSparseTensorData<T, S>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
    policy: ExecPolicy,
) -> Result<BlockSparseContractResultBsp<T, S>, LinalgError> {
    validate_contraction_axes(lhs, rhs, axes_lhs, axes_rhs)?;

    let num_contracted = axes_lhs.len();
    let lhs_rank = lhs.layout().rank();
    let rhs_rank = rhs.layout().rank();
    let output_rank = lhs_rank + rhs_rank - 2 * num_contracted;

    let free_lhs: Vec<usize> = (0..lhs_rank).filter(|a| !axes_lhs.contains(a)).collect();
    let free_rhs: Vec<usize> = (0..rhs_rank).filter(|a| !axes_rhs.contains(a)).collect();

    let rhs_groups = group_by_contracted_key(rhs, axes_rhs);

    if output_rank == 0 {
        let order = backend.preferred_order();
        return contract_to_scalar(lhs, rhs, axes_lhs, axes_rhs, &rhs_groups, order);
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
        policy,
    )
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_contraction_axes<T, S: Sector>(
    lhs: &BlockSparseTensorData<T, S>,
    rhs: &BlockSparseTensorData<T, S>,
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

    let lhs_rank = lhs.layout().rank();
    let rhs_rank = rhs.layout().rank();
    for (i, &a) in axes_lhs.iter().enumerate() {
        if a >= lhs_rank {
            return Err(LinalgError::InvalidArgument(format!(
                "axes_lhs[{i}] = {a} out of range for lhs rank {lhs_rank}"
            )));
        }
        if axes_lhs[..i].contains(&a) {
            return Err(LinalgError::InvalidArgument(format!(
                "axes_lhs contains duplicate axis {a}"
            )));
        }
    }
    for (i, &a) in axes_rhs.iter().enumerate() {
        if a >= rhs_rank {
            return Err(LinalgError::InvalidArgument(format!(
                "axes_rhs[{i}] = {a} out of range for rhs rank {rhs_rank}"
            )));
        }
        if axes_rhs[..i].contains(&a) {
            return Err(LinalgError::InvalidArgument(format!(
                "axes_rhs contains duplicate axis {a}"
            )));
        }
    }

    for (i, (&al, &ar)) in axes_lhs.iter().zip(axes_rhs.iter()).enumerate() {
        let il = &lhs.layout().indices()[al];
        let ir = &rhs.layout().indices()[ar];

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
    tensor: &BlockSparseTensorData<T, S>,
    contracted_axes: &[usize],
) -> HashMap<Vec<usize>, Vec<usize>> {
    let mut groups: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for (idx, meta) in tensor.layout().block_metas().iter().enumerate() {
        let key: Vec<usize> = contracted_axes.iter().map(|&a| meta.coord.0[a]).collect();
        groups.entry(key).or_default().push(idx);
    }
    groups
}

// ---------------------------------------------------------------------------
// Full contraction → scalar
// ---------------------------------------------------------------------------

fn contract_to_scalar<T: Scalar, S: Sector>(
    lhs: &BlockSparseTensorData<T, S>,
    rhs: &BlockSparseTensorData<T, S>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
    rhs_groups: &HashMap<Vec<usize>, Vec<usize>>,
    order: MemoryOrder,
) -> Result<BlockSparseContractResultBsp<T, S>, LinalgError> {
    if lhs.layout().flux().fuse(rhs.layout().flux()) != S::identity() {
        return Ok(BlockSparseContractResultBsp::Scalar(T::zero()));
    }

    let lhs_rank = lhs.layout().rank();
    let rhs_rank = rhs.layout().rank();
    // Permute rhs so its axes align with lhs axis order for dot product
    let rhs_perm: Vec<usize> = (0..lhs_rank)
        .map(|l| {
            let pos = axes_lhs.iter().position(|&a| a == l).unwrap();
            axes_rhs[pos]
        })
        .collect();
    let needs_transpose = !is_identity_perm(&rhs_perm);

    let rhs_metas = rhs.layout().block_metas();
    let mut sum = T::zero();

    let mut key = Vec::with_capacity(axes_lhs.len());

    for lhs_meta in lhs.layout().block_metas() {
        key.clear();
        key.extend(axes_lhs.iter().map(|&a| lhs_meta.coord.0[a]));
        let Some(rhs_indices) = rhs_groups.get(key.as_slice()) else {
            continue;
        };

        let lhs_data = lhs.block_data(&lhs_meta.coord).unwrap();

        for &ri in rhs_indices {
            let rhs_meta = &rhs_metas[ri];
            let rhs_data = rhs.block_data(&rhs_meta.coord).unwrap();

            if needs_transpose {
                let rhs_shape: Vec<usize> = (0..rhs_rank)
                    .map(|a| rhs.layout().indices()[a].block_dim(rhs_meta.coord.0[a]))
                    .collect();
                let rhs_t = transpose_block_data(rhs_data, &rhs_shape, &rhs_perm, order);
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

    Ok(BlockSparseContractResultBsp::Scalar(sum))
}

// ---------------------------------------------------------------------------
// Partial contraction → tensor
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn contract_to_tensor<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    lhs: &BlockSparseTensorData<T, S>,
    rhs: &BlockSparseTensorData<T, S>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
    free_lhs: &[usize],
    free_rhs: &[usize],
    rhs_groups: &HashMap<Vec<usize>, Vec<usize>>,
    policy: ExecPolicy,
) -> Result<BlockSparseContractResultBsp<T, S>, LinalgError> {
    let lhs_rank = lhs.layout().rank();
    let rhs_rank = rhs.layout().rank();
    let order = backend.preferred_order();
    let mut output_indices: Vec<QNIndex<S>> = Vec::with_capacity(free_lhs.len() + free_rhs.len());
    for &a in free_lhs {
        output_indices.push(lhs.layout().indices()[a].clone());
    }
    for &a in free_rhs {
        output_indices.push(rhs.layout().indices()[a].clone());
    }
    let output_flux = lhs.layout().flux().fuse(rhs.layout().flux());
    let mut output = BlockSparseTensorData::zeros(output_indices, output_flux, order);

    // Determine transpose strategy per operand.
    // GEMM wants lhs as (m, k) and rhs as (k, n). When the permutation
    // is a simple prefix/suffix swap, the reshaped block can be read in
    // the backend's preferred order via trans_a/trans_b without physical
    // data movement. Other non-identity permutations require explicit
    // transpose_block_data.
    let lhs_is_id = free_lhs
        .iter()
        .chain(axes_lhs.iter())
        .copied()
        .enumerate()
        .all(|(i, p)| p == i);
    let lhs_trans_flag = compute_lhs_trans_flag(lhs_is_id, axes_lhs, free_lhs, lhs_rank);
    let lhs_needs_physical_t = !lhs_is_id && !lhs_trans_flag;

    let rhs_is_id = axes_rhs
        .iter()
        .chain(free_rhs.iter())
        .copied()
        .enumerate()
        .all(|(i, p)| p == i);
    let rhs_trans_flag = compute_rhs_trans_flag(rhs_is_id, axes_rhs, free_rhs, rhs_rank);
    let rhs_needs_physical_t = !rhs_is_id && !rhs_trans_flag;

    // `lhs` / `rhs` are immutable inputs; `output` is a separate value
    // that we mutate. Borrowing `lhs.layout().block_metas()` for the
    // loop is compatible with `lhs.block_data(...)` inside the body
    // (both immutable on the same value) and with
    // `output.block_data_mut(...)` (different value).
    let rhs_metas = rhs.layout().block_metas();
    let mut key = Vec::with_capacity(axes_lhs.len());

    for lhs_meta in lhs.layout().block_metas() {
        key.clear();
        key.extend(axes_lhs.iter().map(|&a| lhs_meta.coord.0[a]));
        let Some(rhs_indices) = rhs_groups.get(key.as_slice()) else {
            continue;
        };

        let lhs_block_shape: Vec<usize> = (0..lhs_rank)
            .map(|a| lhs.layout().indices()[a].block_dim(lhs_meta.coord.0[a]))
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
        let lhs_slice: &[T] = if lhs_needs_physical_t {
            let lhs_perm: Vec<usize> = free_lhs.iter().chain(axes_lhs.iter()).copied().collect();
            lhs_buf = transpose_block_data(lhs_data, &lhs_block_shape, &lhs_perm, order);
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

            let rhs_block_shape: Vec<usize> = (0..rhs_rank)
                .map(|a| rhs.layout().indices()[a].block_dim(rhs_meta.coord.0[a]))
                .collect();
            let n: usize = free_rhs
                .iter()
                .map(|&a| rhs_block_shape[a])
                .product::<usize>()
                .max(1);

            let rhs_data = rhs.block_data(&rhs_meta.coord).unwrap();
            let rhs_buf;
            let rhs_slice: &[T] = if rhs_needs_physical_t {
                let rhs_perm: Vec<usize> =
                    axes_rhs.iter().chain(free_rhs.iter()).copied().collect();
                rhs_buf = transpose_block_data(rhs_data, &rhs_block_shape, &rhs_perm, order);
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
                trans_a: lhs_trans_flag,
                trans_b: rhs_trans_flag,
                order,
                policy,
            })?;
        }
    }

    Ok(BlockSparseContractResultBsp::Tensor(output))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Transpose block data in the given memory order layout.
///
/// Convention: `perm[new_axis] = old_axis`.
pub(crate) fn transpose_block_data<T: Copy>(
    data: &[T],
    shape: &[usize],
    perm: &[usize],
    order: MemoryOrder,
) -> Vec<T> {
    let rank = shape.len();
    if rank <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let total = data.len();
    let old_strides = compute_strides(shape, order);
    let perm_strides: Vec<usize> = perm.iter().map(|&p| old_strides[p]).collect();

    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let new_strides = compute_strides(&new_shape, order);

    // Axis iteration order for flat→multi-index decomposition:
    // RowMajor strides are descending → process 0..rank
    // ColumnMajor strides are ascending → process (0..rank).rev()
    let decomp_order: Vec<usize> = match order {
        MemoryOrder::RowMajor => (0..rank).collect(),
        MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
    };

    let mut result = vec![data[0]; total];
    let mut idx = vec![0usize; rank];

    for (flat, out) in result.iter_mut().enumerate() {
        let mut rem = flat;
        for &ax in &decomp_order {
            idx[ax] = rem / new_strides[ax];
            rem %= new_strides[ax];
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

/// Compute strides for the given memory order.
pub(crate) fn compute_strides(shape: &[usize], order: MemoryOrder) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    match order {
        MemoryOrder::RowMajor => {
            for i in (0..rank.saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        MemoryOrder::ColumnMajor => {
            for i in 1..rank {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
        }
    }
    strides
}

fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| p == i)
}

/// Check if axes are exactly `[0, 1, ..., n-1]` in order.
///
/// When contracted axes form an in-order leading prefix, the block's
/// column-major 2D view is naturally (k, m), so GEMM `trans_a=true`
/// reads it as (m, k) without data movement. The axes must be in
/// ascending order (not just a prefix as a set) to ensure the k-dimension
/// linearization matches between operands.
fn is_ascending_prefix(axes: &[usize]) -> bool {
    axes.iter().enumerate().all(|(i, &a)| a == i)
}

/// Check if axes are exactly `[rank-n, rank-n+1, ..., rank-1]` in order.
fn is_ascending_suffix(axes: &[usize], rank: usize) -> bool {
    let offset = rank - axes.len();
    axes.iter().enumerate().all(|(i, &a)| a == offset + i)
}

/// LHS GEMM trans-flag predicate: the contraction axes are an ascending
/// prefix and the free axes are an ascending suffix of the rank, in which
/// case the block can be read as `(m, k)` directly via the trans flag
/// without physical permutation.
///
/// Wraps the equivalent `delete !` operand (`!lhs_is_id`) in a named fn so
/// the equivalent-mutant exclusion can be anchored by function name rather
/// than line number. Both routes produce identical numerical output by
/// design (trans-flag GEMM vs. physical-transpose GEMM).
fn compute_lhs_trans_flag(
    lhs_is_id: bool,
    axes_lhs: &[usize],
    free_lhs: &[usize],
    rank: usize,
) -> bool {
    !lhs_is_id && is_ascending_prefix(axes_lhs) && is_ascending_suffix(free_lhs, rank)
}

/// RHS GEMM trans-flag predicate: symmetric counterpart of
/// [`compute_lhs_trans_flag`]. Free axes ascending prefix, contraction
/// axes ascending suffix.
fn compute_rhs_trans_flag(
    rhs_is_id: bool,
    axes_rhs: &[usize],
    free_rhs: &[usize],
    rank: usize,
) -> bool {
    !rhs_is_id && is_ascending_prefix(free_rhs) && is_ascending_suffix(axes_rhs, rank)
}

#[cfg(test)]
mod tests;
