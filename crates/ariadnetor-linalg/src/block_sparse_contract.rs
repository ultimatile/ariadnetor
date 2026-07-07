//! Block-sparse tensor contraction.
//!
//! Contracts two `BlockSparseTensor<T, S>` tensors over specified axis
//! pairs, performing QN compatibility validation, block pairing, and dense
//! GEMM per block pair.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{ComputeBackend, ExecPolicy, GemmDescriptor, TransposeDescriptor};
use ariadnetor_tensor::Sector;
use ariadnetor_tensor::{BlockCoord, BlockSparseTensorData, QNIndex};

use crate::contract_spec::validate_contraction_axes_pair;
use crate::error::LinalgError;

/// Internal tensordot kernel for the block-sparse contraction on joined-form
/// [`BlockSparseTensorData<T, S>`], contracting `axes_lhs` against `axes_rhs`
/// and emitting the output legs in their natural order
/// (free left legs, then free right legs, each in input axis order). Free output
/// reordering is layered on top by the layout-dispatch entry
/// [`crate::LinalgContract::contract`]. A full contraction (no free legs) yields
/// a rank-0 tensor — one block holding the scalar for identity flux, zero blocks
/// for flux mismatch. `policy` is forwarded to every per-sector GEMM descriptor.
pub(crate) fn contract_block_sparse_with_policy_dense<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    lhs: &BlockSparseTensorData<T, S>,
    rhs: &BlockSparseTensorData<T, S>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
    policy: ExecPolicy,
) -> Result<BlockSparseTensorData<T, S>, LinalgError> {
    validate_contraction_axes(lhs, rhs, axes_lhs, axes_rhs)?;

    let num_contracted = axes_lhs.len();
    let lhs_rank = lhs.layout().rank();
    let rhs_rank = rhs.layout().rank();
    let output_rank = lhs_rank + rhs_rank - 2 * num_contracted;

    let free_lhs: Vec<usize> = (0..lhs_rank).filter(|a| !axes_lhs.contains(a)).collect();
    let free_rhs: Vec<usize> = (0..rhs_rank).filter(|a| !axes_rhs.contains(a)).collect();

    let rhs_groups = group_by_contracted_key(rhs, axes_rhs);

    if output_rank == 0 {
        return contract_to_scalar(backend, lhs, rhs, axes_lhs, axes_rhs, &rhs_groups);
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
    // Arity / range / duplicate checks are shared with the dense axis kernel.
    validate_contraction_axes_pair(axes_lhs, lhs.layout().rank(), axes_rhs, rhs.layout().rank())?;

    // Quantum-number compatibility is block-sparse-specific: each contracted
    // pair must oppose directions and agree on the per-sector block structure.
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
    backend: &impl ComputeBackend,
    lhs: &BlockSparseTensorData<T, S>,
    rhs: &BlockSparseTensorData<T, S>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
    rhs_groups: &HashMap<Vec<usize>, Vec<usize>>,
) -> Result<BlockSparseTensorData<T, S>, LinalgError> {
    let order = backend.preferred_order();
    let output_flux = lhs.layout().flux().fuse(rhs.layout().flux());
    // Rank-0 output: identity flux yields one block holding the scalar; a
    // non-identity fused flux is symmetry-forbidden and yields zero blocks (the
    // zero scalar). `block_data_mut` returns `Some` only in the identity case.
    let mut output = BlockSparseTensorData::zeros(Vec::new(), output_flux, order);
    if output.block_data_mut(&BlockCoord(Vec::new())).is_none() {
        return Ok(output);
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

    // Each distinct rhs block is transposed once and cached: a given rhs block
    // pairs with every lhs block sharing its contracted-axis key, so without the
    // cache it would be re-transposed once per matching lhs block.
    let mut rhs_t_cache: HashMap<usize, Vec<T>> = HashMap::new();
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
                let rhs_t = match rhs_t_cache.entry(ri) {
                    Entry::Occupied(e) => e.into_mut(),
                    Entry::Vacant(e) => {
                        let rhs_shape: Vec<usize> = (0..rhs_rank)
                            .map(|a| rhs.layout().indices()[a].block_dim(rhs_meta.coord.0[a]))
                            .collect();
                        e.insert(transpose_block(backend, rhs_data, &rhs_shape, &rhs_perm)?)
                    }
                };
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

    // The identity-flux block was confirmed present above.
    output.block_data_mut(&BlockCoord(Vec::new())).unwrap()[0] = sum;
    Ok(output)
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
) -> Result<BlockSparseTensorData<T, S>, LinalgError> {
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
    // data movement. Other non-identity permutations require a physical
    // transpose through the backend.
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
    let rhs_perm: Vec<usize> = axes_rhs.iter().chain(free_rhs.iter()).copied().collect();
    // Cache each rhs block's physical transpose: an rhs block is revisited once
    // per lhs block sharing its contracted-axis key, so caching transposes it at
    // most once per contraction rather than once per matching lhs block.
    let mut rhs_t_cache: HashMap<usize, Vec<T>> = HashMap::new();
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
            lhs_buf = transpose_block(backend, lhs_data, &lhs_block_shape, &lhs_perm)?;
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
            let rhs_slice: &[T] = if rhs_needs_physical_t {
                match rhs_t_cache.entry(ri) {
                    Entry::Occupied(e) => e.into_mut(),
                    Entry::Vacant(e) => e.insert(transpose_block(
                        backend,
                        rhs_data,
                        &rhs_block_shape,
                        &rhs_perm,
                    )?),
                }
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

    Ok(output)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Physically transpose a block into a fresh buffer via the backend.
///
/// Runs sequentially: per-block transposes are small enough that Rayon dispatch
/// costs more than it saves. HPTT (when the build enables it) or the native
/// naive kernel is selected inside the backend.
fn transpose_block<T: Scalar>(
    backend: &impl ComputeBackend,
    data: &[T],
    shape: &[usize],
    perm: &[usize],
) -> Result<Vec<T>, LinalgError> {
    let mut buf = vec![T::zero(); data.len()];
    backend.transpose(TransposeDescriptor {
        input: data,
        output: &mut buf,
        shape,
        perm,
        order: backend.preferred_order(),
        conj: false,
        policy: ExecPolicy::Sequential,
    })?;
    Ok(buf)
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
