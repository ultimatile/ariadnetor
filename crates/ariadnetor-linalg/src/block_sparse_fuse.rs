//! Block-sparse tensor leg fusion.
//!
//! Fuses consecutive legs of a [`BlockSparse<T, S>`] tensor into a single leg
//! via Kronecker-product sector fusion.

use std::collections::{BTreeMap, HashMap};

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_tensor::{BlockCoord, BlockSparse, Direction, QNIndex, Sector};

use crate::block_sparse_decomp::fused_sector::enumerate_fused_tuples;
use crate::error::LinalgError;

/// Fuse consecutive legs of a block-sparse tensor into a single leg.
///
/// Legs `start`, `start+1`, ..., `start+count-1` are fused into a single leg
/// at position `start`. The fused leg's [`QNIndex`] contains only sectors
/// and block-index tuples that are actually populated in the tensor's stored
/// blocks. Each sector's dimension equals the sum of tuple dimensions for the
/// populated tuples that fuse to that sector (not the full Kronecker product).
///
/// The fused sector for each block-index tuple is computed by applying each
/// input leg's direction to its sector, then fusing all directed sectors.
/// The stored sector in the output QNIndex accounts for `fused_direction`:
/// `Out` stores the directed fusion as-is, `In` stores its dual.
/// Within each sector, tuples are ordered lexicographically (matching the
/// canonical order from [`enumerate_fused_tuples`]).
///
/// # Errors
///
/// Returns `LinalgError::InvalidArgument` if:
/// - `count < 2`
/// - `start + count > tensor.rank()`
pub fn fuse_legs_block_sparse<T, S, B>(
    backend: &B,
    tensor: &BlockSparse<T, S>,
    start: usize,
    count: usize,
    fused_direction: Direction,
) -> Result<BlockSparse<T, S>, LinalgError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let rank = tensor.rank();

    if count < 2 {
        return Err(LinalgError::InvalidArgument(format!(
            "fuse_legs count must be >= 2, got {count}"
        )));
    }
    if start + count > rank {
        return Err(LinalgError::InvalidArgument(format!(
            "fuse_legs range [{start}..{}] out of range for rank {rank}",
            start + count
        )));
    }

    let indices = tensor.indices();
    let order = backend.preferred_order();

    // Enumerate ALL possible fused tuples for the fused legs.
    // Build an O(1) lookup: tuple → (directed_fused_sector, dim).
    let all_fused_groups = enumerate_fused_tuples(&indices[start..start + count]);
    let mut tuple_lookup: HashMap<Vec<usize>, (S, usize)> = HashMap::new();
    for (directed_sector, tuples) in &all_fused_groups {
        for (tuple, dim) in tuples {
            tuple_lookup.insert(tuple.clone(), (directed_sector.clone(), *dim));
        }
    }

    // Scan input blocks to collect populated (sector, tuple) pairs.
    let mut populated_sectors: BTreeMap<S, Vec<(Vec<usize>, usize)>> = BTreeMap::new();
    for meta in tensor.block_metas() {
        let fuse_tuple: Vec<usize> = meta.coord.0[start..start + count].to_vec();
        let (directed_sector, dim) = tuple_lookup
            .get(&fuse_tuple)
            .expect("input block tuple must map to a fused sector");
        let entry = populated_sectors
            .entry(directed_sector.clone())
            .or_default();
        if !entry.iter().any(|(t, _)| *t == fuse_tuple) {
            entry.push((fuse_tuple, *dim));
        }
    }

    // Sort tuples within each sector in lexicographic order (canonical ordering).
    for tuples in populated_sectors.values_mut() {
        tuples.sort_by(|(a, _), (b, _)| a.cmp(b));
    }

    // Build fused QNIndex from populated sectors only.
    let mut fused_qn_blocks: Vec<(S, usize)> = Vec::with_capacity(populated_sectors.len());
    for (directed_sector, tuples) in &populated_sectors {
        let total_dim: usize = tuples.iter().map(|(_, d)| d).sum();
        let stored_sector = match fused_direction {
            Direction::Out => directed_sector.clone(),
            Direction::In => directed_sector.dual(),
        };
        fused_qn_blocks.push((stored_sector, total_dim));
    }

    let fused_index = QNIndex::new(fused_qn_blocks, fused_direction);

    // QNIndex::new sorts by stored sector. Build a reverse lookup from
    // directed fused sector → block index in the sorted QNIndex.
    let sector_to_block_idx: HashMap<S, usize> = populated_sectors
        .keys()
        .map(|directed_sector| {
            let stored = match fused_direction {
                Direction::Out => directed_sector.clone(),
                Direction::In => directed_sector.dual(),
            };
            let block_idx = fused_index
                .blocks()
                .iter()
                .position(|(s, _)| *s == stored)
                .expect("fused sector must exist in QNIndex");
            (directed_sector.clone(), block_idx)
        })
        .collect();

    // Build per-tuple offset within the fused dimension.
    // Tuples are in lexicographic order within each sector (sorted above).
    let mut tuple_to_fused: HashMap<Vec<usize>, (usize, usize)> = HashMap::new();
    for (directed_sector, tuples) in &populated_sectors {
        let &block_idx = sector_to_block_idx.get(directed_sector).unwrap();
        let mut offset = 0;
        for (tuple, dim) in tuples {
            tuple_to_fused.insert(tuple.clone(), (block_idx, offset));
            offset += dim;
        }
    }

    // Build output indices
    let mut out_indices: Vec<QNIndex<S>> = Vec::with_capacity(rank - count + 1);
    out_indices.extend(indices[..start].iter().cloned());
    out_indices.push(fused_index);
    out_indices.extend(indices[start + count..].iter().cloned());

    let mut output = BlockSparse::zeros(out_indices, tensor.flux().clone());

    // Pre-compute fused dimension sizes per fused block index (avoids borrow conflict)
    let fused_dim_per_block: Vec<usize> = (0..output.indices()[start].num_blocks())
        .map(|bi| output.indices()[start].block_dim(bi))
        .collect();

    // Copy data from each input block to the correct output block
    for meta in tensor.block_metas() {
        let fuse_tuple: Vec<usize> = meta.coord.0[start..start + count].to_vec();
        let &(fused_block_idx, fused_offset) = tuple_to_fused
            .get(&fuse_tuple)
            .expect("input block tuple must have a fused mapping");

        // Output coordinate: [original[..start], fused_block_idx, original[start+count..]]
        let mut out_coord_vec = Vec::with_capacity(rank - count + 1);
        out_coord_vec.extend_from_slice(&meta.coord.0[..start]);
        out_coord_vec.push(fused_block_idx);
        out_coord_vec.extend_from_slice(&meta.coord.0[start + count..]);
        let out_coord = BlockCoord(out_coord_vec);

        // Block shape for the input block
        let block_shape: Vec<usize> = meta
            .coord
            .0
            .iter()
            .enumerate()
            .map(|(a, &bi)| indices[a].block_dim(bi))
            .collect();

        let leading_prod: usize = block_shape[..start].iter().product::<usize>().max(1);
        let fused_prod: usize = block_shape[start..start + count]
            .iter()
            .product::<usize>()
            .max(1);
        let trailing_prod: usize = block_shape[start + count..]
            .iter()
            .product::<usize>()
            .max(1);
        let fused_total = fused_dim_per_block[fused_block_idx];

        let src_data = tensor.block_data(&meta.coord).unwrap();
        let dst_data = output
            .block_data_mut(&out_coord)
            .expect("output block must exist");

        copy_fused_block(
            src_data,
            dst_data,
            leading_prod,
            fused_prod,
            fused_total,
            trailing_prod,
            fused_offset,
            order,
        );
    }

    Ok(output)
}

/// Copy data from an input block into the fused output block,
/// respecting the memory order layout.
///
/// For RM: iterates over leading indices, copies contiguous fused×trailing slabs.
/// For CM: iterates over trailing indices, copies contiguous leading×fused slabs.
#[allow(clippy::too_many_arguments)]
fn copy_fused_block<T: Copy>(
    src: &[T],
    dst: &mut [T],
    leading: usize,
    fused: usize,
    fused_total: usize,
    trailing: usize,
    fused_offset: usize,
    order: MemoryOrder,
) {
    match order {
        MemoryOrder::RowMajor => {
            // RM layout: element (l, f, t) at l * fused * trailing + f * trailing + t
            // Output:    element (l, f, t) at l * fused_total * trailing + (fused_offset + f) * trailing + t
            // → for each l, copy contiguous slab of size fused * trailing
            let src_stride = fused * trailing;
            let dst_stride = fused_total * trailing;
            for l in 0..leading {
                let src_start = l * src_stride;
                let dst_start = l * dst_stride + fused_offset * trailing;
                dst[dst_start..dst_start + src_stride]
                    .copy_from_slice(&src[src_start..src_start + src_stride]);
            }
        }
        MemoryOrder::ColumnMajor => {
            // CM layout: element (l, f, t) at l + f * leading + t * leading * fused
            // Output:    element (l, f, t) at l + (fused_offset + f) * leading + t * leading * fused_total
            // → for each t, copy contiguous slab of size leading * fused
            let src_stride = leading * fused;
            let dst_stride = leading * fused_total;
            for t in 0..trailing {
                let src_start = t * src_stride;
                let dst_start = fused_offset * leading + t * dst_stride;
                dst[dst_start..dst_start + src_stride]
                    .copy_from_slice(&src[src_start..src_start + src_stride]);
            }
        }
    }
}

#[cfg(test)]
mod tests;
