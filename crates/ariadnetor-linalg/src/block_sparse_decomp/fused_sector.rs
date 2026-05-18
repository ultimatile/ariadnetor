//! Fused sector computation, matrix assembly, and output tensor construction.
//!
//! Internal helpers shared by SVD, QR, and LQ block-sparse decompositions.

use std::collections::BTreeMap;

use arnet_core::Scalar;
use arnet_core::backend::MemoryOrder;
use arnet_tensor::Sector;
use arnet_tensor::{BlockCoord, BlockSparseTensorData, Direction, QNIndex};

/// Per-sector grouping of block-index tuples for matrix assembly.
pub(super) struct FusedSectorGroup<S: Sector> {
    /// The fused left sector that keys this group.
    pub(super) sector: S,
    /// Left block-index tuples (sorted lexicographically).
    pub(super) left_tuples: Vec<Vec<usize>>,
    /// Right block-index tuples (sorted lexicographically).
    pub(super) right_tuples: Vec<Vec<usize>>,
    /// Row dimension for each left tuple.
    pub(super) left_dims: Vec<usize>,
    /// Column dimension for each right tuple.
    pub(super) right_dims: Vec<usize>,
    /// Cumulative row offsets.
    pub(super) left_offsets: Vec<usize>,
    /// Cumulative column offsets.
    pub(super) right_offsets: Vec<usize>,
    /// Total row dimension.
    pub(super) m: usize,
    /// Total column dimension.
    pub(super) n: usize,
}

/// Compute fused sector groups for a bipartition at `nrow`.
///
/// For each fused left sector with a matching fused right sector (determined
/// by flux), collects the left/right block-index tuples and their dimensions.
pub(super) fn compute_fused_sector_groups<T, S: Sector>(
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
) -> Vec<FusedSectorGroup<S>> {
    let indices = tensor.indices();
    let flux = tensor.flux();

    let left_groups = enumerate_fused_tuples(&indices[..nrow]);
    let right_groups = enumerate_fused_tuples(&indices[nrow..]);

    let mut result = Vec::new();
    for (s_l, left_entries) in &left_groups {
        // For abelian symmetry: left.fuse(right) = flux → right = left.dual().fuse(flux)
        let s_r = s_l.dual().fuse(flux);
        if let Some(right_entries) = right_groups.get(&s_r) {
            let left_tuples: Vec<Vec<usize>> =
                left_entries.iter().map(|(t, _)| t.clone()).collect();
            let right_tuples: Vec<Vec<usize>> =
                right_entries.iter().map(|(t, _)| t.clone()).collect();
            let left_dims: Vec<usize> = left_entries.iter().map(|(_, d)| *d).collect();
            let right_dims: Vec<usize> = right_entries.iter().map(|(_, d)| *d).collect();

            let left_offsets = cumulative_offsets(&left_dims);
            let m = left_dims.iter().sum();
            let right_offsets = cumulative_offsets(&right_dims);
            let n = right_dims.iter().sum();

            result.push(FusedSectorGroup {
                sector: s_l.clone(),
                left_tuples,
                right_tuples,
                left_dims,
                right_dims,
                left_offsets,
                right_offsets,
                m,
                n,
            });
        }
    }
    result
}

/// Enumerate all block-index tuples for a set of legs and group by fused sector.
///
/// Returns a map from directed fused sector to a list of (block-index tuple, block dimension)
/// pairs. The tuples are sorted lexicographically within each sector group.
pub(crate) fn enumerate_fused_tuples<S: Sector>(
    indices: &[QNIndex<S>],
) -> BTreeMap<S, Vec<(Vec<usize>, usize)>> {
    let mut groups: BTreeMap<S, Vec<(Vec<usize>, usize)>> = BTreeMap::new();
    let nlegs = indices.len();

    if nlegs == 0 {
        groups.entry(S::identity()).or_default().push((vec![], 1));
        return groups;
    }

    let num_blocks: Vec<usize> = indices.iter().map(|idx| idx.num_blocks()).collect();
    if num_blocks.contains(&0) {
        return groups;
    }

    let mut current = vec![0usize; nlegs];
    loop {
        let mut fused = S::identity();
        let mut dim = 1usize;
        for (i, &bi) in current.iter().enumerate() {
            let sector = indices[i].sector(bi);
            let directed = indices[i].direction().apply(sector);
            fused = fused.fuse(&directed);
            dim *= indices[i].block_dim(bi);
        }
        groups
            .entry(fused)
            .or_default()
            .push((current.clone(), dim));

        let mut carry = true;
        for i in (0..nlegs).rev() {
            current[i] += 1;
            if current[i] < num_blocks[i] {
                carry = false;
                break;
            }
            current[i] = 0;
        }
        if carry {
            break;
        }
    }
    groups
}

fn cumulative_offsets(dims: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(dims.len());
    let mut offset = 0;
    for &d in dims {
        offsets.push(offset);
        offset += d;
    }
    offsets
}

/// Assemble a dense matrix from all blocks in one fused sector group.
///
/// The output matrix layout follows the given `order`.
pub(super) fn assemble_sector_matrix<T: Scalar, S: Sector>(
    tensor: &BlockSparseTensorData<T, S>,
    group: &FusedSectorGroup<S>,
    order: MemoryOrder,
) -> Vec<T> {
    let m = group.m;
    let n = group.n;
    let mut matrix = vec![T::zero(); m * n];

    for (li, left_tup) in group.left_tuples.iter().enumerate() {
        let row_off = group.left_offsets[li];
        let m_i = group.left_dims[li];

        for (ri, right_tup) in group.right_tuples.iter().enumerate() {
            let col_off = group.right_offsets[ri];
            let n_j = group.right_dims[ri];

            let mut coord_vec = Vec::with_capacity(left_tup.len() + right_tup.len());
            coord_vec.extend_from_slice(left_tup);
            coord_vec.extend_from_slice(right_tup);
            let coord = BlockCoord(coord_vec);

            if let Some(block_data) = tensor.block_data(&coord) {
                match order {
                    MemoryOrder::RowMajor => {
                        for r in 0..m_i {
                            let dst = (row_off + r) * n + col_off;
                            let src = r * n_j;
                            matrix[dst..dst + n_j].copy_from_slice(&block_data[src..src + n_j]);
                        }
                    }
                    MemoryOrder::ColumnMajor => {
                        for c in 0..n_j {
                            let dst = (col_off + c) * m + row_off;
                            let src = c * m_i;
                            matrix[dst..dst + m_i].copy_from_slice(&block_data[src..src + m_i]);
                        }
                    }
                }
            }
        }
    }
    matrix
}

/// Build the left output tensor (U, Q, or L) from per-sector matrices.
///
/// Legs: `[original_left_legs..., bond(In)]`, flux = identity.
/// `order` specifies the memory layout of both source matrices and output block data.
pub(super) fn build_left_tensor<T: Scalar, S: Sector>(
    groups: &[FusedSectorGroup<S>],
    left_matrices: &[Vec<T>],
    k_per_sector: &[usize],
    original_indices: &[QNIndex<S>],
    nrow: usize,
    order: MemoryOrder,
) -> BlockSparseTensorData<T, S> {
    let bond_blocks: Vec<(S, usize)> = groups
        .iter()
        .zip(k_per_sector.iter())
        .filter(|&(_, &k)| k > 0)
        .map(|(g, &k)| (g.sector.clone(), k))
        .collect();
    let bond_index = QNIndex::new(bond_blocks, Direction::In);

    let mut out_indices: Vec<QNIndex<S>> = original_indices[..nrow].to_vec();
    out_indices.push(bond_index);
    let mut output = BlockSparseTensorData::zeros(out_indices, S::identity(), order);

    let mut bond_idx = 0;
    for (gi, group) in groups.iter().enumerate() {
        let k = k_per_sector[gi];
        if k == 0 {
            continue;
        }
        let m = group.m;
        let mat = &left_matrices[gi];
        for (li, left_tup) in group.left_tuples.iter().enumerate() {
            let row_off = group.left_offsets[li];
            let m_i = group.left_dims[li];
            let mut coord_vec = left_tup.clone();
            coord_vec.push(bond_idx);
            let coord = BlockCoord(coord_vec);
            let block_data = output
                .block_data_mut(&coord)
                .expect("internal error: missing output block in build_left_tensor");
            match order {
                MemoryOrder::RowMajor => {
                    for r in 0..m_i {
                        let src = (row_off + r) * k;
                        let dst = r * k;
                        block_data[dst..dst + k].copy_from_slice(&mat[src..src + k]);
                    }
                }
                MemoryOrder::ColumnMajor => {
                    for c in 0..k {
                        let src = c * m + row_off;
                        let dst = c * m_i;
                        block_data[dst..dst + m_i].copy_from_slice(&mat[src..src + m_i]);
                    }
                }
            }
        }
        bond_idx += 1;
    }
    output
}

/// Build the right output tensor (Vt, R, or Q) from per-sector matrices.
///
/// Legs: `[bond(Out), original_right_legs...]`, flux = original flux.
/// `order` specifies the memory layout of both source matrices and output block data.
pub(super) fn build_right_tensor<T: Scalar, S: Sector>(
    groups: &[FusedSectorGroup<S>],
    right_matrices: &[Vec<T>],
    k_per_sector: &[usize],
    original_indices: &[QNIndex<S>],
    nrow: usize,
    flux: S,
    order: MemoryOrder,
) -> BlockSparseTensorData<T, S> {
    let bond_blocks: Vec<(S, usize)> = groups
        .iter()
        .zip(k_per_sector.iter())
        .filter(|&(_, &k)| k > 0)
        .map(|(g, &k)| (g.sector.clone(), k))
        .collect();
    let bond_index = QNIndex::new(bond_blocks, Direction::Out);

    let mut out_indices: Vec<QNIndex<S>> = vec![bond_index];
    out_indices.extend_from_slice(&original_indices[nrow..]);
    let mut output = BlockSparseTensorData::zeros(out_indices, flux, order);

    let mut bond_idx = 0;
    for (gi, group) in groups.iter().enumerate() {
        let k = k_per_sector[gi];
        if k == 0 {
            continue;
        }
        let n = group.n;
        let mat = &right_matrices[gi];
        for (ri, right_tup) in group.right_tuples.iter().enumerate() {
            let col_off = group.right_offsets[ri];
            let n_j = group.right_dims[ri];
            let mut coord_vec = vec![bond_idx];
            coord_vec.extend_from_slice(right_tup);
            let coord = BlockCoord(coord_vec);
            let block_data = output
                .block_data_mut(&coord)
                .expect("internal error: missing output block in build_right_tensor");
            match order {
                MemoryOrder::RowMajor => {
                    for r in 0..k {
                        let src = r * n + col_off;
                        let dst = r * n_j;
                        block_data[dst..dst + n_j].copy_from_slice(&mat[src..src + n_j]);
                    }
                }
                MemoryOrder::ColumnMajor => {
                    for c in 0..n_j {
                        let src = (col_off + c) * k;
                        let dst = c * k;
                        block_data[dst..dst + k].copy_from_slice(&mat[src..src + k]);
                    }
                }
            }
        }
        bond_idx += 1;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::cumulative_offsets;

    /// Multi-tuple input with `block_dim > 1` per tuple — kills the
    /// `+= with -=` and `+= with *=` mutants on the offset accumulator.
    /// (`-=` underflows on usize on the first iteration since `0 -= 2`
    /// wraps in release / panics in debug; `*=` produces `[0, 0, 0]`
    /// since `0 *= d = 0`.)
    #[test]
    fn cumulative_offsets_multi_tuple_multi_dim() {
        assert_eq!(cumulative_offsets(&[2, 3, 4]), vec![0, 2, 5]);
    }

    #[test]
    fn cumulative_offsets_empty_input() {
        assert_eq!(cumulative_offsets(&[]), Vec::<usize>::new());
    }

    #[test]
    fn cumulative_offsets_single_tuple() {
        assert_eq!(cumulative_offsets(&[7]), vec![0]);
    }
}
