//! Axis-permutation validation shared by the dense and block-sparse permute
//! kernels.

use crate::error::LinalgError;

/// Validate that `perm` is a permutation of the axes `0..rank`.
///
/// Three checks are necessary and sufficient: a map over the finite set
/// `0..rank` with `perm.len() == rank`, every entry in range, and no
/// duplicates is an injection from `0..rank` into itself, hence a bijection —
/// every axis is covered exactly once, so no separate coverage check is needed.
///
/// Returns [`LinalgError::InvalidArgument`] on any violation. The dense
/// `transpose_inner` and block-sparse `permute_block_sparse_dense` kernels both
/// call this at their entry so neither indexes a shape with an unchecked index.
pub(crate) fn validate_perm(perm: &[usize], rank: usize) -> Result<(), LinalgError> {
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
    Ok(())
}
