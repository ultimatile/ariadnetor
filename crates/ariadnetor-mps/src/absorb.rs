//! Shared absorb helpers: multiply a factor matrix into an adjacent site.
//!
//! Both the canonicalize and truncate sweeps push a residual factor (R / L
//! from QR / LQ, or the scaled SVD factors from a truncated SVD) into the
//! neighbouring site. The contraction is identical across both sweeps and
//! across the Dense / BlockSparse storages, so it lives here as one
//! definition per shape rather than duplicated in each sweep file.
//!
//! Every caller validates the contraction's preconditions at its entry
//! point, so the internal `expect` failures are unreachable in practice.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{contract, tensordot};
use ariadnetor_tensor::{
    BlockSparseStorage, BlockSparseTensor, DenseStorage, DenseTensor, OpsFor, Sector,
};

/// Multiply a factor matrix into the next site: `factor(k, d) × next(d, ...) → (k, ...)`.
/// Fuses next's trailing legs to a matrix for the matmul, then splits the
/// result's fused leg back to restore the original rank. The logical leg
/// operations handle the memory-order round-trip internally.
pub(crate) fn absorb_from_left<T, B>(
    factor: &DenseTensor<T>,
    next: &DenseTensor<T>,
    backend: &B,
) -> DenseTensor<T>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    // Fuse next's trailing legs into a matrix, contract factor · next, then
    // split the fused leg back; axis 0 carries the factor's new bond.
    let next_shape = next.shape().to_vec();
    let next_2d = next.fuse_legs(1..next_shape.len());
    let result_2d = contract(backend, factor, &next_2d, "ab,bc->ac")
        .expect("left absorption: validated by entry point");
    result_2d.split_leg(1, &next_shape[1..])
}

/// Multiply a factor matrix into the previous site: `prev(..., d) × factor(d, k) → (..., k)`.
pub(crate) fn absorb_from_right<T, B>(
    prev: &DenseTensor<T>,
    factor: &DenseTensor<T>,
    backend: &B,
) -> DenseTensor<T>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    // Fuse prev's leading legs into a matrix, contract prev · factor, then
    // split the fused leg back; the last axis carries the factor's new bond.
    let prev_shape = prev.shape().to_vec();
    let split = prev_shape.len() - 1;
    let prev_2d = prev.fuse_legs(0..split);
    let result_2d = contract(backend, &prev_2d, factor, "ab,bc->ac")
        .expect("right absorption: validated by entry point");
    result_2d.split_leg(0, &prev_shape[..split])
}

/// BlockSparse analogue of [`absorb_from_left`]: contract the factor's bond
/// leg (axis 1) against the next site's leading leg (axis 0).
pub(crate) fn absorb_from_left_bsp<T, S, B>(
    factor: &BlockSparseTensor<T, S>,
    next: &BlockSparseTensor<T, S>,
    backend: &B,
) -> BlockSparseTensor<T, S>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    tensordot(backend, factor, next, &[1], &[0]).expect("left absorption: validated by entry point")
}

/// BlockSparse analogue of [`absorb_from_right`]: contract the prev site's
/// trailing leg against the factor's leading leg (axis 0).
pub(crate) fn absorb_from_right_bsp<T, S, B>(
    prev: &BlockSparseTensor<T, S>,
    factor: &BlockSparseTensor<T, S>,
    backend: &B,
) -> BlockSparseTensor<T, S>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    let last = prev.rank() - 1;
    tensordot(backend, prev, factor, &[last], &[0])
        .expect("right absorption: validated by entry point")
}
