//! Canonicalize: move the orthogonality center of a tensor chain via
//! QR / LQ sweeps.

use arnet::{
    BlockSparseContractResult, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor,
    ComputeBackend, DenseLayout, DenseStorage, DenseTensor, Scalar, Sector, contract,
    contract_block_sparse, lq, lq_block_sparse, qr, qr_block_sparse,
};

use super::chain::TensorChain;
use super::types::CanonicalForm;

/// Move the orthogonality center of a Dense tensor chain to the specified site.
///
/// Performs left-to-right QR sweeps (sites 0..center) and right-to-left LQ
/// sweeps (sites N-1..center+1). After completion, the canonical form is
/// `Mixed { center }`.
///
/// Works for both MPS (rank-3) and MPO (rank-4) tensor chains.
///
/// # Panics
///
/// Panics if `center >= chain.len()` or if the chain is empty.
pub(super) fn canonicalize_dense<T, B, C>(chain: &mut C, center: usize)
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<DenseStorage<T>, DenseLayout, B>,
{
    let n = chain.len();
    assert!(
        center < n,
        "center {center} out of range for chain of length {n}",
    );

    // Left-to-right QR sweep: make sites 0..center left-canonical.
    for j in 0..center {
        left_qr_step(chain, j);
    }

    // Right-to-left LQ sweep: make sites center+1..N right-canonical.
    for j in (center + 1..n).rev() {
        right_lq_step(chain, j);
    }

    chain.set_canonical_form(CanonicalForm::Mixed { center });
}

/// QR step: decompose site j, replace with Q, absorb R into site j+1.
fn left_qr_step<T, B, C>(chain: &mut C, j: usize)
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<DenseStorage<T>, DenseLayout, B>,
{
    let (q_tensor, r) = {
        let site = chain.site(j);
        let rank = site.rank();
        let orig_shape = site.shape().to_vec();

        let (q, r) = qr(site, rank - 1).expect("QR: validated by entry point");

        // Split Q's fused row leg (m, k) back into (*orig[..rank-1], k).
        let q_back = q.split_leg(0, &orig_shape[..rank - 1]);

        (q_back, r)
    };

    *chain.site_mut(j) = q_tensor;

    // Absorb R into site j+1.
    let new_next = {
        let next = chain.site(j + 1);
        absorb_from_left(&r, next)
    };

    *chain.site_mut(j + 1) = new_next;
}

/// LQ step: decompose site j, replace with Q, absorb L into site j-1.
fn right_lq_step<T, B, C>(chain: &mut C, j: usize)
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<DenseStorage<T>, DenseLayout, B>,
{
    let (q_tensor, l) = {
        let site = chain.site(j);
        let orig_shape = site.shape().to_vec();

        let (l, q) = lq(site, 1).expect("LQ: validated by entry point");

        // Split Q's fused column leg (k, m) back into (k, *orig[1..]).
        let q_back = q.split_leg(1, &orig_shape[1..]);

        (q_back, l)
    };

    *chain.site_mut(j) = q_tensor;

    let new_prev = {
        let prev = chain.site(j - 1);
        absorb_from_right(prev, &l)
    };

    *chain.site_mut(j - 1) = new_prev;
}

/// Multiply R matrix into the next site: `R(k, d) × next(d, ...) → (k, ...)`.
/// Fuses next's trailing legs to a matrix for the matmul, then splits the
/// result's fused leg back to restore the original rank. The logical leg
/// operations handle the memory-order round-trip internally.
fn absorb_from_left<T, B>(r: &DenseTensor<T, B>, next: &DenseTensor<T, B>) -> DenseTensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    // Fuse next's trailing legs into a matrix, contract R · next, then
    // split the fused leg back; axis 0 carries R's new bond.
    let next_shape = next.shape().to_vec();
    let next_2d = next.fuse_legs(1..next_shape.len());
    let result_2d =
        contract(r, &next_2d, "ab,bc->ac").expect("R absorption: validated by entry point");
    result_2d.split_leg(1, &next_shape[1..])
}

/// Multiply L matrix into the previous site: `prev(..., d) × L(d, k) → (..., k)`.
fn absorb_from_right<T, B>(prev: &DenseTensor<T, B>, l: &DenseTensor<T, B>) -> DenseTensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    // Fuse prev's leading legs into a matrix, contract prev · L, then
    // split the fused leg back; the last axis carries L's new bond.
    let prev_shape = prev.shape().to_vec();
    let split = prev_shape.len() - 1;
    let prev_2d = prev.fuse_legs(0..split);
    let result_2d =
        contract(&prev_2d, l, "ab,bc->ac").expect("L absorption: validated by entry point");
    result_2d.split_leg(0, &prev_shape[..split])
}

// ============================================================================
// BlockSparse canonicalize.
// ============================================================================

/// Move the orthogonality center of a BlockSparse tensor chain.
///
/// BlockSparse analogue of [`canonicalize_dense`]. The block-sparse
/// decomposition primitives preserve rank and bond directions, so no
/// reshape / row-major conversion is needed.
///
/// # Per-site flux under asymmetric QR / LQ
///
/// - `qr_block_sparse` returns an isometric `Q` with `flux = identity()`
///   and a residual `R` carrying the original flux. The left-to-right
///   sweep therefore accumulates per-site charges onto the
///   orthogonality center.
/// - `lq_block_sparse` puts the original flux on the isometric `Q` and
///   returns an identity-flux `L`. The right-to-left sweep preserves
///   each site's flux.
///
/// Both outcomes are valid canonical forms.
///
/// # Panics
///
/// Panics if `center >= chain.len()` or if the chain is empty.
pub(super) fn canonicalize_bsp<T, S, B, C>(chain: &mut C, center: usize)
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
{
    let n = chain.len();
    assert!(
        center < n,
        "center {center} out of range for chain of length {n}",
    );

    for j in 0..center {
        left_qr_step_bsp(chain, j);
    }

    for j in (center + 1..n).rev() {
        right_lq_step_bsp(chain, j);
    }

    chain.set_canonical_form(CanonicalForm::Mixed { center });
}

fn left_qr_step_bsp<T, S, B, C>(chain: &mut C, j: usize)
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
{
    let (q, r) = {
        let site = chain.site(j);
        let rank = site.rank();
        qr_block_sparse(site, rank - 1).expect("QR: validated by entry point")
    };

    *chain.site_mut(j) = q;

    let new_next = {
        let next = chain.site(j + 1);
        absorb_from_left_bsp(&r, next)
    };

    *chain.site_mut(j + 1) = new_next;
}

fn right_lq_step_bsp<T, S, B, C>(chain: &mut C, j: usize)
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
{
    let (l, q) = {
        let site = chain.site(j);
        lq_block_sparse(site, 1).expect("LQ: validated by entry point")
    };

    *chain.site_mut(j) = q;

    let new_prev = {
        let prev = chain.site(j - 1);
        absorb_from_right_bsp(prev, &l)
    };

    *chain.site_mut(j - 1) = new_prev;
}

fn absorb_from_left_bsp<T, S, B>(
    r: &BlockSparseTensor<T, S, B>,
    next: &BlockSparseTensor<T, S, B>,
) -> BlockSparseTensor<T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    match contract_block_sparse(r, next, &[1], &[0])
        .expect("R absorption: validated by entry point")
    {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => {
            unreachable!("left absorption contraction always produces a tensor")
        }
    }
}

fn absorb_from_right_bsp<T, S, B>(
    prev: &BlockSparseTensor<T, S, B>,
    l: &BlockSparseTensor<T, S, B>,
) -> BlockSparseTensor<T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let last = prev.rank() - 1;
    match contract_block_sparse(prev, l, &[last], &[0])
        .expect("L absorption: validated by entry point")
    {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => {
            unreachable!("right absorption contraction always produces a tensor")
        }
    }
}
