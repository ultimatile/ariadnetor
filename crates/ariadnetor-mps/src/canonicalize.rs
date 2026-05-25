//! Canonicalize: move the orthogonality center of a tensor chain via
//! QR / LQ sweeps.

use arnet::{
    BlockSparseContractResult, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor,
    ComputeBackend, DenseLayout, DenseStorage, DenseTensor, MemoryOrder, Scalar, Sector, contract,
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
        let order = chain.backend().preferred_order();

        let (q, r) = qr(site, rank - 1).expect("QR: validated by entry point");

        // Reshape Q from (m, k) back to (*orig[..rank-1], k). Convert to
        // RowMajor for correct axis-split semantics, then back to backend
        // order.
        let q_rm = q.reordered(MemoryOrder::RowMajor);
        let k = q_rm.shape()[1];
        let mut q_shape = orig_shape[..rank - 1].to_vec();
        q_shape.push(k);
        let q_multi = q_rm.reshape(q_shape);
        let q_back = q_multi.reordered(order);

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
        let order = chain.backend().preferred_order();

        let (l, q) = lq(site, 1).expect("LQ: validated by entry point");

        let q_rm = q.reordered(MemoryOrder::RowMajor);
        let k = q_rm.shape()[0];
        let mut q_shape = vec![k];
        q_shape.extend_from_slice(&orig_shape[1..]);
        let q_multi = q_rm.reshape(q_shape);
        let q_back = q_multi.reordered(order);

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
/// Reshapes next to 2D for matmul, then restores original rank.
///
/// Site / factor tensors are guaranteed to carry the backend's preferred
/// order under the Tier 1 / Tier 2 ordering invariant, so the source
/// order at the RowMajor conversion boundary is the backend's preferred
/// order rather than a per-tensor read.
fn absorb_from_left<T, B>(r: &DenseTensor<T, B>, next: &DenseTensor<T, B>) -> DenseTensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let order = next.backend().preferred_order();
    let next_rm = next.reordered(MemoryOrder::RowMajor);
    let next_shape = next_rm.shape().to_vec();
    let first = next_shape[0];
    let rest: usize = next_shape[1..].iter().product();

    let next_2d_rm = next_rm.reshape(vec![first, rest]);
    let next_2d = next_2d_rm.reordered(order);
    let result_2d =
        contract(r, &next_2d, "ab,bc->ac").expect("R absorption: validated by entry point");

    let result_2d_rm = result_2d.reordered(MemoryOrder::RowMajor);
    let k = r.shape()[0];
    let mut new_shape = next_shape;
    new_shape[0] = k;
    let result_multi = result_2d_rm.reshape(new_shape);
    result_multi.reordered(order)
}

/// Multiply L matrix into the previous site: `prev(..., d) × L(d, k) → (..., k)`.
fn absorb_from_right<T, B>(prev: &DenseTensor<T, B>, l: &DenseTensor<T, B>) -> DenseTensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let order = prev.backend().preferred_order();
    let prev_rm = prev.reordered(MemoryOrder::RowMajor);
    let prev_shape = prev_rm.shape().to_vec();
    let last = *prev_shape.last().unwrap();
    let rest: usize = prev_shape[..prev_shape.len() - 1].iter().product();

    let prev_2d_rm = prev_rm.reshape(vec![rest, last]);
    let prev_2d = prev_2d_rm.reordered(order);
    let result_2d =
        contract(&prev_2d, l, "ab,bc->ac").expect("L absorption: validated by entry point");

    let result_2d_rm = result_2d.reordered(MemoryOrder::RowMajor);
    let k = l.shape()[1];
    let mut new_shape = prev_shape;
    *new_shape.last_mut().unwrap() = k;
    let result_multi = result_2d_rm.reshape(new_shape);
    result_multi.reordered(order)
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
