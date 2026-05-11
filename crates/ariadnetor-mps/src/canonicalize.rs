//! Canonicalize: move the orthogonality center of a tensor chain via QR/LQ sweeps

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::{
    BlockSparseContractResult, contract, contract_block_sparse, lq, lq_block_sparse, qr,
    qr_block_sparse,
};
use arnet_tensor::BlockSparse;
use arnet_tensor::Sector;
use arnet_tensor::{Dense, reorder};

use super::chain::TensorChain;
use super::types::CanonicalForm;

/// Move the orthogonality center of a tensor chain to the specified site.
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
    C: TensorChain<Dense<T>, B>,
{
    let n = chain.len();
    assert!(
        center < n,
        "center {center} out of range for chain of length {n}"
    );

    // Left-to-right QR sweep: make sites 0..center left-canonical
    for j in 0..center {
        left_qr_step(chain, j);
    }

    // Right-to-left LQ sweep: make sites center+1..N right-canonical
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
    C: TensorChain<Dense<T>, B>,
{
    // QR decomposition: group all modes except the last as "rows"
    let (q_storage, r) = {
        let dense = chain.storage(j);
        let rank = dense.rank();
        let orig_shape = dense.shape().to_vec();
        let order = chain.backend().preferred_order();

        let (q, r) = qr(chain.backend(), dense, rank - 1)
            .expect("QR decomposition failed during canonicalize");

        // Reshape Q from (m, k) back to (*orig[..rank-1], k).
        // Reorder to row-major for correct axis-split semantics, then back.
        let q_rm = reorder(&q, order, arnet_core::MemoryOrder::RowMajor);
        let k = q_rm.shape()[1];
        let mut q_shape = orig_shape[..rank - 1].to_vec();
        q_shape.push(k);
        let q_multi = q_rm.reshape(q_shape);
        let q_back = reorder(&q_multi, arnet_core::MemoryOrder::RowMajor, order);

        (q_back, r)
    };

    *chain.storage_mut(j) = q_storage;

    // Absorb R into site j+1: R(k, old_bond) × next(old_bond, ...) → (k, ...)
    let new_next = {
        let next = chain.storage(j + 1);
        absorb_from_left(&r, next, chain.backend())
    };

    *chain.storage_mut(j + 1) = new_next;
}

/// LQ step: decompose site j, replace with Q, absorb L into site j-1.
fn right_lq_step<T, B, C>(chain: &mut C, j: usize)
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<Dense<T>, B>,
{
    // LQ decomposition: group only the first mode as "rows"
    let (q_storage, l) = {
        let dense = chain.storage(j);
        let orig_shape = dense.shape().to_vec();
        let order = chain.backend().preferred_order();

        let (l, q) =
            lq(chain.backend(), dense, 1).expect("LQ decomposition failed during canonicalize");

        // Reshape Q from (k, n) back to (k, *orig[1..]).
        // Reorder to row-major for correct axis-split semantics, then back.
        let q_rm = reorder(&q, order, arnet_core::MemoryOrder::RowMajor);
        let k = q_rm.shape()[0];
        let mut q_shape = vec![k];
        q_shape.extend_from_slice(&orig_shape[1..]);
        let q_multi = q_rm.reshape(q_shape);
        let q_back = reorder(&q_multi, arnet_core::MemoryOrder::RowMajor, order);

        (q_back, l)
    };

    *chain.storage_mut(j) = q_storage;

    // Absorb L into site j-1: prev(..., old_bond) × L(old_bond, k) → (..., k)
    let new_prev = {
        let prev = chain.storage(j - 1);
        absorb_from_right(prev, &l, chain.backend())
    };

    *chain.storage_mut(j - 1) = new_prev;
}

/// Multiply R matrix into the next site: R(k, d) × next(d, ...) → (k, ...).
/// Reshapes next to 2D for matmul, then restores original rank.
fn absorb_from_left<T: Scalar>(
    r: &Dense<T>,
    next: &Dense<T>,
    backend: &impl ComputeBackend,
) -> Dense<T> {
    let order = backend.preferred_order();
    // Reorder to RM for correct axis-merge semantics in reshape. Use
    // `next.order()` as the source — site tensors stored on an MPS
    // are not guaranteed to be in `backend.preferred_order()` once
    // callers can construct Dense with explicit `source_order`.
    let next_rm = reorder(next, next.order(), arnet_core::MemoryOrder::RowMajor);
    let next_shape = next_rm.shape().to_vec();
    let first = next_shape[0];
    let rest: usize = next_shape[1..].iter().product();

    // Reshape to 2D in RM, then convert to backend order for contract.
    let next_2d_rm = next_rm.reshape(vec![first, rest]);
    let next_2d = reorder(&next_2d_rm, arnet_core::MemoryOrder::RowMajor, order);
    let result_2d = contract(backend, r, &next_2d, "ab,bc->ac")
        .expect("R absorption into next site failed during canonicalize");

    // Reorder result to RM for axis-split reshape, then back to backend order.
    let result_2d_rm = reorder(&result_2d, order, arnet_core::MemoryOrder::RowMajor);
    let k = r.shape()[0];
    let mut new_shape = next_shape;
    new_shape[0] = k;
    let result_multi = result_2d_rm.reshape(new_shape);
    reorder(&result_multi, arnet_core::MemoryOrder::RowMajor, order)
}

/// Multiply L matrix into the previous site: prev(..., d) × L(d, k) → (..., k).
/// Reshapes prev to 2D for matmul, then restores original rank.
fn absorb_from_right<T: Scalar>(
    prev: &Dense<T>,
    l: &Dense<T>,
    backend: &impl ComputeBackend,
) -> Dense<T> {
    let order = backend.preferred_order();
    // Reorder to RM for correct axis-merge semantics in reshape. Use
    // `prev.order()` as the source — site tensors stored on an MPS
    // are not guaranteed to be in `backend.preferred_order()` once
    // callers can construct Dense with explicit `source_order`.
    let prev_rm = reorder(prev, prev.order(), arnet_core::MemoryOrder::RowMajor);
    let prev_shape = prev_rm.shape().to_vec();
    let last = *prev_shape.last().unwrap();
    let rest: usize = prev_shape[..prev_shape.len() - 1].iter().product();

    // Reshape to 2D in RM, then convert to backend order for contract.
    let prev_2d_rm = prev_rm.reshape(vec![rest, last]);
    let prev_2d = reorder(&prev_2d_rm, arnet_core::MemoryOrder::RowMajor, order);
    let result_2d = contract(backend, &prev_2d, l, "ab,bc->ac")
        .expect("L absorption into previous site failed during canonicalize");

    // Reorder result to RM for axis-split reshape, then back to backend order.
    let result_2d_rm = reorder(&result_2d, order, arnet_core::MemoryOrder::RowMajor);
    let k = l.shape()[1];
    let mut new_shape = prev_shape;
    *new_shape.last_mut().unwrap() = k;
    let result_multi = result_2d_rm.reshape(new_shape);
    reorder(&result_multi, arnet_core::MemoryOrder::RowMajor, order)
}

// ============================================================================
// BlockSparse canonicalize — parallel path for Mps<BlockSparse<T, S>, B>
// ============================================================================

/// Move the orthogonality center of a block-sparse tensor chain.
///
/// BlockSparse analogue of [`canonicalize`]. Performs left-to-right QR sweeps
/// and right-to-left LQ sweeps using [`qr_block_sparse`] / [`lq_block_sparse`]
/// and [`contract_block_sparse`]. After completion, the canonical form is
/// `Mixed { center }`.
///
/// Works for both MPS (rank-3) and MPO (rank-4) tensor chains whose sites are
/// `BlockSparse<T, S>`.
///
/// Unlike the Dense path, the block-sparse primitives preserve the site rank
/// across decomposition, so no intermediate reshape / row-major conversion is
/// required: `qr_block_sparse` with `nrow = rank - 1` returns a rank-`rank` Q
/// and a rank-2 R that matches the original right bond.
///
/// # Per-site flux under asymmetric QR / LQ
///
/// The block-sparse decomposition primitives are asymmetric in where they
/// park the input tensor's flux:
///
/// - `qr_block_sparse` returns an isometric `Q` with `flux = identity()` and
///   a residual `R` that inherits the original flux. Absorbing `R` into the
///   right neighbor therefore moves the site's charge one step rightward,
///   and the full left-to-right sweep accumulates all per-site charges from
///   sites `0..center` onto the orthogonality center.
///
/// - `lq_block_sparse` puts the original flux on the isometric `Q` (which
///   stays in place) and returns an identity-flux `L`. The right-to-left
///   sweep therefore preserves each site's flux label; every site in
///   `center+1..N` ends up right-isometric while carrying its original
///   per-site charge.
///
/// Both outcomes are valid canonical forms: block-sparse isometry is a
/// per-sector orthogonality condition that holds regardless of the tensor's
/// overall flux label. Callers using the conventional zero-flux MPS
/// encoding (every site starts at `identity()`) observe no flux motion,
/// while charged chains are canonicalized without panic — the resulting
/// per-site flux distribution simply reflects the asymmetric sweep.
///
/// # Panics
///
/// Panics if `center >= chain.len()` or if the chain is empty.
pub(super) fn canonicalize_bsp<T, S, B, C>(chain: &mut C, center: usize)
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparse<T, S>, B>,
{
    let n = chain.len();
    assert!(
        center < n,
        "center {center} out of range for chain of length {n}"
    );

    // Left-to-right QR sweep: make sites 0..center left-canonical.
    for j in 0..center {
        left_qr_step_block_sparse(chain, j);
    }

    // Right-to-left LQ sweep: make sites center+1..N right-canonical.
    for j in (center + 1..n).rev() {
        right_lq_step_block_sparse(chain, j);
    }

    chain.set_canonical_form(CanonicalForm::Mixed { center });
}

/// Block-sparse QR step: decompose site j, replace with Q, absorb R into site j+1.
fn left_qr_step_block_sparse<T, S, B, C>(chain: &mut C, j: usize)
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparse<T, S>, B>,
{
    // QR decomposition: group all modes except the last as "rows".
    // For nrow = rank - 1, Q inherits legs [original[..rank-1], new_bond(In)]
    // and R has legs [new_bond(Out), original[rank-1]].
    let (q, r) = {
        let site = chain.storage(j);
        let rank = site.rank();
        qr_block_sparse(chain.backend(), site, rank - 1)
            .expect("qr_block_sparse failed during canonicalize")
    };

    *chain.storage_mut(j) = q;

    // Absorb R into site j+1: R(new_bond, old_right_bond) × next(old_left_bond, ...)
    // → (new_bond, ...). R's axis 1 is the original right bond of site j, which pairs
    // with site j+1's axis 0 by construction; contract_block_sparse validates the
    // opposing directions.
    let new_next = {
        let next = chain.storage(j + 1);
        absorb_from_left_block_sparse(&r, next, chain.backend())
    };

    *chain.storage_mut(j + 1) = new_next;
}

/// Block-sparse LQ step: decompose site j, replace with Q, absorb L into site j-1.
fn right_lq_step_block_sparse<T, S, B, C>(chain: &mut C, j: usize)
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparse<T, S>, B>,
{
    // LQ decomposition: group only the first mode as "rows".
    // For nrow = 1, L has legs [original[0], new_bond(In)] and Q has legs
    // [new_bond(Out), original[1..]].
    let (l, q) = {
        let site = chain.storage(j);
        lq_block_sparse(chain.backend(), site, 1)
            .expect("lq_block_sparse failed during canonicalize")
    };

    *chain.storage_mut(j) = q;

    // Absorb L into site j-1: prev(..., old_right_bond) × L(old_left_bond, new_bond)
    // → (..., new_bond). prev's last axis pairs with L.axis(0), which is the original
    // left bond of site j before decomposition.
    let new_prev = {
        let prev = chain.storage(j - 1);
        absorb_from_right_block_sparse(prev, &l, chain.backend())
    };

    *chain.storage_mut(j - 1) = new_prev;
}

/// Multiply R into the next site from the left: `R(new, old_right) × next(old_left, ...)`.
///
/// Scalar result cannot occur because the output rank is
/// `r.rank() + next.rank() - 2 >= 3`, so the matcher panic arm is unreachable by
/// construction.
fn absorb_from_left_block_sparse<T, S, B>(
    r: &BlockSparse<T, S>,
    next: &BlockSparse<T, S>,
    backend: &B,
) -> BlockSparse<T, S>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    match contract_block_sparse(backend, r, next, &[1], &[0])
        .expect("R absorption into next site failed during canonicalize")
    {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => {
            unreachable!("left absorption contraction always produces a tensor")
        }
    }
}

/// Multiply L into the previous site from the right: `prev(..., old_right) × L(old_left, new)`.
///
/// Scalar result cannot occur for the same rank accounting reason as
/// [`absorb_from_left_block_sparse`].
fn absorb_from_right_block_sparse<T, S, B>(
    prev: &BlockSparse<T, S>,
    l: &BlockSparse<T, S>,
    backend: &B,
) -> BlockSparse<T, S>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let last = prev.rank() - 1;
    match contract_block_sparse(backend, prev, l, &[last], &[0])
        .expect("L absorption into previous site failed during canonicalize")
    {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => {
            unreachable!("right absorption contraction always produces a tensor")
        }
    }
}
