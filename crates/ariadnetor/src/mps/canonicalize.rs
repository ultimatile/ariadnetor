//! Canonicalize: move the orthogonality center of a tensor chain via QR/LQ sweeps

use arnet_core::backend::ComputeBackend;
use arnet_core::scalar::Scalar;
use arnet_linalg::{
    BlockSparseContractResult, contract, contract_block_sparse, lq, lq_block_sparse, qr,
    qr_block_sparse,
};
use arnet_tensor::block_sparse::BlockSparse;
use arnet_tensor::sector::Sector;
use arnet_tensor::{Dense, MemoryOrder};

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
pub fn canonicalize<T, B, C>(chain: &mut C, center: usize)
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

        let (q, r) = qr(chain.backend(), dense, rank - 1)
            .expect("QR decomposition failed during canonicalize");

        // Reshape Q from (m, k) back to (*orig[..rank-1], k).
        // Convert to row-major first so reshape uses standard axis merge order.
        let q_rm = q.to_contiguous(MemoryOrder::RowMajor);
        let k = q_rm.shape()[1];
        let mut q_shape = orig_shape[..rank - 1].to_vec();
        q_shape.push(k);

        (q_rm.reshape(q_shape), r)
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

        let (l, q) =
            lq(chain.backend(), dense, 1).expect("LQ decomposition failed during canonicalize");

        // Reshape Q from (k, n) back to (k, *orig[1..]).
        // Convert to row-major first so reshape uses standard axis merge order.
        let q_rm = q.to_contiguous(MemoryOrder::RowMajor);
        let k = q_rm.shape()[0];
        let mut q_shape = vec![k];
        q_shape.extend_from_slice(&orig_shape[1..]);

        (q_rm.reshape(q_shape), l)
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
    // Ensure row-major so reshape uses standard axis merge order.
    let next = next.to_contiguous(MemoryOrder::RowMajor);
    let next_shape = next.shape().to_vec();
    let first = next_shape[0];
    let rest: usize = next_shape[1..].iter().product();

    let next_2d = next.reshape(vec![first, rest]);
    let result_2d = contract(backend, r, &next_2d, "ab,bc->ac")
        .expect("R absorption into next site failed during canonicalize");

    // Convert to row-major before rank-restoring reshape (axis split semantics).
    let result_2d = result_2d.to_contiguous(MemoryOrder::RowMajor);
    let k = r.shape()[0];
    let mut new_shape = next_shape;
    new_shape[0] = k;
    result_2d.reshape(new_shape)
}

/// Multiply L matrix into the previous site: prev(..., d) × L(d, k) → (..., k).
/// Reshapes prev to 2D for matmul, then restores original rank.
fn absorb_from_right<T: Scalar>(
    prev: &Dense<T>,
    l: &Dense<T>,
    backend: &impl ComputeBackend,
) -> Dense<T> {
    // Ensure row-major so reshape uses standard axis merge order.
    let prev = prev.to_contiguous(MemoryOrder::RowMajor);
    let prev_shape = prev.shape().to_vec();
    let last = *prev_shape.last().unwrap();
    let rest: usize = prev_shape[..prev_shape.len() - 1].iter().product();

    let prev_2d = prev.reshape(vec![rest, last]);
    let result_2d = contract(backend, &prev_2d, l, "ab,bc->ac")
        .expect("L absorption into previous site failed during canonicalize");

    // Convert to row-major before rank-restoring reshape (axis split semantics).
    let result_2d = result_2d.to_contiguous(MemoryOrder::RowMajor);
    let k = l.shape()[1];
    let mut new_shape = prev_shape;
    *new_shape.last_mut().unwrap() = k;
    result_2d.reshape(new_shape)
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
/// # Precondition: per-site identity flux
///
/// This function requires every site in `chain` to already carry
/// `flux = identity()`, which matches the conventional block-sparse MPS
/// encoding (total charge is represented by the boundary bond sectors, not by
/// per-tensor flux labels).
///
/// The two sweeps are not symmetric in how they propagate non-identity per-
/// site flux: `qr_block_sparse` returns an isometric `Q` with identity flux
/// and puts the original flux into `R`, which this sweep then absorbs into
/// the right neighbor, shifting left-side flux toward the center. In contrast
/// `lq_block_sparse` puts the original flux on the isometric `Q` and returns
/// an identity-flux `L`, so the right-to-left sweep leaves right-side flux
/// untouched. Supporting canonicalization of arbitrarily charged per-site
/// fluxes requires a different decomposition path and is tracked as future
/// work; accepting such chains silently would yield an asymmetric result that
/// is neither what callers expect nor a valid canonical form.
///
/// # Panics
///
/// Panics if `center >= chain.len()`, if the chain is empty, or if any site
/// has non-identity flux.
pub fn canonicalize_block_sparse<T, S, B, C>(chain: &mut C, center: usize)
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

    // Reject charged per-site flux up front: the sweeps below would otherwise
    // produce an asymmetric, non-canonical result (see function docstring).
    let identity = S::identity();
    for j in 0..n {
        assert!(
            *chain.storage(j).flux() == identity,
            "canonicalize_block_sparse requires site {j} to have identity flux; \
             charged per-site flux is not yet supported"
        );
    }

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
            .expect("qr_block_sparse failed during canonicalize_block_sparse")
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
            .expect("lq_block_sparse failed during canonicalize_block_sparse")
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
        .expect("R absorption into next site failed during canonicalize_block_sparse")
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
        .expect("L absorption into previous site failed during canonicalize_block_sparse")
    {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => {
            unreachable!("right absorption contraction always produces a tensor")
        }
    }
}
