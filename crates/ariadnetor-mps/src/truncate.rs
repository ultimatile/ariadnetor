//! Truncate: reduce bond dimensions of a tensor chain via SVD sweeps

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::{
    BlockSparseContractResult, TruncSvdParams, contract, contract_block_sparse, diagonal_scale,
    diagonal_scale_block_sparse, trunc_svd, trunc_svd_block_sparse,
};
use arnet_tensor::BlockSparse;
use arnet_tensor::Sector;
use arnet_tensor::{Dense, reorder};
use num_traits::{Float, Zero};

use super::canonicalize::{canonicalize_bsp, canonicalize_dense};
use super::chain::TensorChain;
use super::types::{CanonicalForm, SvdAbsorb, TruncResult, TruncateParams};

/// Truncate bond dimensions of a tensor chain via SVD sweeps.
///
/// Performs SVD sweeps from the orthogonality center outward in both
/// directions, applying truncation at each bond. Returns the total
/// truncation error (Frobenius norm of discarded singular values).
///
/// If the chain is not in `Mixed` canonical form, auto-canonicalizes first:
/// - `Mixed`: uses existing center (no extra work).
/// - `Left`: treats the last site as center.
/// - `Right`: treats site 0 as center.
/// - `Partial` / `Unknown`: canonicalizes to `params.center` (default 0).
///
/// # Panics
///
/// Panics if the chain is empty, or if `params.center` is `Some(c)` with
/// `c >= chain.len()`.
pub(super) fn truncate_dense<T, B, C>(chain: &mut C, params: &TruncateParams) -> TruncResult<T>
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<Dense<T>, B>,
{
    let n = chain.len();
    assert!(n > 0, "truncate requires a non-empty chain");

    let center = match chain.canonical_form() {
        CanonicalForm::Mixed { center } => *center,
        CanonicalForm::Left => n - 1,
        CanonicalForm::Right => 0,
        _ => {
            let c = params.center.unwrap_or(0);
            canonicalize_dense(chain, c);
            c
        }
    };
    if n <= 1 {
        chain.set_canonical_form(CanonicalForm::Mixed { center });
        return TruncResult {
            error: T::Real::zero(),
        };
    }

    let svd_params = &params.svd;
    let absorb = params.absorb;
    let mut total_err_sq = T::Real::zero();

    // Right sweep from center to N-2: truncate bonds, center moves to N-1
    for j in center..n - 1 {
        total_err_sq = total_err_sq + right_trunc_step(chain, j, svd_params, absorb);
    }

    // Left sweep from N-1 to 1: truncate all bonds, center moves to 0
    for j in (1..n).rev() {
        total_err_sq = total_err_sq + left_trunc_step(chain, j, svd_params, absorb);
    }

    // Right sweep from 0 to center-1: restore center position
    for j in 0..center {
        total_err_sq = total_err_sq + right_trunc_step(chain, j, svd_params, absorb);
    }

    // Both distributes √S to both sides, breaking isometry on all sites.
    let form = match absorb {
        SvdAbsorb::Both => CanonicalForm::Unknown,
        _ => CanonicalForm::Mixed { center },
    };
    chain.set_canonical_form(form);
    TruncResult {
        error: total_err_sq.sqrt(),
    }
}

/// Right SVD step at site j: decompose, absorb into j+1 based on absorb mode.
/// Returns squared truncation error.
fn right_trunc_step<T, B, C>(
    chain: &mut C,
    j: usize,
    params: &TruncSvdParams,
    absorb: SvdAbsorb,
) -> T::Real
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<Dense<T>, B>,
{
    let order = chain.backend().preferred_order();
    let rm = arnet_core::MemoryOrder::RowMajor;

    let (left_storage, right_factor, err) = {
        let dense = chain.storage(j);
        let rank = dense.rank();
        let orig_shape = dense.shape().to_vec();

        let (u, s, vt, err) = trunc_svd(chain.backend(), dense, rank - 1, params)
            .expect("trunc_svd failed during truncate");

        let chi = u.shape()[1];
        let mut u_shape = orig_shape[..rank - 1].to_vec();
        u_shape.push(chi);

        // Helper: reshape U from 2D backend-order to multi-dim in backend order.
        // Converts to RM for axis-split semantics, then back.
        let reshape_u = |u_2d: Dense<T>| -> Dense<T> {
            let u_rm = reorder(&u_2d, order, rm);
            let multi = u_rm.reshape(u_shape.clone());
            reorder(&multi, rm, order)
        };

        match absorb {
            SvdAbsorb::Right => {
                // U stays at j (left-canonical), S·Vt absorbed into j+1
                let svt = diagonal_scale(chain.backend(), &vt, s.data(), 0)
                    .expect("S·Vt scaling failed during truncate");
                (reshape_u(u), svt, err)
            }
            SvdAbsorb::Left => {
                // U·S stays at j, Vt absorbed into j+1 (right-canonical Vt)
                let us = diagonal_scale(chain.backend(), &u, s.data(), 1)
                    .expect("U·S scaling failed during truncate");
                (reshape_u(us), vt, err)
            }
            SvdAbsorb::Both => {
                // sqrt(S) applied to both sides
                let sqrt_s: Vec<T::Real> = s.data().iter().map(|v| v.sqrt()).collect();
                let u_scaled = diagonal_scale(chain.backend(), &u, &sqrt_s, 1)
                    .expect("sqrt(S)*U scaling failed during truncate");
                let vt_scaled = diagonal_scale(chain.backend(), &vt, &sqrt_s, 0)
                    .expect("sqrt(S)*Vt scaling failed during truncate");
                (reshape_u(u_scaled), vt_scaled, err)
            }
        }
    };

    *chain.storage_mut(j) = left_storage;

    let new_next = {
        let next = chain.storage(j + 1);
        absorb_from_left(&right_factor, next, chain.backend())
    };

    *chain.storage_mut(j + 1) = new_next;

    err * err
}

/// Left SVD step at site j: decompose, absorb into j-1 based on absorb mode.
/// Returns squared truncation error.
fn left_trunc_step<T, B, C>(
    chain: &mut C,
    j: usize,
    params: &TruncSvdParams,
    absorb: SvdAbsorb,
) -> T::Real
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<Dense<T>, B>,
{
    let order = chain.backend().preferred_order();
    let rm = arnet_core::MemoryOrder::RowMajor;

    let (right_storage, left_factor, err) = {
        let dense = chain.storage(j);
        let orig_shape = dense.shape().to_vec();

        let (u, s, vt, err) =
            trunc_svd(chain.backend(), dense, 1, params).expect("trunc_svd failed during truncate");

        let chi = vt.shape()[0];
        let mut vt_shape = vec![chi];
        vt_shape.extend_from_slice(&orig_shape[1..]);

        // Helper: reshape Vt from 2D backend-order to multi-dim in backend order.
        let reshape_vt = |vt_2d: Dense<T>| -> Dense<T> {
            let vt_rm = reorder(&vt_2d, order, rm);
            let multi = vt_rm.reshape(vt_shape.clone());
            reorder(&multi, rm, order)
        };

        match absorb {
            SvdAbsorb::Right => {
                // Vt stays at j (right-isometric), U·S absorbed into j-1
                // S accompanies the sweep direction (leftward), producing mixed-canonical form.
                let us = diagonal_scale(chain.backend(), &u, s.data(), 1)
                    .expect("U·S scaling failed during truncate");
                (reshape_vt(vt), us, err)
            }
            SvdAbsorb::Left => {
                // S·Vt stays at j, bare U absorbed into j-1
                // S stays against the sweep direction.
                let svt = diagonal_scale(chain.backend(), &vt, s.data(), 0)
                    .expect("S·Vt scaling failed during truncate");
                (reshape_vt(svt), u, err)
            }
            SvdAbsorb::Both => {
                // sqrt(S) applied to both sides
                let sqrt_s: Vec<T::Real> = s.data().iter().map(|v| v.sqrt()).collect();
                let vt_scaled = diagonal_scale(chain.backend(), &vt, &sqrt_s, 0)
                    .expect("sqrt(S)*Vt scaling failed during truncate");
                let u_scaled = diagonal_scale(chain.backend(), &u, &sqrt_s, 1)
                    .expect("sqrt(S)*U scaling failed during truncate");
                (reshape_vt(vt_scaled), u_scaled, err)
            }
        }
    };

    *chain.storage_mut(j) = right_storage;

    let new_prev = {
        let prev = chain.storage(j - 1);
        absorb_from_right(prev, &left_factor, chain.backend())
    };

    *chain.storage_mut(j - 1) = new_prev;

    err * err
}

/// Multiply a 2D matrix into the next site tensor from the left.
fn absorb_from_left<T: Scalar>(
    left: &Dense<T>,
    next: &Dense<T>,
    backend: &impl ComputeBackend,
) -> Dense<T> {
    let order = backend.preferred_order();
    let rm = arnet_core::MemoryOrder::RowMajor;
    // Reorder to RM for correct axis-merge semantics in reshape.
    let next_rm = reorder(next, order, rm);
    let next_shape = next_rm.shape().to_vec();
    let first = next_shape[0];
    let rest: usize = next_shape[1..].iter().product();

    let next_2d_rm = next_rm.reshape(vec![first, rest]);
    let next_2d = reorder(&next_2d_rm, rm, order);
    let result_2d = contract(backend, left, &next_2d, "ab,bc->ac")
        .expect("left absorption failed during truncate");

    // Reorder result to RM for axis-split reshape, then back to backend order.
    let result_2d_rm = reorder(&result_2d, order, rm);
    let k = left.shape()[0];
    let mut new_shape = next_shape;
    new_shape[0] = k;
    let result_multi = result_2d_rm.reshape(new_shape);
    reorder(&result_multi, rm, order)
}

/// Multiply a 2D matrix into the previous site tensor from the right.
fn absorb_from_right<T: Scalar>(
    prev: &Dense<T>,
    right: &Dense<T>,
    backend: &impl ComputeBackend,
) -> Dense<T> {
    let order = backend.preferred_order();
    let rm = arnet_core::MemoryOrder::RowMajor;
    // Reorder to RM for correct axis-merge semantics in reshape.
    let prev_rm = reorder(prev, order, rm);
    let prev_shape = prev_rm.shape().to_vec();
    let last = *prev_shape.last().unwrap();
    let rest: usize = prev_shape[..prev_shape.len() - 1].iter().product();

    let prev_2d_rm = prev_rm.reshape(vec![rest, last]);
    let prev_2d = reorder(&prev_2d_rm, rm, order);
    let result_2d = contract(backend, &prev_2d, right, "ab,bc->ac")
        .expect("right absorption failed during truncate");

    // Reorder result to RM for axis-split reshape, then back to backend order.
    let result_2d_rm = reorder(&result_2d, order, rm);
    let k = right.shape()[1];
    let mut new_shape = prev_shape;
    *new_shape.last_mut().unwrap() = k;
    let result_multi = result_2d_rm.reshape(new_shape);
    reorder(&result_multi, rm, order)
}

// ============================================================================
// BlockSparse truncate — parallel path for Mps<BlockSparse<T, S>, B>
// ============================================================================

/// Truncate bond dimensions of a block-sparse tensor chain via SVD sweeps.
///
/// BlockSparse analogue of [`truncate`]. Uses [`trunc_svd_block_sparse`],
/// [`diagonal_scale_block_sparse`], and [`contract_block_sparse`] instead of
/// their Dense counterparts. No reshape or memory-order conversion is needed
/// because block-sparse SVD preserves the leg structure.
///
/// See [`truncate`] for the sweep structure and canonical-form semantics.
///
/// # Panics
///
/// Panics if the chain is empty, or if `params.center` is `Some(c)` with
/// `c >= chain.len()` and the chain is not already in `Mixed`, `Left`, or
/// `Right` canonical form (since those forms determine the center
/// internally and ignore `params.center`).
pub(super) fn truncate_bsp<T, S, B, C>(chain: &mut C, params: &TruncateParams) -> TruncResult<T>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparse<T, S>, B>,
{
    let n = chain.len();
    assert!(n > 0, "truncate requires a non-empty chain");

    let center = match chain.canonical_form() {
        CanonicalForm::Mixed { center } => *center,
        CanonicalForm::Left => n - 1,
        CanonicalForm::Right => 0,
        _ => {
            let c = params.center.unwrap_or(0);
            canonicalize_bsp(chain, c);
            c
        }
    };
    if n <= 1 {
        chain.set_canonical_form(CanonicalForm::Mixed { center });
        return TruncResult {
            error: T::Real::zero(),
        };
    }

    let svd_params = &params.svd;
    let absorb = params.absorb;
    let mut total_err_sq = T::Real::zero();

    // Right sweep from center to N-2: truncate bonds, center moves to N-1
    for j in center..n - 1 {
        total_err_sq = total_err_sq + right_trunc_step_block_sparse(chain, j, svd_params, absorb);
    }

    // Left sweep from N-1 to 1: truncate all bonds, center moves to 0
    for j in (1..n).rev() {
        total_err_sq = total_err_sq + left_trunc_step_block_sparse(chain, j, svd_params, absorb);
    }

    // Right sweep from 0 to center-1: restore center position
    for j in 0..center {
        total_err_sq = total_err_sq + right_trunc_step_block_sparse(chain, j, svd_params, absorb);
    }

    // Both distributes √S to both sides, breaking isometry on all sites.
    let form = match absorb {
        SvdAbsorb::Both => CanonicalForm::Unknown,
        _ => CanonicalForm::Mixed { center },
    };
    chain.set_canonical_form(form);
    TruncResult {
        error: total_err_sq.sqrt(),
    }
}

/// Right SVD step at site j for block-sparse chains.
/// Returns squared truncation error.
fn right_trunc_step_block_sparse<T, S, B, C>(
    chain: &mut C,
    j: usize,
    params: &TruncSvdParams,
    absorb: SvdAbsorb,
) -> T::Real
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparse<T, S>, B>,
{
    let (left_storage, right_factor, err) = {
        let site = chain.storage(j);
        let rank = site.rank();

        let (u, s, vt, err) = trunc_svd_block_sparse(chain.backend(), site, rank - 1, params)
            .expect("trunc_svd_block_sparse failed during truncate");

        match absorb {
            SvdAbsorb::Right => {
                // U stays at j (left-canonical), S·Vt absorbed into j+1
                let svt = diagonal_scale_block_sparse(chain.backend(), &vt, &s, 0)
                    .expect("S·Vt scaling failed during truncate");
                (u, svt, err)
            }
            SvdAbsorb::Left => {
                // U·S stays at j, Vt absorbed into j+1
                let us = diagonal_scale_block_sparse(chain.backend(), &u, &s, u.rank() - 1)
                    .expect("U·S scaling failed during truncate");
                (us, vt, err)
            }
            SvdAbsorb::Both => {
                // √S applied to both sides
                let sqrt_s = s.map(|v| (*v).sqrt());
                let u_scaled =
                    diagonal_scale_block_sparse(chain.backend(), &u, &sqrt_s, u.rank() - 1)
                        .expect("√S·U scaling failed during truncate");
                let vt_scaled = diagonal_scale_block_sparse(chain.backend(), &vt, &sqrt_s, 0)
                    .expect("√S·Vt scaling failed during truncate");
                (u_scaled, vt_scaled, err)
            }
        }
    };

    *chain.storage_mut(j) = left_storage;

    let new_next = {
        let next = chain.storage(j + 1);
        absorb_from_left_bsp(&right_factor, next, chain.backend())
    };

    *chain.storage_mut(j + 1) = new_next;

    err * err
}

/// Left SVD step at site j for block-sparse chains.
/// Returns squared truncation error.
fn left_trunc_step_block_sparse<T, S, B, C>(
    chain: &mut C,
    j: usize,
    params: &TruncSvdParams,
    absorb: SvdAbsorb,
) -> T::Real
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparse<T, S>, B>,
{
    let (right_storage, left_factor, err) = {
        let site = chain.storage(j);

        let (u, s, vt, err) = trunc_svd_block_sparse(chain.backend(), site, 1, params)
            .expect("trunc_svd_block_sparse failed during truncate");

        match absorb {
            SvdAbsorb::Right => {
                // Vt stays at j (right-isometric), U·S absorbed into j-1
                let us = diagonal_scale_block_sparse(chain.backend(), &u, &s, u.rank() - 1)
                    .expect("U·S scaling failed during truncate");
                (vt, us, err)
            }
            SvdAbsorb::Left => {
                // S·Vt stays at j, bare U absorbed into j-1
                let svt = diagonal_scale_block_sparse(chain.backend(), &vt, &s, 0)
                    .expect("S·Vt scaling failed during truncate");
                (svt, u, err)
            }
            SvdAbsorb::Both => {
                // √S applied to both sides
                let sqrt_s = s.map(|v| (*v).sqrt());
                let vt_scaled = diagonal_scale_block_sparse(chain.backend(), &vt, &sqrt_s, 0)
                    .expect("√S·Vt scaling failed during truncate");
                let u_scaled =
                    diagonal_scale_block_sparse(chain.backend(), &u, &sqrt_s, u.rank() - 1)
                        .expect("√S·U scaling failed during truncate");
                (vt_scaled, u_scaled, err)
            }
        }
    };

    *chain.storage_mut(j) = right_storage;

    let new_prev = {
        let prev = chain.storage(j - 1);
        absorb_from_right_bsp(prev, &left_factor, chain.backend())
    };

    *chain.storage_mut(j - 1) = new_prev;

    err * err
}

/// Multiply a rank-2 factor into the next block-sparse site from the left.
fn absorb_from_left_bsp<T, S, B>(
    left: &BlockSparse<T, S>,
    next: &BlockSparse<T, S>,
    backend: &B,
) -> BlockSparse<T, S>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    match contract_block_sparse(backend, left, next, &[1], &[0])
        .expect("left absorption failed during truncate")
    {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => {
            unreachable!("left absorption contraction always produces a tensor")
        }
    }
}

/// Multiply a rank-2 factor into the previous block-sparse site from the right.
fn absorb_from_right_bsp<T, S, B>(
    prev: &BlockSparse<T, S>,
    right: &BlockSparse<T, S>,
    backend: &B,
) -> BlockSparse<T, S>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let last = prev.rank() - 1;
    match contract_block_sparse(backend, prev, right, &[last], &[0])
        .expect("right absorption failed during truncate")
    {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => {
            unreachable!("right absorption contraction always produces a tensor")
        }
    }
}
