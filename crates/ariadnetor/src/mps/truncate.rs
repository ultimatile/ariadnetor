//! Truncate: reduce bond dimensions of a tensor chain via SVD sweeps

use arnet_core::backend::ComputeBackend;
use arnet_core::scalar::Scalar;
use arnet_linalg::{TruncSvdParams, contract, diagonal_scale, trunc_svd};
use arnet_tensor::{Dense, MemoryOrder};
use num_traits::{Float, Zero};

use super::chain::TensorChain;
use super::orthogonalize::orthogonalize;
use super::types::{CanonicalForm, SvdAbsorb, TruncResult, TruncateParams};

/// Truncate bond dimensions of a tensor chain via SVD sweeps.
///
/// Performs SVD sweeps from the orthogonality center outward in both
/// directions, applying truncation at each bond. Returns the total
/// truncation error (Frobenius norm of discarded singular values).
///
/// If the chain is not in `Mixed` canonical form, auto-orthogonalizes first:
/// - `Mixed`: uses existing center (no extra work).
/// - `Left`: treats the last site as center.
/// - `Right`: treats site 0 as center.
/// - `Partial` / `Unknown`: orthogonalizes to `params.center` (default 0).
///
/// # Panics
///
/// Panics if the chain is empty, or if `params.center` is `Some(c)` with
/// `c >= chain.len()`.
pub fn truncate<T, B, C>(chain: &mut C, params: &TruncateParams) -> TruncResult<T>
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<T, B>,
{
    let n = chain.len();
    assert!(n > 0, "truncate requires a non-empty chain");

    let center = match chain.canonical_form() {
        CanonicalForm::Mixed { center } => *center,
        CanonicalForm::Left => n - 1,
        CanonicalForm::Right => 0,
        _ => {
            let c = params.center.unwrap_or(0);
            orthogonalize(chain, c);
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
    C: TensorChain<T, B>,
{
    let (left_storage, right_factor, err) = {
        let dense = chain.storage(j);
        let rank = dense.rank();
        let orig_shape = dense.shape().to_vec();

        let (u, s, vt, err) = trunc_svd(chain.backend(), dense, rank - 1, params)
            .expect("trunc_svd failed during truncate");

        let u_rm = u.to_contiguous(MemoryOrder::RowMajor);
        let chi = u_rm.shape()[1];
        let mut u_shape = orig_shape[..rank - 1].to_vec();
        u_shape.push(chi);

        match absorb {
            SvdAbsorb::Right => {
                // U stays at j (left-canonical), S·Vt absorbed into j+1
                let svt =
                    diagonal_scale(&vt, s.data(), 0).expect("S·Vt scaling failed during truncate");
                (u_rm.reshape(u_shape), svt, err)
            }
            SvdAbsorb::Left => {
                // U·S stays at j, Vt absorbed into j+1 (right-canonical Vt)
                let us =
                    diagonal_scale(&u_rm, s.data(), 1).expect("U·S scaling failed during truncate");
                let us = us.to_contiguous(MemoryOrder::RowMajor);
                let us_reshaped = us.reshape(u_shape);
                (us_reshaped, vt, err)
            }
            SvdAbsorb::Both => {
                // √S applied to both sides
                let sqrt_s: Vec<T::Real> = s.data().iter().map(|v| v.sqrt()).collect();
                let u_scaled =
                    diagonal_scale(&u_rm, &sqrt_s, 1).expect("√S·U scaling failed during truncate");
                let u_scaled = u_scaled.to_contiguous(MemoryOrder::RowMajor);
                let vt_scaled =
                    diagonal_scale(&vt, &sqrt_s, 0).expect("√S·Vt scaling failed during truncate");
                (u_scaled.reshape(u_shape), vt_scaled, err)
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
    C: TensorChain<T, B>,
{
    let (right_storage, left_factor, err) = {
        let dense = chain.storage(j);
        let orig_shape = dense.shape().to_vec();

        let (u, s, vt, err) =
            trunc_svd(chain.backend(), dense, 1, params).expect("trunc_svd failed during truncate");

        let vt_rm = vt.to_contiguous(MemoryOrder::RowMajor);
        let chi = vt_rm.shape()[0];
        let mut vt_shape = vec![chi];
        vt_shape.extend_from_slice(&orig_shape[1..]);

        match absorb {
            SvdAbsorb::Right => {
                // Vt stays at j (right-isometric), U·S absorbed into j-1
                // S accompanies the sweep direction (leftward), producing mixed-canonical form.
                let us =
                    diagonal_scale(&u, s.data(), 1).expect("U·S scaling failed during truncate");
                (vt_rm.reshape(vt_shape), us, err)
            }
            SvdAbsorb::Left => {
                // S·Vt stays at j, bare U absorbed into j-1
                // S stays against the sweep direction.
                let svt = diagonal_scale(&vt_rm, s.data(), 0)
                    .expect("S·Vt scaling failed during truncate");
                let svt = svt.to_contiguous(MemoryOrder::RowMajor);
                let svt_reshaped = svt.reshape(vt_shape);
                (svt_reshaped, u, err)
            }
            SvdAbsorb::Both => {
                // √S applied to both sides
                let sqrt_s: Vec<T::Real> = s.data().iter().map(|v| v.sqrt()).collect();
                let vt_scaled = diagonal_scale(&vt_rm, &sqrt_s, 0)
                    .expect("√S·Vt scaling failed during truncate");
                let vt_scaled = vt_scaled.to_contiguous(MemoryOrder::RowMajor);
                let u_scaled =
                    diagonal_scale(&u, &sqrt_s, 1).expect("√S·U scaling failed during truncate");
                (vt_scaled.reshape(vt_shape), u_scaled, err)
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
    // Ensure row-major so reshape uses standard axis merge order.
    let next = next.to_contiguous(MemoryOrder::RowMajor);
    let next_shape = next.shape().to_vec();
    let first = next_shape[0];
    let rest: usize = next_shape[1..].iter().product();

    let next_2d = next.reshape(vec![first, rest]);
    let result_2d = contract(backend, left, &next_2d, "ab,bc->ac")
        .expect("left absorption failed during truncate");

    // Convert to row-major before rank-restoring reshape (axis split semantics).
    let result_2d = result_2d.to_contiguous(MemoryOrder::RowMajor);
    let k = left.shape()[0];
    let mut new_shape = next_shape;
    new_shape[0] = k;
    result_2d.reshape(new_shape)
}

/// Multiply a 2D matrix into the previous site tensor from the right.
fn absorb_from_right<T: Scalar>(
    prev: &Dense<T>,
    right: &Dense<T>,
    backend: &impl ComputeBackend,
) -> Dense<T> {
    // Ensure row-major so reshape uses standard axis merge order.
    let prev = prev.to_contiguous(MemoryOrder::RowMajor);
    let prev_shape = prev.shape().to_vec();
    let last = *prev_shape.last().unwrap();
    let rest: usize = prev_shape[..prev_shape.len() - 1].iter().product();

    let prev_2d = prev.reshape(vec![rest, last]);
    let result_2d = contract(backend, &prev_2d, right, "ab,bc->ac")
        .expect("right absorption failed during truncate");

    // Convert to row-major before rank-restoring reshape (axis split semantics).
    let result_2d = result_2d.to_contiguous(MemoryOrder::RowMajor);
    let k = right.shape()[1];
    let mut new_shape = prev_shape;
    *new_shape.last_mut().unwrap() = k;
    result_2d.reshape(new_shape)
}
