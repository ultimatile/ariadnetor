//! Truncate: reduce bond dimensions of a tensor chain via SVD sweeps

use arnet_core::backend::ComputeBackend;
use arnet_core::scalar::Scalar;
use arnet_linalg::{TruncSvdParams, contract, diagonal_scale, trunc_svd};
use arnet_tensor::{DenseTensor, MemoryOrder, TensorStorage};
use num_traits::{Float, Zero};

use super::chain::TensorChain;
use super::types::CanonicalForm;

/// Truncate bond dimensions of a canonicalized tensor chain.
///
/// Performs SVD sweeps from the orthogonality center outward in both
/// directions, applying truncation at each bond. Returns the total
/// truncation error (Frobenius norm of discarded singular values).
///
/// # Panics
///
/// Panics if the chain is not in `Canonicalized` state.
pub fn truncate<T, B, C>(chain: &mut C, params: &TruncSvdParams) -> T::Real
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<T, B>,
{
    let center = match chain.canonical_form() {
        CanonicalForm::Canonicalized { center } => *center,
        _ => panic!("truncate requires Canonicalized state; call orthogonalize first"),
    };

    let n = chain.len();
    if n <= 1 {
        chain.set_canonical_form(CanonicalForm::Canonicalized { center });
        return T::Real::zero();
    }

    let mut total_err_sq = T::Real::zero();

    // Right sweep from center to N-2: truncate bonds, center moves to N-1
    for j in center..n - 1 {
        total_err_sq = total_err_sq + right_trunc_step(chain, j, params);
    }

    // Left sweep from N-1 to 1: truncate all bonds, center moves to 0
    for j in (1..n).rev() {
        total_err_sq = total_err_sq + left_trunc_step(chain, j, params);
    }

    // Right sweep from 0 to center-1: restore center position
    for j in 0..center {
        total_err_sq = total_err_sq + right_trunc_step(chain, j, params);
    }

    chain.set_canonical_form(CanonicalForm::Canonicalized { center });
    total_err_sq.sqrt()
}

/// Right SVD step at site j: U → left-canonical at j, S·Vt → absorbed into j+1.
/// Returns squared truncation error.
fn right_trunc_step<T, B, C>(chain: &mut C, j: usize, params: &TruncSvdParams) -> T::Real
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<T, B>,
{
    let (u_storage, svt, err) = {
        let dense = as_dense(chain.storage(j));
        let rank = dense.rank();
        let orig_shape = dense.shape().to_vec();

        let (u, s, vt, err) = trunc_svd(chain.backend(), dense, rank - 1, params)
            .expect("trunc_svd failed during truncate");

        // Reshape U from (m, chi) to (*orig[..rank-1], chi).
        // Convert to row-major first so reshape uses standard axis merge order.
        let u_rm = u.to_contiguous(MemoryOrder::RowMajor);
        let chi = u_rm.shape()[1];
        let mut u_shape = orig_shape[..rank - 1].to_vec();
        u_shape.push(chi);

        // S·Vt: scale each row of Vt by corresponding singular value
        let svt = diagonal_scale(&vt, s.data(), 0).expect("S·Vt scaling failed during truncate");

        (TensorStorage::Dense(u_rm.reshape(u_shape)), svt, err)
    };

    *chain.storage_mut(j) = u_storage;

    // Absorb S·Vt into site j+1
    let new_next = {
        let next = as_dense(chain.storage(j + 1));
        absorb_from_left(&svt, next, chain.backend())
    };

    *chain.storage_mut(j + 1) = TensorStorage::Dense(new_next);

    err * err
}

/// Left SVD step at site j: Vt → right-canonical at j, U·S → absorbed into j-1.
/// Returns squared truncation error.
fn left_trunc_step<T, B, C>(chain: &mut C, j: usize, params: &TruncSvdParams) -> T::Real
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<T, B>,
{
    let (vt_storage, us, err) = {
        let dense = as_dense(chain.storage(j));
        let orig_shape = dense.shape().to_vec();

        let (u, s, vt, err) =
            trunc_svd(chain.backend(), dense, 1, params).expect("trunc_svd failed during truncate");

        // Reshape Vt from (chi, n) to (chi, *orig[1..]).
        // Convert to row-major first so reshape uses standard axis merge order.
        let vt_rm = vt.to_contiguous(MemoryOrder::RowMajor);
        let chi = vt_rm.shape()[0];
        let mut vt_shape = vec![chi];
        vt_shape.extend_from_slice(&orig_shape[1..]);

        // U·S: scale each column of U by corresponding singular value
        let us = diagonal_scale(&u, s.data(), 1).expect("U·S scaling failed during truncate");

        (TensorStorage::Dense(vt_rm.reshape(vt_shape)), us, err)
    };

    *chain.storage_mut(j) = vt_storage;

    // Absorb U·S into site j-1
    let new_prev = {
        let prev = as_dense(chain.storage(j - 1));
        absorb_from_right(prev, &us, chain.backend())
    };

    *chain.storage_mut(j - 1) = TensorStorage::Dense(new_prev);

    err * err
}

/// Multiply a 2D matrix into the next site tensor from the left.
fn absorb_from_left<T: Scalar>(
    left: &DenseTensor<T>,
    next: &DenseTensor<T>,
    backend: &impl ComputeBackend,
) -> DenseTensor<T> {
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
    prev: &DenseTensor<T>,
    right: &DenseTensor<T>,
    backend: &impl ComputeBackend,
) -> DenseTensor<T> {
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

fn as_dense<T>(storage: &TensorStorage<T>) -> &DenseTensor<T> {
    match storage {
        TensorStorage::Dense(d) => d,
    }
}
