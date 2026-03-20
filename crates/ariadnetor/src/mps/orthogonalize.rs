//! Orthogonalize: move the orthogonality center of a tensor chain via QR/LQ sweeps

use arnet_core::backend::ComputeBackend;
use arnet_core::scalar::Scalar;
use arnet_linalg::{contract, lq, qr};
use arnet_tensor::{DenseTensor, TensorStorage};

use super::chain::TensorChain;
use super::types::CanonicalForm;

/// Move the orthogonality center of a tensor chain to the specified site.
///
/// Performs left-to-right QR sweeps (sites 0..center) and right-to-left LQ
/// sweeps (sites N-1..center+1). After completion, the canonical form is
/// `Canonicalized { center }`.
///
/// Works for both MPS (rank-3) and MPO (rank-4) tensor chains.
///
/// # Panics
///
/// Panics if `center >= chain.len()` or if the chain is empty.
pub fn orthogonalize<T, B, C>(chain: &mut C, center: usize)
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<T, B>,
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

    chain.set_canonical_form(CanonicalForm::Canonicalized { center });
}

/// QR step: decompose site j, replace with Q, absorb R into site j+1.
fn left_qr_step<T, B, C>(chain: &mut C, j: usize)
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<T, B>,
{
    // QR decomposition: group all modes except the last as "rows"
    let (q_storage, r) = {
        let dense = as_dense(chain.storage(j));
        let rank = dense.rank();
        let orig_shape = dense.shape().to_vec();

        let (q, r) = qr(chain.backend(), dense, rank - 1)
            .expect("QR decomposition failed during orthogonalize");

        // Reshape Q from (m, k) back to (*orig[..rank-1], k)
        let k = q.shape()[1];
        let mut q_shape = orig_shape[..rank - 1].to_vec();
        q_shape.push(k);

        (TensorStorage::Dense(q.reshape(q_shape)), r)
    };

    *chain.storage_mut(j) = q_storage;

    // Absorb R into site j+1: R(k, old_bond) × next(old_bond, ...) → (k, ...)
    let new_next = {
        let next = as_dense(chain.storage(j + 1));
        absorb_from_left(&r, next, chain.backend())
    };

    *chain.storage_mut(j + 1) = TensorStorage::Dense(new_next);
}

/// LQ step: decompose site j, replace with Q, absorb L into site j-1.
fn right_lq_step<T, B, C>(chain: &mut C, j: usize)
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<T, B>,
{
    // LQ decomposition: group only the first mode as "rows"
    let (q_storage, l) = {
        let dense = as_dense(chain.storage(j));
        let orig_shape = dense.shape().to_vec();

        let (l, q) =
            lq(chain.backend(), dense, 1).expect("LQ decomposition failed during orthogonalize");

        // Reshape Q from (k, n) back to (k, *orig[1..])
        let k = q.shape()[0];
        let mut q_shape = vec![k];
        q_shape.extend_from_slice(&orig_shape[1..]);

        (TensorStorage::Dense(q.reshape(q_shape)), l)
    };

    *chain.storage_mut(j) = q_storage;

    // Absorb L into site j-1: prev(..., old_bond) × L(old_bond, k) → (..., k)
    let new_prev = {
        let prev = as_dense(chain.storage(j - 1));
        absorb_from_right(prev, &l, chain.backend())
    };

    *chain.storage_mut(j - 1) = TensorStorage::Dense(new_prev);
}

/// Multiply R matrix into the next site: R(k, d) × next(d, ...) → (k, ...).
/// Reshapes next to 2D for matmul, then restores original rank.
fn absorb_from_left<T: Scalar>(
    r: &DenseTensor<T>,
    next: &DenseTensor<T>,
    backend: &impl ComputeBackend,
) -> DenseTensor<T> {
    let next_shape = next.shape().to_vec();
    let first = next_shape[0];
    let rest: usize = next_shape[1..].iter().product();

    let next_2d = next.reshape(vec![first, rest]);
    let result_2d = contract(backend, r, &next_2d, "ab,bc->ac")
        .expect("R absorption into next site failed during orthogonalize");

    let k = r.shape()[0];
    let mut new_shape = next_shape;
    new_shape[0] = k;
    result_2d.reshape(new_shape)
}

/// Multiply L matrix into the previous site: prev(..., d) × L(d, k) → (..., k).
/// Reshapes prev to 2D for matmul, then restores original rank.
fn absorb_from_right<T: Scalar>(
    prev: &DenseTensor<T>,
    l: &DenseTensor<T>,
    backend: &impl ComputeBackend,
) -> DenseTensor<T> {
    let prev_shape = prev.shape().to_vec();
    let last = *prev_shape.last().unwrap();
    let rest: usize = prev_shape[..prev_shape.len() - 1].iter().product();

    let prev_2d = prev.reshape(vec![rest, last]);
    let result_2d = contract(backend, &prev_2d, l, "ab,bc->ac")
        .expect("L absorption into previous site failed during orthogonalize");

    let k = l.shape()[1];
    let mut new_shape = prev_shape;
    *new_shape.last_mut().unwrap() = k;
    result_2d.reshape(new_shape)
}

fn as_dense<T>(storage: &TensorStorage<T>) -> &DenseTensor<T> {
    match storage {
        TensorStorage::Dense(d) => d,
    }
}
