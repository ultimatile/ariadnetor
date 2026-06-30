//! Canonicalize: move the orthogonality center of a tensor chain via
//! QR / LQ sweeps.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{lq, qr};
use ariadnetor_tensor::{
    BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, OpsFor, Sector,
};

use super::absorb::{
    absorb_from_left, absorb_from_left_bsp, absorb_from_right, absorb_from_right_bsp,
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
pub(super) fn canonicalize_dense<T, B, C>(backend: &B, chain: &mut C, center: usize)
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
    C: TensorChain<DenseStorage<T>, DenseLayout>,
{
    let n = chain.len();
    assert!(
        center < n,
        "center {center} out of range for chain of length {n}",
    );

    // Left-to-right QR sweep: make sites 0..center left-canonical.
    for j in 0..center {
        left_qr_step(chain, j, backend);
    }

    // Right-to-left LQ sweep: make sites center+1..N right-canonical.
    for j in (center + 1..n).rev() {
        right_lq_step(chain, j, backend);
    }

    chain.set_canonical_form(CanonicalForm::Mixed { center });
}

/// QR step: decompose site j, replace with Q, absorb R into site j+1.
fn left_qr_step<T, B, C>(chain: &mut C, j: usize, backend: &B)
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
    C: TensorChain<DenseStorage<T>, DenseLayout>,
{
    let (q_tensor, r) = {
        let site = chain.site(j);
        let rank = site.rank();
        let orig_shape = site.shape().to_vec();

        let (q, r) = qr(backend, site, rank - 1).expect("QR: validated by entry point");

        // Split Q's fused row leg (m, k) back into (*orig[..rank-1], k).
        let q_back = q.split_leg(0, &orig_shape[..rank - 1]);

        (q_back, r)
    };

    *chain.site_mut(j) = q_tensor;

    // Absorb R into site j+1.
    let new_next = {
        let next = chain.site(j + 1);
        absorb_from_left(&r, next, backend)
    };

    *chain.site_mut(j + 1) = new_next;
}

/// LQ step: decompose site j, replace with Q, absorb L into site j-1.
fn right_lq_step<T, B, C>(chain: &mut C, j: usize, backend: &B)
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
    C: TensorChain<DenseStorage<T>, DenseLayout>,
{
    let (q_tensor, l) = {
        let site = chain.site(j);
        let orig_shape = site.shape().to_vec();

        let (l, q) = lq(backend, site, 1).expect("LQ: validated by entry point");

        // Split Q's fused column leg (k, m) back into (k, *orig[1..]).
        let q_back = q.split_leg(1, &orig_shape[1..]);

        (q_back, l)
    };

    *chain.site_mut(j) = q_tensor;

    let new_prev = {
        let prev = chain.site(j - 1);
        absorb_from_right(prev, &l, backend)
    };

    *chain.site_mut(j - 1) = new_prev;
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
/// - `qr` returns an isometric `Q` with
///   `flux = identity()`
///   and a residual `R` carrying the original flux. The left-to-right
///   sweep therefore accumulates per-site charges onto the
///   orthogonality center.
/// - `lq` puts the original flux on the isometric `Q` and
///   returns an identity-flux `L`. The right-to-left sweep preserves
///   each site's flux.
///
/// Both outcomes are valid canonical forms.
///
/// # Panics
///
/// Panics if `center >= chain.len()` or if the chain is empty.
pub(super) fn canonicalize_bsp<T, S, B, C>(backend: &B, chain: &mut C, center: usize)
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>>,
{
    let n = chain.len();
    assert!(
        center < n,
        "center {center} out of range for chain of length {n}",
    );

    for j in 0..center {
        left_qr_step_bsp(chain, j, backend);
    }

    for j in (center + 1..n).rev() {
        right_lq_step_bsp(chain, j, backend);
    }

    chain.set_canonical_form(CanonicalForm::Mixed { center });
}

fn left_qr_step_bsp<T, S, B, C>(chain: &mut C, j: usize, backend: &B)
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>>,
{
    let (q, r) = {
        let site = chain.site(j);
        let rank = site.rank();
        qr(backend, site, rank - 1).expect("QR: validated by entry point")
    };

    *chain.site_mut(j) = q;

    let new_next = {
        let next = chain.site(j + 1);
        absorb_from_left_bsp(&r, next, backend)
    };

    *chain.site_mut(j + 1) = new_next;
}

fn right_lq_step_bsp<T, S, B, C>(chain: &mut C, j: usize, backend: &B)
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>>,
{
    let (l, q) = {
        let site = chain.site(j);
        lq(backend, site, 1).expect("LQ: validated by entry point")
    };

    *chain.site_mut(j) = q;

    let new_prev = {
        let prev = chain.site(j - 1);
        absorb_from_right_bsp(prev, &l, backend)
    };

    *chain.site_mut(j - 1) = new_prev;
}
