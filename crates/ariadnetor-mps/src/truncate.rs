//! Truncate: reduce bond dimensions of a tensor chain via SVD sweeps.

use arnet_core::Scalar;
use arnet_linalg::{
    TruncSvdParams, diagonal_scale_block_sparse_with_backend, diagonal_scale_with_backend,
    trunc_svd_block_sparse_with_backend, trunc_svd_with_backend,
};
use arnet_tensor::{
    BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, DenseTensor, OpsFor, Sector,
};
use num_traits::{Float, Zero};

use super::absorb::{
    absorb_from_left, absorb_from_left_bsp, absorb_from_right, absorb_from_right_bsp,
};
use super::canonicalize::{canonicalize_bsp, canonicalize_dense};
use super::chain::TensorChain;
use super::types::{CanonicalForm, SvdAbsorb, TruncResult, TruncateParams};

/// Truncate bond dimensions of a Dense tensor chain via SVD sweeps.
///
/// Performs SVD sweeps from the orthogonality center outward in both
/// directions, applying truncation at each bond. Returns the total
/// truncation error (Frobenius norm of discarded singular values).
///
/// If the chain is not in `Mixed` canonical form, auto-canonicalizes first.
pub(super) fn truncate_dense<T, B, C>(
    backend: &B,
    chain: &mut C,
    params: &TruncateParams,
) -> TruncResult<T>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
    C: TensorChain<DenseStorage<T>, DenseLayout>,
{
    let n = chain.len();
    assert!(n > 0, "truncate requires a non-empty chain");

    let center = match chain.canonical_form() {
        CanonicalForm::Mixed { center } => *center,
        CanonicalForm::Left => n - 1,
        CanonicalForm::Right => 0,
        _ => {
            let c = params.center.unwrap_or(0);
            canonicalize_dense(backend, chain, c);
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

    for j in center..n - 1 {
        total_err_sq = total_err_sq + right_trunc_step(chain, j, svd_params, absorb, backend);
    }

    for j in (1..n).rev() {
        total_err_sq = total_err_sq + left_trunc_step(chain, j, svd_params, absorb, backend);
    }

    // Final right sweep restoring the orthogonality center after the
    // preceding right and left sweeps; defensively accumulates any
    // residual squared error from each step in case of numerical drift.
    for j in 0..center {
        total_err_sq = total_err_sq + right_trunc_step(chain, j, svd_params, absorb, backend);
    }

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
    backend: &B,
) -> T::Real
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
    C: TensorChain<DenseStorage<T>, DenseLayout>,
{
    let (left_storage, right_factor, err) = {
        let site = chain.site(j);
        let rank = site.rank();
        let orig_shape = site.shape().to_vec();

        let (u, s, vt, err) = trunc_svd_with_backend(backend, site, rank - 1, params)
            .expect("trunc_svd failed during truncate");

        // Split U's fused row leg back into (*orig[..rank-1], chi).
        let reshape_u =
            |u_2d: DenseTensor<T>| -> DenseTensor<T> { u_2d.split_leg(0, &orig_shape[..rank - 1]) };

        match absorb {
            SvdAbsorb::Right => {
                // U stays at j (left-canonical), S·Vt absorbed into j+1.
                let svt = diagonal_scale_with_backend(backend, &vt, s.data_slice(), 0)
                    .expect("S·Vt scaling failed during truncate");
                (reshape_u(u), svt, err)
            }
            SvdAbsorb::Left => {
                // U·S stays at j, Vt absorbed into j+1.
                let us = diagonal_scale_with_backend(backend, &u, s.data_slice(), 1)
                    .expect("U·S scaling failed during truncate");
                (reshape_u(us), vt, err)
            }
            SvdAbsorb::Both => {
                // sqrt(S) applied to both sides.
                let sqrt_s: Vec<T::Real> = s.data_slice().iter().map(|v| v.sqrt()).collect();
                let u_scaled = diagonal_scale_with_backend(backend, &u, &sqrt_s, 1)
                    .expect("sqrt(S)*U scaling failed during truncate");
                let vt_scaled = diagonal_scale_with_backend(backend, &vt, &sqrt_s, 0)
                    .expect("sqrt(S)*Vt scaling failed during truncate");
                (reshape_u(u_scaled), vt_scaled, err)
            }
        }
    };

    *chain.site_mut(j) = left_storage;

    let new_next = {
        let next = chain.site(j + 1);
        absorb_from_left(&right_factor, next, backend)
    };

    *chain.site_mut(j + 1) = new_next;

    err * err
}

/// Left SVD step at site j: decompose, absorb into j-1 based on absorb mode.
fn left_trunc_step<T, B, C>(
    chain: &mut C,
    j: usize,
    params: &TruncSvdParams,
    absorb: SvdAbsorb,
    backend: &B,
) -> T::Real
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
    C: TensorChain<DenseStorage<T>, DenseLayout>,
{
    let (right_storage, left_factor, err) = {
        let site = chain.site(j);
        let orig_shape = site.shape().to_vec();

        let (u, s, vt, err) = trunc_svd_with_backend(backend, site, 1, params)
            .expect("trunc_svd failed during truncate");

        // Split Vt's fused column leg back into (chi, *orig[1..]).
        let reshape_vt =
            |vt_2d: DenseTensor<T>| -> DenseTensor<T> { vt_2d.split_leg(1, &orig_shape[1..]) };

        match absorb {
            SvdAbsorb::Right => {
                let us = diagonal_scale_with_backend(backend, &u, s.data_slice(), 1)
                    .expect("U·S scaling failed during truncate");
                (reshape_vt(vt), us, err)
            }
            SvdAbsorb::Left => {
                let svt = diagonal_scale_with_backend(backend, &vt, s.data_slice(), 0)
                    .expect("S·Vt scaling failed during truncate");
                (reshape_vt(svt), u, err)
            }
            SvdAbsorb::Both => {
                let sqrt_s: Vec<T::Real> = s.data_slice().iter().map(|v| v.sqrt()).collect();
                let vt_scaled = diagonal_scale_with_backend(backend, &vt, &sqrt_s, 0)
                    .expect("sqrt(S)*Vt scaling failed during truncate");
                let u_scaled = diagonal_scale_with_backend(backend, &u, &sqrt_s, 1)
                    .expect("sqrt(S)*U scaling failed during truncate");
                (reshape_vt(vt_scaled), u_scaled, err)
            }
        }
    };

    *chain.site_mut(j) = right_storage;

    let new_prev = {
        let prev = chain.site(j - 1);
        absorb_from_right(prev, &left_factor, backend)
    };

    *chain.site_mut(j - 1) = new_prev;

    err * err
}

// ============================================================================
// BlockSparse truncate
// ============================================================================

pub(super) fn truncate_bsp<T, S, B, C>(
    backend: &B,
    chain: &mut C,
    params: &TruncateParams,
) -> TruncResult<T>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>>,
{
    let n = chain.len();
    assert!(n > 0, "truncate requires a non-empty chain");

    let center = match chain.canonical_form() {
        CanonicalForm::Mixed { center } => *center,
        CanonicalForm::Left => n - 1,
        CanonicalForm::Right => 0,
        _ => {
            let c = params.center.unwrap_or(0);
            canonicalize_bsp(backend, chain, c);
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

    for j in center..n - 1 {
        total_err_sq = total_err_sq + right_trunc_step_bsp(chain, j, svd_params, absorb, backend);
    }

    for j in (1..n).rev() {
        total_err_sq = total_err_sq + left_trunc_step_bsp(chain, j, svd_params, absorb, backend);
    }

    for j in 0..center {
        total_err_sq = total_err_sq + right_trunc_step_bsp(chain, j, svd_params, absorb, backend);
    }

    let form = match absorb {
        SvdAbsorb::Both => CanonicalForm::Unknown,
        _ => CanonicalForm::Mixed { center },
    };
    chain.set_canonical_form(form);
    TruncResult {
        error: total_err_sq.sqrt(),
    }
}

fn right_trunc_step_bsp<T, S, B, C>(
    chain: &mut C,
    j: usize,
    params: &TruncSvdParams,
    absorb: SvdAbsorb,
    backend: &B,
) -> T::Real
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>>,
{
    let (left_storage, right_factor, err) = {
        let site = chain.site(j);
        let rank = site.rank();

        let (u, s, vt, err) = trunc_svd_block_sparse_with_backend(backend, site, rank - 1, params)
            .expect("trunc_svd_block_sparse failed during truncate");

        match absorb {
            SvdAbsorb::Right => {
                let svt = diagonal_scale_block_sparse_with_backend(backend, &vt, &s, 0)
                    .expect("S·Vt scaling failed during truncate");
                (u, svt, err)
            }
            SvdAbsorb::Left => {
                let u_rank = u.rank();
                let us = diagonal_scale_block_sparse_with_backend(backend, &u, &s, u_rank - 1)
                    .expect("U·S scaling failed during truncate");
                (us, vt, err)
            }
            SvdAbsorb::Both => {
                let sqrt_s = s.map(|v| (*v).sqrt());
                let u_rank = u.rank();
                let u_scaled =
                    diagonal_scale_block_sparse_with_backend(backend, &u, &sqrt_s, u_rank - 1)
                        .expect("√S·U scaling failed during truncate");
                let vt_scaled = diagonal_scale_block_sparse_with_backend(backend, &vt, &sqrt_s, 0)
                    .expect("√S·Vt scaling failed during truncate");
                (u_scaled, vt_scaled, err)
            }
        }
    };

    *chain.site_mut(j) = left_storage;

    let new_next = {
        let next = chain.site(j + 1);
        absorb_from_left_bsp(&right_factor, next, backend)
    };

    *chain.site_mut(j + 1) = new_next;

    err * err
}

fn left_trunc_step_bsp<T, S, B, C>(
    chain: &mut C,
    j: usize,
    params: &TruncSvdParams,
    absorb: SvdAbsorb,
    backend: &B,
) -> T::Real
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>>,
{
    let (right_storage, left_factor, err) = {
        let site = chain.site(j);

        let (u, s, vt, err) = trunc_svd_block_sparse_with_backend(backend, site, 1, params)
            .expect("trunc_svd_block_sparse failed during truncate");

        match absorb {
            SvdAbsorb::Right => {
                let u_rank = u.rank();
                let us = diagonal_scale_block_sparse_with_backend(backend, &u, &s, u_rank - 1)
                    .expect("U·S scaling failed during truncate");
                (vt, us, err)
            }
            SvdAbsorb::Left => {
                let svt = diagonal_scale_block_sparse_with_backend(backend, &vt, &s, 0)
                    .expect("S·Vt scaling failed during truncate");
                (svt, u, err)
            }
            SvdAbsorb::Both => {
                let sqrt_s = s.map(|v| (*v).sqrt());
                let vt_scaled = diagonal_scale_block_sparse_with_backend(backend, &vt, &sqrt_s, 0)
                    .expect("√S·Vt scaling failed during truncate");
                let u_rank = u.rank();
                let u_scaled =
                    diagonal_scale_block_sparse_with_backend(backend, &u, &sqrt_s, u_rank - 1)
                        .expect("√S·U scaling failed during truncate");
                (vt_scaled, u_scaled, err)
            }
        }
    };

    *chain.site_mut(j) = right_storage;

    let new_prev = {
        let prev = chain.site(j - 1);
        absorb_from_right_bsp(prev, &left_factor, backend)
    };

    *chain.site_mut(j - 1) = new_prev;

    err * err
}
