//! Truncate: reduce bond dimensions of a tensor chain via SVD sweeps.

use arnet::{
    BlockSparseContractResult, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor,
    ComputeBackend, DenseLayout, DenseStorage, DenseTensor, MemoryOrder, Scalar, Sector,
    TruncSvdParams, contract, contract_block_sparse, diagonal_scale, diagonal_scale_block_sparse,
    trunc_svd, trunc_svd_block_sparse,
};
use num_traits::{Float, Zero};

use super::canonicalize::{canonicalize_bsp, canonicalize_dense};
use super::chain::TensorChain;
use super::internal_helpers::dense_reshape;
use super::types::{CanonicalForm, SvdAbsorb, TruncResult, TruncateParams};

/// Truncate bond dimensions of a Dense tensor chain via SVD sweeps.
///
/// Performs SVD sweeps from the orthogonality center outward in both
/// directions, applying truncation at each bond. Returns the total
/// truncation error (Frobenius norm of discarded singular values).
///
/// If the chain is not in `Mixed` canonical form, auto-canonicalizes first.
pub(super) fn truncate_dense<T, B, C>(chain: &mut C, params: &TruncateParams) -> TruncResult<T>
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<DenseStorage<T>, DenseLayout, B>,
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

    for j in center..n - 1 {
        total_err_sq = total_err_sq + right_trunc_step(chain, j, svd_params, absorb);
    }

    for j in (1..n).rev() {
        total_err_sq = total_err_sq + left_trunc_step(chain, j, svd_params, absorb);
    }

    restore_center_sweep_dense(chain, center, svd_params, absorb, &mut total_err_sq);

    let form = match absorb {
        SvdAbsorb::Both => CanonicalForm::Unknown,
        _ => CanonicalForm::Mixed { center },
    };
    chain.set_canonical_form(form);
    TruncResult {
        error: total_err_sq.sqrt(),
    }
}

/// Final right sweep that restores the orthogonality center after the
/// preceding right and left sweeps. Defensively accumulates any residual
/// squared-error from each step in case of numerical drift.
fn restore_center_sweep_dense<T, B, C>(
    chain: &mut C,
    center: usize,
    svd_params: &TruncSvdParams,
    absorb: SvdAbsorb,
    total_err_sq: &mut T::Real,
) where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<DenseStorage<T>, DenseLayout, B>,
{
    for j in 0..center {
        *total_err_sq = *total_err_sq + right_trunc_step(chain, j, svd_params, absorb);
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
    C: TensorChain<DenseStorage<T>, DenseLayout, B>,
{
    let order = chain.backend().preferred_order();
    let rm = MemoryOrder::RowMajor;

    let (left_storage, right_factor, err) = {
        let site = chain.site(j);
        let rank = site.rank();
        let orig_shape = site.shape().to_vec();

        let (u, s, vt, err) =
            trunc_svd(site, rank - 1, params).expect("trunc_svd failed during truncate");

        let chi = u.shape()[1];
        let mut u_shape = orig_shape[..rank - 1].to_vec();
        u_shape.push(chi);

        let reshape_u = |u_2d: DenseTensor<T, B>| -> DenseTensor<T, B> {
            let u_rm = u_2d.reordered(rm);
            let multi = dense_reshape(&u_rm, u_shape.clone());
            multi.reordered(order)
        };

        match absorb {
            SvdAbsorb::Right => {
                // U stays at j (left-canonical), S·Vt absorbed into j+1.
                let svt = diagonal_scale(&vt, s.data_slice(), 0)
                    .expect("S·Vt scaling failed during truncate");
                (reshape_u(u), svt, err)
            }
            SvdAbsorb::Left => {
                // U·S stays at j, Vt absorbed into j+1.
                let us = diagonal_scale(&u, s.data_slice(), 1)
                    .expect("U·S scaling failed during truncate");
                (reshape_u(us), vt, err)
            }
            SvdAbsorb::Both => {
                // sqrt(S) applied to both sides.
                let sqrt_s: Vec<T::Real> = s.data_slice().iter().map(|v| v.sqrt()).collect();
                let u_scaled = diagonal_scale(&u, &sqrt_s, 1)
                    .expect("sqrt(S)*U scaling failed during truncate");
                let vt_scaled = diagonal_scale(&vt, &sqrt_s, 0)
                    .expect("sqrt(S)*Vt scaling failed during truncate");
                (reshape_u(u_scaled), vt_scaled, err)
            }
        }
    };

    *chain.site_mut(j) = left_storage;

    let new_next = {
        let next = chain.site(j + 1);
        absorb_from_left(&right_factor, next)
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
) -> T::Real
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<DenseStorage<T>, DenseLayout, B>,
{
    let order = chain.backend().preferred_order();
    let rm = MemoryOrder::RowMajor;

    let (right_storage, left_factor, err) = {
        let site = chain.site(j);
        let orig_shape = site.shape().to_vec();

        let (u, s, vt, err) = trunc_svd(site, 1, params).expect("trunc_svd failed during truncate");

        let chi = vt.shape()[0];
        let mut vt_shape = vec![chi];
        vt_shape.extend_from_slice(&orig_shape[1..]);

        let reshape_vt = |vt_2d: DenseTensor<T, B>| -> DenseTensor<T, B> {
            let vt_rm = vt_2d.reordered(rm);
            let multi = dense_reshape(&vt_rm, vt_shape.clone());
            multi.reordered(order)
        };

        match absorb {
            SvdAbsorb::Right => {
                let us = diagonal_scale(&u, s.data_slice(), 1)
                    .expect("U·S scaling failed during truncate");
                (reshape_vt(vt), us, err)
            }
            SvdAbsorb::Left => {
                let svt = diagonal_scale(&vt, s.data_slice(), 0)
                    .expect("S·Vt scaling failed during truncate");
                (reshape_vt(svt), u, err)
            }
            SvdAbsorb::Both => {
                let sqrt_s: Vec<T::Real> = s.data_slice().iter().map(|v| v.sqrt()).collect();
                let vt_scaled = diagonal_scale(&vt, &sqrt_s, 0)
                    .expect("sqrt(S)*Vt scaling failed during truncate");
                let u_scaled = diagonal_scale(&u, &sqrt_s, 1)
                    .expect("sqrt(S)*U scaling failed during truncate");
                (reshape_vt(vt_scaled), u_scaled, err)
            }
        }
    };

    *chain.site_mut(j) = right_storage;

    let new_prev = {
        let prev = chain.site(j - 1);
        absorb_from_right(prev, &left_factor)
    };

    *chain.site_mut(j - 1) = new_prev;

    err * err
}

fn absorb_from_left<T, B>(left: &DenseTensor<T, B>, next: &DenseTensor<T, B>) -> DenseTensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let order = next.backend().preferred_order();
    let rm = MemoryOrder::RowMajor;
    let next_rm = next.reordered(rm);
    let next_shape = next_rm.shape().to_vec();
    let first = next_shape[0];
    let rest: usize = next_shape[1..].iter().product();

    let next_2d_rm = dense_reshape(&next_rm, vec![first, rest]);
    let next_2d = next_2d_rm.reordered(order);
    let result_2d =
        contract(left, &next_2d, "ab,bc->ac").expect("left absorption failed during truncate");

    let result_2d_rm = result_2d.reordered(rm);
    let k = left.shape()[0];
    let mut new_shape = next_shape;
    new_shape[0] = k;
    let result_multi = dense_reshape(&result_2d_rm, new_shape);
    result_multi.reordered(order)
}

fn absorb_from_right<T, B>(prev: &DenseTensor<T, B>, right: &DenseTensor<T, B>) -> DenseTensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let order = prev.backend().preferred_order();
    let rm = MemoryOrder::RowMajor;
    let prev_rm = prev.reordered(rm);
    let prev_shape = prev_rm.shape().to_vec();
    let last = *prev_shape.last().unwrap();
    let rest: usize = prev_shape[..prev_shape.len() - 1].iter().product();

    let prev_2d_rm = dense_reshape(&prev_rm, vec![rest, last]);
    let prev_2d = prev_2d_rm.reordered(order);
    let result_2d =
        contract(&prev_2d, right, "ab,bc->ac").expect("right absorption failed during truncate");

    let result_2d_rm = result_2d.reordered(rm);
    let k = right.shape()[1];
    let mut new_shape = prev_shape;
    *new_shape.last_mut().unwrap() = k;
    let result_multi = dense_reshape(&result_2d_rm, new_shape);
    result_multi.reordered(order)
}

// ============================================================================
// BlockSparse truncate
// ============================================================================

pub(super) fn truncate_bsp<T, S, B, C>(chain: &mut C, params: &TruncateParams) -> TruncResult<T>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
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

    for j in center..n - 1 {
        total_err_sq = total_err_sq + right_trunc_step_bsp(chain, j, svd_params, absorb);
    }

    for j in (1..n).rev() {
        total_err_sq = total_err_sq + left_trunc_step_bsp(chain, j, svd_params, absorb);
    }

    for j in 0..center {
        total_err_sq = total_err_sq + right_trunc_step_bsp(chain, j, svd_params, absorb);
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
) -> T::Real
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
{
    let (left_storage, right_factor, err) = {
        let site = chain.site(j);
        let rank = site.rank();

        let (u, s, vt, err) = trunc_svd_block_sparse(site, rank - 1, params)
            .expect("trunc_svd_block_sparse failed during truncate");

        match absorb {
            SvdAbsorb::Right => {
                let svt = diagonal_scale_block_sparse(&vt, &s, 0)
                    .expect("S·Vt scaling failed during truncate");
                (u, svt, err)
            }
            SvdAbsorb::Left => {
                let u_rank = u.rank();
                let us = diagonal_scale_block_sparse(&u, &s, u_rank - 1)
                    .expect("U·S scaling failed during truncate");
                (us, vt, err)
            }
            SvdAbsorb::Both => {
                let sqrt_s = s.map(|v| (*v).sqrt());
                let u_rank = u.rank();
                let u_scaled = diagonal_scale_block_sparse(&u, &sqrt_s, u_rank - 1)
                    .expect("√S·U scaling failed during truncate");
                let vt_scaled = diagonal_scale_block_sparse(&vt, &sqrt_s, 0)
                    .expect("√S·Vt scaling failed during truncate");
                (u_scaled, vt_scaled, err)
            }
        }
    };

    *chain.site_mut(j) = left_storage;

    let new_next = {
        let next = chain.site(j + 1);
        absorb_from_left_bsp(&right_factor, next)
    };

    *chain.site_mut(j + 1) = new_next;

    err * err
}

fn left_trunc_step_bsp<T, S, B, C>(
    chain: &mut C,
    j: usize,
    params: &TruncSvdParams,
    absorb: SvdAbsorb,
) -> T::Real
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
{
    let (right_storage, left_factor, err) = {
        let site = chain.site(j);

        let (u, s, vt, err) = trunc_svd_block_sparse(site, 1, params)
            .expect("trunc_svd_block_sparse failed during truncate");

        match absorb {
            SvdAbsorb::Right => {
                let u_rank = u.rank();
                let us = diagonal_scale_block_sparse(&u, &s, u_rank - 1)
                    .expect("U·S scaling failed during truncate");
                (vt, us, err)
            }
            SvdAbsorb::Left => {
                let svt = diagonal_scale_block_sparse(&vt, &s, 0)
                    .expect("S·Vt scaling failed during truncate");
                (svt, u, err)
            }
            SvdAbsorb::Both => {
                let sqrt_s = s.map(|v| (*v).sqrt());
                let vt_scaled = diagonal_scale_block_sparse(&vt, &sqrt_s, 0)
                    .expect("√S·Vt scaling failed during truncate");
                let u_rank = u.rank();
                let u_scaled = diagonal_scale_block_sparse(&u, &sqrt_s, u_rank - 1)
                    .expect("√S·U scaling failed during truncate");
                (vt_scaled, u_scaled, err)
            }
        }
    };

    *chain.site_mut(j) = right_storage;

    let new_prev = {
        let prev = chain.site(j - 1);
        absorb_from_right_bsp(prev, &left_factor)
    };

    *chain.site_mut(j - 1) = new_prev;

    err * err
}

fn absorb_from_left_bsp<T, S, B>(
    left: &BlockSparseTensor<T, S, B>,
    next: &BlockSparseTensor<T, S, B>,
) -> BlockSparseTensor<T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    match contract_block_sparse(left, next, &[1], &[0])
        .expect("left absorption failed during truncate")
    {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => {
            unreachable!("left absorption contraction always produces a tensor")
        }
    }
}

fn absorb_from_right_bsp<T, S, B>(
    prev: &BlockSparseTensor<T, S, B>,
    right: &BlockSparseTensor<T, S, B>,
) -> BlockSparseTensor<T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let last = prev.rank() - 1;
    match contract_block_sparse(prev, right, &[last], &[0])
        .expect("right absorption failed during truncate")
    {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => {
            unreachable!("right absorption contraction always produces a tensor")
        }
    }
}
