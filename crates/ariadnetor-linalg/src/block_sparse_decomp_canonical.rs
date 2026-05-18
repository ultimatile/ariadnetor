//! Canonical [`BlockSparseTensorData`]-typed entry points for the
//! block-sparse decompositions.
//!
//! Each wrapper normalizes its input to `backend.preferred_order()`
//! via [`normalize_to_order`], delegates to the matching
//! `*_block_sparse_with_policy_repr` kernel in
//! [`super`](super::block_sparse_decomp), and tags both outputs at
//! `backend.preferred_order()`. The pair (canonical fn + `_repr`
//! sister) collapses in Unit 5 when `BlockSparse<T, S>` is removed.

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, MemoryOrder};
use arnet_tensor::{BlockSparse, BlockSparseTensorData, Sector};

use crate::block_sparse_contract::repack_block_data;
use crate::block_sparse_decomp::{
    BlockSparseQrResult, BlockSparseSvdResult, BlockSparseTruncSvdResult,
    lq_block_sparse_with_policy_repr, qr_block_sparse_with_policy_repr,
    svd_block_sparse_with_policy_repr, trunc_svd_block_sparse_with_policy_repr,
};
use crate::decomposition::TruncSvdParams;
use crate::error::LinalgError;

// ---------------------------------------------------------------------------
// SVD
// ---------------------------------------------------------------------------

/// Thin SVD of a block-sparse tensor via fused sector method.
///
/// Returns `(U, S, Vt)` where:
/// - `U`: legs = `[left_legs..., bond(In)]`, `flux = identity()`
/// - `S`: singular values per fused sector (descending within each sector)
/// - `Vt`: legs = `[bond(Out), right_legs...]`, `flux = original_flux`
///
/// Input data is normalized to `backend.preferred_order()` before the
/// per-sector dense SVD; both output tensors are tagged at
/// `backend.preferred_order()`.
pub fn svd_block_sparse<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
) -> Result<BlockSparseSvdResult<T, S>, LinalgError> {
    svd_block_sparse_with_policy(backend, tensor, nrow, ExecPolicy::Sequential)
}

/// Block-sparse SVD with caller-specified execution policy for per-sector
/// dense decomposition.
///
/// Expert-layer counterpart of [`svd_block_sparse`]; see its docs for
/// memory-order behavior.
pub fn svd_block_sparse_with_policy<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<BlockSparseSvdResult<T, S>, LinalgError> {
    let preferred = backend.preferred_order();
    let bs = normalize_to_order(tensor, preferred);
    let (u, s, vt) = svd_block_sparse_with_policy_repr(backend, &bs, nrow, policy)?;
    Ok((
        u.into_tensor_data(preferred),
        s,
        vt.into_tensor_data(preferred),
    ))
}

// ---------------------------------------------------------------------------
// Truncated SVD
// ---------------------------------------------------------------------------

/// Truncated SVD of a block-sparse tensor via fused sector method.
///
/// Performs full per-sector SVD, then applies cross-sector truncation using
/// `chi_max` and/or `target_trunc_err` from `params`. When both are set,
/// the stricter (smaller) bound applies.
///
/// Returns `(U, S, Vt, trunc_err)` where `trunc_err` is the Frobenius norm
/// of discarded singular values.
pub fn trunc_svd_block_sparse<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<BlockSparseTruncSvdResult<T, S>, LinalgError> {
    trunc_svd_block_sparse_with_policy(backend, tensor, nrow, params, ExecPolicy::Sequential)
}

/// Truncated block-sparse SVD with caller-specified execution policy for
/// per-sector dense decomposition.
///
/// Expert-layer counterpart of [`trunc_svd_block_sparse`]; the default wrapper
/// hardcodes `ExecPolicy::Sequential`.
pub fn trunc_svd_block_sparse_with_policy<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    params: &TruncSvdParams,
    policy: ExecPolicy,
) -> Result<BlockSparseTruncSvdResult<T, S>, LinalgError> {
    let preferred = backend.preferred_order();
    let bs = normalize_to_order(tensor, preferred);
    let (u, s, vt, err) =
        trunc_svd_block_sparse_with_policy_repr(backend, &bs, nrow, params, policy)?;
    Ok((
        u.into_tensor_data(preferred),
        s,
        vt.into_tensor_data(preferred),
        err,
    ))
}

// ---------------------------------------------------------------------------
// QR
// ---------------------------------------------------------------------------

/// QR decomposition of a block-sparse tensor via fused sector method.
///
/// Returns `(Q, R)` where:
/// - `Q`: legs = `[left_legs..., bond(In)]`, `flux = identity()`
/// - `R`: legs = `[bond(Out), right_legs...]`, `flux = original_flux`
pub fn qr_block_sparse<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
    qr_block_sparse_with_policy(backend, tensor, nrow, ExecPolicy::Sequential)
}

/// Block-sparse QR with caller-specified execution policy for per-sector
/// dense decomposition.
///
/// Expert-layer counterpart of [`qr_block_sparse`]; the default wrapper
/// hardcodes `ExecPolicy::Sequential`.
pub fn qr_block_sparse_with_policy<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
    let preferred = backend.preferred_order();
    let bs = normalize_to_order(tensor, preferred);
    let (q, r) = qr_block_sparse_with_policy_repr(backend, &bs, nrow, policy)?;
    Ok((q.into_tensor_data(preferred), r.into_tensor_data(preferred)))
}

// ---------------------------------------------------------------------------
// LQ
// ---------------------------------------------------------------------------

/// LQ decomposition of a block-sparse tensor via fused sector method.
///
/// Returns `(L, Q)` where:
/// - `L`: legs = `[left_legs..., bond(In)]`, `flux = identity()`
/// - `Q`: legs = `[bond(Out), right_legs...]`, `flux = original_flux`
pub fn lq_block_sparse<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
    lq_block_sparse_with_policy(backend, tensor, nrow, ExecPolicy::Sequential)
}

/// Block-sparse LQ with caller-specified execution policy for per-sector
/// dense decomposition.
///
/// Expert-layer counterpart of [`lq_block_sparse`]; the default wrapper
/// hardcodes `ExecPolicy::Sequential`.
pub fn lq_block_sparse_with_policy<T: Scalar, S: Sector>(
    backend: &impl ComputeBackend,
    tensor: &BlockSparseTensorData<T, S>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
    let preferred = backend.preferred_order();
    let bs = normalize_to_order(tensor, preferred);
    let (l, q) = lq_block_sparse_with_policy_repr(backend, &bs, nrow, policy)?;
    Ok((l.into_tensor_data(preferred), q.into_tensor_data(preferred)))
}

// ---------------------------------------------------------------------------
// Normalize helper
// ---------------------------------------------------------------------------

/// Normalize a `BlockSparseTensorData` to a target memory order,
/// returning a legacy `BlockSparse` view suitable for the per-sector
/// kernel. No physical work when the target equals the current order
/// (Arc-move only); per-block data is repacked otherwise.
fn normalize_to_order<T: Scalar, S: Sector>(
    tensor: &BlockSparseTensorData<T, S>,
    target: MemoryOrder,
) -> BlockSparse<T, S> {
    let current = tensor.layout().order();
    if current == target {
        return BlockSparse::from_tensor_data(tensor.clone());
    }
    let layout = tensor.layout();
    let indices: Vec<_> = layout.indices().to_vec();
    let flux = layout.flux().clone();
    let mut out = BlockSparse::zeros(indices, flux);
    for meta in layout.block_metas() {
        let src = tensor
            .block_data(&meta.coord)
            .expect("layout-enumerated block must have storage");
        let block_shape: Vec<usize> = (0..layout.rank())
            .map(|a| layout.indices()[a].block_dim(meta.coord.0[a]))
            .collect();
        let repacked = repack_block_data(src, &block_shape, current, target);
        let dst = out
            .block_data_mut(&meta.coord)
            .expect("zero-initialized output must have matching block");
        dst.copy_from_slice(&repacked);
    }
    out
}
