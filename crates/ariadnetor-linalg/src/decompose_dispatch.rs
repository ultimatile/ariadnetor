//! Layout-keyed dispatch for the four tensor decompositions
//! (SVD / truncated SVD / QR / LQ).
//!
//! [`LinalgDecompose`] is implemented on the concrete layout types
//! ([`DenseLayout`] and [`BlockSparseLayout<S>`]); each implementation pairs a
//! storage type via [`LinalgDecompose::Storage`] and routes to its
//! storage-specific kernel. Callers parameterized over `L: LinalgDecompose<T>`
//! issue one generic call that serves both flavors, mirroring the `MpsOps`
//! pattern in `arnet-mps`.
//!
//! # Auto-policy vs explicit policy
//!
//! Each operation has an auto-policy form (`svd`, …) and a policy-explicit
//! primitive (`svd_with_policy`, …). Both live on the trait because the
//! auto-policy *choice* is layout-specific and cannot be expressed in a
//! layout-generic free function: the dense path consults the backend's
//! `par_for_{svd,qr,lq}` hooks over the reshaped `(m, n)`, while the
//! block-sparse path has no single `(m, n)` and pins
//! [`ExecPolicy::Sequential`]. The public auto entry points are still the free
//! functions below; the per-layout policy choice is what the trait carries.
//!
//! # Operation authority
//!
//! Every method takes its compute backend explicitly at the call site, bound
//! by [`OpsFor<Self::Storage>`](arnet_tensor::OpsFor) — the same capability
//! gate the rest of the linalg surface enforces. The tensor carries no
//! backend, so there is a single, unambiguous authority per call. Block-sparse
//! methods enforce the layout-order invariant against the supplied backend
//! before dispatching; dense paths self-normalize.

use arnet_core::Scalar;
use arnet_core::backend::ExecPolicy;
use arnet_tensor::{
    BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, OpsFor, Sector, Storage,
    StorageFor, Tensor, TensorLayout,
};

use crate::block_sparse_decomp::{
    BlockSparseQrResult, BlockSparseSvdResult, BlockSparseTruncSvdResult,
    lq_block_sparse_with_policy_dense, qr_block_sparse_with_policy_dense,
    svd_block_sparse_with_policy_dense, trunc_svd_block_sparse_with_policy_dense,
};
use crate::decomposition::{
    LqResult, QrResult, SvdResult, TruncSvdParams, TruncSvdResult, lq_dense, lq_with_policy_dense,
    qr_dense, qr_with_policy_dense, svd_dense, svd_with_policy_dense, trunc_svd_dense,
    trunc_svd_with_policy_dense,
};
use crate::error::LinalgError;
use crate::tensor_bridge::check_bsp_data_layout_order_matches;

/// Layout-keyed dispatch trait for the four tensor decompositions.
///
/// Implemented for [`DenseLayout`] and [`BlockSparseLayout<S>`], each pairing a
/// storage type via [`Storage`](Self::Storage) and routing each operation to
/// its storage-specific kernel. The structural difference between the dense
/// `(U, S_flat, Vt)` SVD and the block-sparse `(U, BlockSingularValues, Vt)`
/// form is absorbed by the associated output types; QR and LQ are structurally
/// identical across layouts.
pub trait LinalgDecompose<T: Scalar>: TensorLayout + Sized {
    /// Storage type paired with this layout.
    type Storage: Storage + StorageFor<Self>;
    /// Output of [`svd`](Self::svd) / [`svd_with_policy`](Self::svd_with_policy).
    type SvdOutput;
    /// Output of [`trunc_svd`](Self::trunc_svd) /
    /// [`trunc_svd_with_policy`](Self::trunc_svd_with_policy).
    type TruncSvdOutput;
    /// Output of [`qr`](Self::qr) / [`qr_with_policy`](Self::qr_with_policy).
    type QrOutput;
    /// Output of [`lq`](Self::lq) / [`lq_with_policy`](Self::lq_with_policy).
    type LqOutput;

    /// Thin SVD with the layout's auto-selected execution policy.
    fn svd<B: OpsFor<Self::Storage>>(
        backend: &B,
        t: &Tensor<Self::Storage, Self>,
        nrow: usize,
    ) -> Result<Self::SvdOutput, LinalgError>;

    /// Truncated SVD with the layout's auto-selected execution policy.
    fn trunc_svd<B: OpsFor<Self::Storage>>(
        backend: &B,
        t: &Tensor<Self::Storage, Self>,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<Self::TruncSvdOutput, LinalgError>;

    /// Thin QR with the layout's auto-selected execution policy.
    fn qr<B: OpsFor<Self::Storage>>(
        backend: &B,
        t: &Tensor<Self::Storage, Self>,
        nrow: usize,
    ) -> Result<Self::QrOutput, LinalgError>;

    /// Thin LQ with the layout's auto-selected execution policy.
    fn lq<B: OpsFor<Self::Storage>>(
        backend: &B,
        t: &Tensor<Self::Storage, Self>,
        nrow: usize,
    ) -> Result<Self::LqOutput, LinalgError>;

    /// Thin SVD with a caller-specified execution policy.
    fn svd_with_policy<B: OpsFor<Self::Storage>>(
        backend: &B,
        t: &Tensor<Self::Storage, Self>,
        nrow: usize,
        policy: ExecPolicy,
    ) -> Result<Self::SvdOutput, LinalgError>;

    /// Truncated SVD with a caller-specified execution policy.
    fn trunc_svd_with_policy<B: OpsFor<Self::Storage>>(
        backend: &B,
        t: &Tensor<Self::Storage, Self>,
        nrow: usize,
        params: &TruncSvdParams,
        policy: ExecPolicy,
    ) -> Result<Self::TruncSvdOutput, LinalgError>;

    /// Thin QR with a caller-specified execution policy.
    fn qr_with_policy<B: OpsFor<Self::Storage>>(
        backend: &B,
        t: &Tensor<Self::Storage, Self>,
        nrow: usize,
        policy: ExecPolicy,
    ) -> Result<Self::QrOutput, LinalgError>;

    /// Thin LQ with a caller-specified execution policy.
    fn lq_with_policy<B: OpsFor<Self::Storage>>(
        backend: &B,
        t: &Tensor<Self::Storage, Self>,
        nrow: usize,
        policy: ExecPolicy,
    ) -> Result<Self::LqOutput, LinalgError>;
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T: Scalar> LinalgDecompose<T> for DenseLayout {
    type Storage = DenseStorage<T>;
    type SvdOutput = SvdResult<T>;
    type TruncSvdOutput = TruncSvdResult<T>;
    type QrOutput = QrResult<T>;
    type LqOutput = LqResult<T>;

    fn svd<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        t: &Tensor<DenseStorage<T>, DenseLayout>,
        nrow: usize,
    ) -> Result<SvdResult<T>, LinalgError> {
        let (u, s, vt) = svd_dense(backend, t.data(), nrow)?;
        Ok((
            Tensor::from_data(u),
            Tensor::from_data(s),
            Tensor::from_data(vt),
        ))
    }

    fn trunc_svd<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        t: &Tensor<DenseStorage<T>, DenseLayout>,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<TruncSvdResult<T>, LinalgError> {
        let (u, s, vt, err) = trunc_svd_dense(backend, t.data(), nrow, params)?;
        Ok((
            Tensor::from_data(u),
            Tensor::from_data(s),
            Tensor::from_data(vt),
            err,
        ))
    }

    fn qr<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        t: &Tensor<DenseStorage<T>, DenseLayout>,
        nrow: usize,
    ) -> Result<QrResult<T>, LinalgError> {
        let (q, r) = qr_dense(backend, t.data(), nrow)?;
        Ok((Tensor::from_data(q), Tensor::from_data(r)))
    }

    fn lq<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        t: &Tensor<DenseStorage<T>, DenseLayout>,
        nrow: usize,
    ) -> Result<LqResult<T>, LinalgError> {
        let (l, q) = lq_dense(backend, t.data(), nrow)?;
        Ok((Tensor::from_data(l), Tensor::from_data(q)))
    }

    fn svd_with_policy<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        t: &Tensor<DenseStorage<T>, DenseLayout>,
        nrow: usize,
        policy: ExecPolicy,
    ) -> Result<SvdResult<T>, LinalgError> {
        let (u, s, vt) = svd_with_policy_dense(backend, t.data(), nrow, policy)?;
        Ok((
            Tensor::from_data(u),
            Tensor::from_data(s),
            Tensor::from_data(vt),
        ))
    }

    fn trunc_svd_with_policy<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        t: &Tensor<DenseStorage<T>, DenseLayout>,
        nrow: usize,
        params: &TruncSvdParams,
        policy: ExecPolicy,
    ) -> Result<TruncSvdResult<T>, LinalgError> {
        let (u, s, vt, err) = trunc_svd_with_policy_dense(backend, t.data(), nrow, params, policy)?;
        Ok((
            Tensor::from_data(u),
            Tensor::from_data(s),
            Tensor::from_data(vt),
            err,
        ))
    }

    fn qr_with_policy<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        t: &Tensor<DenseStorage<T>, DenseLayout>,
        nrow: usize,
        policy: ExecPolicy,
    ) -> Result<QrResult<T>, LinalgError> {
        let (q, r) = qr_with_policy_dense(backend, t.data(), nrow, policy)?;
        Ok((Tensor::from_data(q), Tensor::from_data(r)))
    }

    fn lq_with_policy<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        t: &Tensor<DenseStorage<T>, DenseLayout>,
        nrow: usize,
        policy: ExecPolicy,
    ) -> Result<LqResult<T>, LinalgError> {
        let (l, q) = lq_with_policy_dense(backend, t.data(), nrow, policy)?;
        Ok((Tensor::from_data(l), Tensor::from_data(q)))
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T: Scalar, S: Sector> LinalgDecompose<T> for BlockSparseLayout<S> {
    type Storage = BlockSparseStorage<T>;
    type SvdOutput = BlockSparseSvdResult<T, S>;
    type TruncSvdOutput = BlockSparseTruncSvdResult<T, S>;
    type QrOutput = BlockSparseQrResult<T, S>;
    type LqOutput = BlockSparseQrResult<T, S>;

    fn svd<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        t: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        nrow: usize,
    ) -> Result<BlockSparseSvdResult<T, S>, LinalgError> {
        Self::svd_with_policy(backend, t, nrow, ExecPolicy::Sequential)
    }

    fn trunc_svd<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        t: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<BlockSparseTruncSvdResult<T, S>, LinalgError> {
        Self::trunc_svd_with_policy(backend, t, nrow, params, ExecPolicy::Sequential)
    }

    fn qr<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        t: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        nrow: usize,
    ) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
        Self::qr_with_policy(backend, t, nrow, ExecPolicy::Sequential)
    }

    fn lq<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        t: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        nrow: usize,
    ) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
        Self::lq_with_policy(backend, t, nrow, ExecPolicy::Sequential)
    }

    fn svd_with_policy<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        t: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        nrow: usize,
        policy: ExecPolicy,
    ) -> Result<BlockSparseSvdResult<T, S>, LinalgError> {
        check_bsp_data_layout_order_matches(t.data(), backend, "svd_block_sparse")?;
        let (u, s, vt) = svd_block_sparse_with_policy_dense(backend, t.data(), nrow, policy)?;
        Ok((Tensor::from_data(u), s, Tensor::from_data(vt)))
    }

    fn trunc_svd_with_policy<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        t: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        nrow: usize,
        params: &TruncSvdParams,
        policy: ExecPolicy,
    ) -> Result<BlockSparseTruncSvdResult<T, S>, LinalgError> {
        check_bsp_data_layout_order_matches(t.data(), backend, "trunc_svd_block_sparse")?;
        let (u, s, vt, err) =
            trunc_svd_block_sparse_with_policy_dense(backend, t.data(), nrow, params, policy)?;
        Ok((Tensor::from_data(u), s, Tensor::from_data(vt), err))
    }

    fn qr_with_policy<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        t: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        nrow: usize,
        policy: ExecPolicy,
    ) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
        check_bsp_data_layout_order_matches(t.data(), backend, "qr_block_sparse")?;
        let (q, r) = qr_block_sparse_with_policy_dense(backend, t.data(), nrow, policy)?;
        Ok((Tensor::from_data(q), Tensor::from_data(r)))
    }

    fn lq_with_policy<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        t: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        nrow: usize,
        policy: ExecPolicy,
    ) -> Result<BlockSparseQrResult<T, S>, LinalgError> {
        check_bsp_data_layout_order_matches(t.data(), backend, "lq_block_sparse")?;
        let (l, q) = lq_block_sparse_with_policy_dense(backend, t.data(), nrow, policy)?;
        Ok((Tensor::from_data(l), Tensor::from_data(q)))
    }
}

// ---------------------------------------------------------------------------
// Unified free functions — type-erase the layout into `L: LinalgDecompose<T>`
// so callers write `svd(backend, &t, nrow)` without naming the storage. `L`
// resolves from `Tensor`'s second argument and `T` through `L::Storage`, the
// same inference the `MpsOps` free fns rely on.
// ---------------------------------------------------------------------------

/// Thin SVD of a tensor reshaped as a matrix, using the supplied backend.
pub fn svd<T, L, B>(
    backend: &B,
    t: &Tensor<L::Storage, L>,
    nrow: usize,
) -> Result<L::SvdOutput, LinalgError>
where
    T: Scalar,
    L: LinalgDecompose<T>,
    B: OpsFor<L::Storage>,
{
    L::svd(backend, t, nrow)
}

/// Truncated SVD of a tensor reshaped as a matrix, using the supplied backend.
pub fn trunc_svd<T, L, B>(
    backend: &B,
    t: &Tensor<L::Storage, L>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<L::TruncSvdOutput, LinalgError>
where
    T: Scalar,
    L: LinalgDecompose<T>,
    B: OpsFor<L::Storage>,
{
    L::trunc_svd(backend, t, nrow, params)
}

/// Thin QR of a tensor reshaped as a matrix, using the supplied backend.
pub fn qr<T, L, B>(
    backend: &B,
    t: &Tensor<L::Storage, L>,
    nrow: usize,
) -> Result<L::QrOutput, LinalgError>
where
    T: Scalar,
    L: LinalgDecompose<T>,
    B: OpsFor<L::Storage>,
{
    L::qr(backend, t, nrow)
}

/// Thin LQ of a tensor reshaped as a matrix, using the supplied backend.
pub fn lq<T, L, B>(
    backend: &B,
    t: &Tensor<L::Storage, L>,
    nrow: usize,
) -> Result<L::LqOutput, LinalgError>
where
    T: Scalar,
    L: LinalgDecompose<T>,
    B: OpsFor<L::Storage>,
{
    L::lq(backend, t, nrow)
}
