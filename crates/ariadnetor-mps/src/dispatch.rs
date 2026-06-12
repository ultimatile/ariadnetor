//! Dispatch trait for MPS operations over different storage / layout
//! flavors.
//!
//! [`MpsOps`] is implemented on the concrete layout types
//! ([`DenseLayout`] and [`BlockSparseLayout<S>`]); each implementation
//! routes to its storage-specific kernel and additionally performs a
//! Tier 2 defensive scan of all participating site orders against the
//! chain's backend's `preferred_order()`. Algorithms parameterized over
//! `L: MpsOps<T>` work uniformly across both flavors.
//!
//! # Operation authority
//!
//! Every operation below derives its compute backend once from the
//! entry chain (`psi` for the cross-chain operations) and dispatches
//! all kernels through that handle via the explicit-backend
//! (`*_with_backend`) paths. The `Arc` a site tensor carries is a
//! result label only on these paths and never receives kernel
//! dispatch. On chains whose sites hold a different backend instance
//! than the chain handle (constructible — site validation compares
//! memory order, not `Arc` identity), this is the operative contract:
//! the chain handle is the single authority, per the call-site
//! backend-supply design.

use std::num::NonZeroUsize;

use arnet::{
    BlockSparseLayout, BlockSparseStorage, ComputeBackend, DenseLayout, DenseStorage, Scalar,
    Sector, Storage, StorageFor, TensorLayout,
};

use super::chain::TensorChain;
use super::types::{ApplyMethod, Mpo, Mps, TruncResult, TruncateParams};

/// Dispatch trait for MPS / MPO operations.
///
/// Implemented for [`DenseLayout`] and [`BlockSparseLayout<S>`], each
/// pairing a storage type via [`MpsOps::Storage`] and routing each
/// operation to its storage-specific implementation.
pub trait MpsOps<T: Scalar>: TensorLayout + Sized {
    /// Storage type paired with this layout.
    type Storage: Storage + StorageFor<Self>;

    /// Position the orthogonality center at `center`.
    fn canonicalize<B: ComputeBackend>(
        chain: &mut impl TensorChain<Self::Storage, Self, B>,
        center: usize,
    );

    /// Truncate bond dimensions according to `params`.
    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChain<Self::Storage, Self, B>,
        params: &TruncateParams,
    ) -> TruncResult<T>;

    /// Compute the inner product ⟨ψ|φ⟩.
    fn inner<B: ComputeBackend>(
        psi: &Mps<Self::Storage, Self, B>,
        phi: &Mps<Self::Storage, Self, B>,
    ) -> T;

    /// Compute the norm ‖ψ‖.
    fn norm<B: ComputeBackend>(psi: &Mps<Self::Storage, Self, B>) -> T::Real;

    /// Compute the expectation value ⟨ψ|O|φ⟩.
    fn braket<B: ComputeBackend>(
        psi: &Mps<Self::Storage, Self, B>,
        op: &Mpo<Self::Storage, Self, B>,
        phi: &Mps<Self::Storage, Self, B>,
    ) -> T;

    /// Apply an MPO to an MPS: O|ψ⟩ via the streaming-naive algorithm.
    ///
    /// `forward_cap = None` keeps the forward pass on its QR-only branch
    /// (lossless streaming naive). `forward_cap = Some(k)` falls to a
    /// truncated SVD with cap `k * chi_max` when the natural per-site
    /// forward rank exceeds it.
    fn apply<B: ComputeBackend>(
        op: &Mpo<Self::Storage, Self, B>,
        psi: &Mps<Self::Storage, Self, B>,
        params: Option<&TruncateParams>,
        forward_cap: Option<NonZeroUsize>,
    ) -> Mps<Self::Storage, Self, B>;
}

// ============================================================================
// Tier 2 helper macros: assert every participating site's layout order
// matches the chain's backend's preferred order. Site-level mutation
// through `TensorChain::site_mut` can in principle violate the Tier 1
// invariant established at construction; the Tier 2 scan catches that
// at every cross-chain op entry point.
// ============================================================================

fn assert_dense_chain_order<T, B, C>(chain: &C, ctx: &str)
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<DenseStorage<T>, DenseLayout, B>,
{
    let expected = chain.backend().preferred_order();
    for i in 0..chain.len() {
        let got = chain.site(i).data().layout().order();
        assert_eq!(
            got, expected,
            "{ctx}: site {i} order ({got:?}) != backend.preferred_order() ({expected:?})",
        );
        // The internal operation paths dispatch through the chain
        // handle (see the module doc), but the legacy public wrappers
        // still dispatch through a tensor's cached backend until they
        // are removed, and `site_mut` can rebind a foreign backend.
        // Per-site cached backend preferred_order must therefore still
        // agree with the chain backend's.
        let site_backend_order = chain.site(i).backend().preferred_order();
        assert_eq!(
            site_backend_order, expected,
            "{ctx}: site {i} cached backend preferred_order ({site_backend_order:?}) != chain backend preferred_order ({expected:?})",
        );
    }
}

fn assert_bsp_chain_order<T, S, B, C>(chain: &C, ctx: &str)
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
{
    let expected = chain.backend().preferred_order();
    for i in 0..chain.len() {
        let got = chain.site(i).data().layout().order();
        assert_eq!(
            got, expected,
            "{ctx}: site {i} order ({got:?}) != backend.preferred_order() ({expected:?})",
        );
        let site_backend_order = chain.site(i).backend().preferred_order();
        assert_eq!(
            site_backend_order, expected,
            "{ctx}: site {i} cached backend preferred_order ({site_backend_order:?}) != chain backend preferred_order ({expected:?})",
        );
    }
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T: Scalar> MpsOps<T> for DenseLayout {
    type Storage = DenseStorage<T>;

    fn canonicalize<B: ComputeBackend>(
        chain: &mut impl TensorChain<DenseStorage<T>, DenseLayout, B>,
        center: usize,
    ) {
        assert_dense_chain_order(chain, "MpsOps::canonicalize");
        super::canonicalize::canonicalize_dense(chain, center);
    }

    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChain<DenseStorage<T>, DenseLayout, B>,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        assert_dense_chain_order(chain, "MpsOps::truncate");
        super::truncate::truncate_dense(chain, params)
    }

    fn inner<B: ComputeBackend>(
        psi: &Mps<DenseStorage<T>, DenseLayout, B>,
        phi: &Mps<DenseStorage<T>, DenseLayout, B>,
    ) -> T {
        assert_eq!(
            psi.backend().preferred_order(),
            phi.backend().preferred_order(),
            "MpsOps::inner: psi/phi backend preferred_order mismatch",
        );
        assert_dense_chain_order(psi, "MpsOps::inner.psi");
        assert_dense_chain_order(phi, "MpsOps::inner.phi");
        super::inner::inner_dense(psi, phi)
    }

    fn norm<B: ComputeBackend>(psi: &Mps<DenseStorage<T>, DenseLayout, B>) -> T::Real {
        assert_dense_chain_order(psi, "MpsOps::norm");
        super::inner::norm_dense(psi)
    }

    fn braket<B: ComputeBackend>(
        psi: &Mps<DenseStorage<T>, DenseLayout, B>,
        op: &Mpo<DenseStorage<T>, DenseLayout, B>,
        phi: &Mps<DenseStorage<T>, DenseLayout, B>,
    ) -> T {
        let order = psi.backend().preferred_order();
        assert_eq!(
            order,
            op.backend().preferred_order(),
            "MpsOps::braket: psi/op backend preferred_order mismatch",
        );
        assert_eq!(
            order,
            phi.backend().preferred_order(),
            "MpsOps::braket: psi/phi backend preferred_order mismatch",
        );
        assert_dense_chain_order(psi, "MpsOps::braket.psi");
        assert_dense_chain_order(op, "MpsOps::braket.op");
        assert_dense_chain_order(phi, "MpsOps::braket.phi");
        super::inner::braket_dense(psi, op, phi)
    }

    fn apply<B: ComputeBackend>(
        op: &Mpo<DenseStorage<T>, DenseLayout, B>,
        psi: &Mps<DenseStorage<T>, DenseLayout, B>,
        params: Option<&TruncateParams>,
        forward_cap: Option<NonZeroUsize>,
    ) -> Mps<DenseStorage<T>, DenseLayout, B> {
        assert_eq!(
            op.backend().preferred_order(),
            psi.backend().preferred_order(),
            "MpsOps::apply: op/psi backend preferred_order mismatch",
        );
        assert_dense_chain_order(op, "MpsOps::apply.op");
        assert_dense_chain_order(psi, "MpsOps::apply.psi");
        super::apply::apply_streaming_naive_dense(op, psi, params, forward_cap)
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T: Scalar, S: Sector> MpsOps<T> for BlockSparseLayout<S> {
    type Storage = BlockSparseStorage<T>;

    fn canonicalize<B: ComputeBackend>(
        chain: &mut impl TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
        center: usize,
    ) {
        assert_bsp_chain_order(chain, "MpsOps::canonicalize");
        super::canonicalize::canonicalize_bsp(chain, center);
    }

    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        assert_bsp_chain_order(chain, "MpsOps::truncate");
        super::truncate::truncate_bsp(chain, params)
    }

    fn inner<B: ComputeBackend>(
        psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
        phi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    ) -> T {
        assert_eq!(
            psi.backend().preferred_order(),
            phi.backend().preferred_order(),
            "MpsOps::inner: psi/phi backend preferred_order mismatch",
        );
        assert_bsp_chain_order(psi, "MpsOps::inner.psi");
        assert_bsp_chain_order(phi, "MpsOps::inner.phi");
        super::inner::inner_bsp(psi, phi)
    }

    fn norm<B: ComputeBackend>(
        psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    ) -> T::Real {
        assert_bsp_chain_order(psi, "MpsOps::norm");
        super::inner::norm_bsp(psi)
    }

    fn braket<B: ComputeBackend>(
        psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
        op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
        phi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    ) -> T {
        let order = psi.backend().preferred_order();
        assert_eq!(
            order,
            op.backend().preferred_order(),
            "MpsOps::braket: psi/op backend preferred_order mismatch",
        );
        assert_eq!(
            order,
            phi.backend().preferred_order(),
            "MpsOps::braket: psi/phi backend preferred_order mismatch",
        );
        assert_bsp_chain_order(psi, "MpsOps::braket.psi");
        assert_bsp_chain_order(op, "MpsOps::braket.op");
        assert_bsp_chain_order(phi, "MpsOps::braket.phi");
        super::inner::braket_bsp(psi, op, phi)
    }

    fn apply<B: ComputeBackend>(
        op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
        psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
        params: Option<&TruncateParams>,
        forward_cap: Option<NonZeroUsize>,
    ) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B> {
        assert_eq!(
            op.backend().preferred_order(),
            psi.backend().preferred_order(),
            "MpsOps::apply: op/psi backend preferred_order mismatch",
        );
        assert_bsp_chain_order(op, "MpsOps::apply.op");
        assert_bsp_chain_order(psi, "MpsOps::apply.psi");
        super::apply::apply_streaming_naive_bsp(op, psi, params, forward_cap)
    }
}

// ---------------------------------------------------------------------------
// Unified free functions — type-erase the layout into `L: MpsOps<T>` so
// algorithms can write `canonicalize(chain, c)` without naming the
// storage explicitly.
// ---------------------------------------------------------------------------

/// Position the orthogonality center at `center`.
pub fn canonicalize<T, L, B, C>(chain: &mut C, center: usize)
where
    T: Scalar,
    L: MpsOps<T>,
    B: ComputeBackend,
    C: TensorChain<L::Storage, L, B>,
{
    <L as MpsOps<T>>::canonicalize(chain, center);
}

/// Truncate bond dimensions according to `params`.
pub fn truncate<T, L, B, C>(chain: &mut C, params: &TruncateParams) -> TruncResult<T>
where
    T: Scalar,
    L: MpsOps<T>,
    B: ComputeBackend,
    C: TensorChain<L::Storage, L, B>,
{
    <L as MpsOps<T>>::truncate(chain, params)
}

/// Compute the inner product ⟨ψ|φ⟩.
pub fn inner<T, L, B>(psi: &Mps<L::Storage, L, B>, phi: &Mps<L::Storage, L, B>) -> T
where
    T: Scalar,
    L: MpsOps<T>,
    B: ComputeBackend,
{
    <L as MpsOps<T>>::inner(psi, phi)
}

/// Compute the norm ‖ψ‖.
pub fn norm<T, L, B>(psi: &Mps<L::Storage, L, B>) -> T::Real
where
    T: Scalar,
    L: MpsOps<T>,
    B: ComputeBackend,
{
    <L as MpsOps<T>>::norm(psi)
}

/// Compute the expectation value ⟨ψ|O|φ⟩.
pub fn braket<T, L, B>(
    psi: &Mps<L::Storage, L, B>,
    op: &Mpo<L::Storage, L, B>,
    phi: &Mps<L::Storage, L, B>,
) -> T
where
    T: Scalar,
    L: MpsOps<T>,
    B: ComputeBackend,
{
    <L as MpsOps<T>>::braket(psi, op, phi)
}

/// Apply an MPO to an MPS: O|ψ⟩ via the streaming-naive algorithm with the
/// default lossless forward sweep (`forward_cap = None`).
///
/// Equivalent to `apply_with_method(op, psi, params, ApplyMethod::default())`.
pub fn apply<T, L, B>(
    op: &Mpo<L::Storage, L, B>,
    psi: &Mps<L::Storage, L, B>,
    params: Option<&TruncateParams>,
) -> Mps<L::Storage, L, B>
where
    T: Scalar,
    L: MpsOps<T>,
    B: ComputeBackend,
{
    <L as MpsOps<T>>::apply(op, psi, params, None)
}

/// Apply an MPO to an MPS using the requested algorithm.
///
/// `ApplyMethod::ZipUp` is reserved for the literature Stoudenmire-White
/// single-pass interleaved-truncation algorithm and is not yet
/// implemented; calling with that variant panics.
pub fn apply_with_method<T, L, B>(
    op: &Mpo<L::Storage, L, B>,
    psi: &Mps<L::Storage, L, B>,
    params: Option<&TruncateParams>,
    method: ApplyMethod,
) -> Mps<L::Storage, L, B>
where
    T: Scalar,
    L: MpsOps<T>,
    B: ComputeBackend,
{
    match method {
        ApplyMethod::StreamingNaive { forward_cap } => {
            <L as MpsOps<T>>::apply(op, psi, params, forward_cap)
        }
        ApplyMethod::ZipUp => unimplemented!(
            "ApplyMethod::ZipUp is reserved for the literature Stoudenmire-White \
             single-pass interleaved-truncation algorithm and is not yet \
             implemented; use ApplyMethod::StreamingNaive for the streaming \
             naive variant",
        ),
    }
}
