//! Dispatch trait for MPS operations over different storage / layout
//! flavors.
//!
//! [`MpsOps`] is implemented on the concrete layout types
//! ([`DenseLayout`] and [`BlockSparseLayout<S>`]); each implementation
//! routes to its storage-specific kernel. Algorithms parameterized over
//! `L: MpsOps<T>` work uniformly across both flavors.
//!
//! # Operation authority
//!
//! Every operation takes its compute backend explicitly at the call
//! site and dispatches all kernels through that handle via the
//! explicit-backend (`*_with_backend`) linalg paths. The backend is
//! bound by [`OpsFor<Self::Storage>`](arnet_tensor::OpsFor) — the same
//! capability gate the linalg surface enforces — so only a backend that
//! has declared it operates on this layout's storage can be supplied.
//! The chain itself carries no backend, so there is a single,
//! unambiguous authority per call. Block-sparse layout-order safety is
//! enforced inside the linalg
//! twins (their release-active entry checks compare each operand's
//! layout order against the supplied backend's preferred order); dense
//! paths self-normalize.

use std::num::NonZeroUsize;

use arnet_core::Scalar;
use arnet_tensor::{
    BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, OpsFor, Sector, Storage,
    StorageFor, TensorLayout,
};

use super::chain::TensorChain;
use super::types::{ApplyMethod, Mpo, Mps, TruncResult, TruncateParams};

/// Dispatch trait for MPS / MPO operations.
///
/// Implemented for [`DenseLayout`] and [`BlockSparseLayout<S>`], each
/// pairing a storage type via [`MpsOps::Storage`] and routing each
/// operation to its storage-specific implementation. Every operation
/// receives the compute backend explicitly.
pub trait MpsOps<T: Scalar>: TensorLayout + Sized {
    /// Storage type paired with this layout.
    type Storage: Storage + StorageFor<Self>;

    /// Position the orthogonality center at `center`.
    fn canonicalize<B: OpsFor<Self::Storage>>(
        backend: &B,
        chain: &mut impl TensorChain<Self::Storage, Self>,
        center: usize,
    );

    /// Truncate bond dimensions according to `params`.
    fn truncate<B: OpsFor<Self::Storage>>(
        backend: &B,
        chain: &mut impl TensorChain<Self::Storage, Self>,
        params: &TruncateParams,
    ) -> TruncResult<T>;

    /// Compute the inner product ⟨ψ|φ⟩.
    fn inner<B: OpsFor<Self::Storage>>(
        backend: &B,
        psi: &Mps<Self::Storage, Self>,
        phi: &Mps<Self::Storage, Self>,
    ) -> T;

    /// Compute the norm ‖ψ‖.
    fn norm<B: OpsFor<Self::Storage>>(backend: &B, psi: &Mps<Self::Storage, Self>) -> T::Real;

    /// Compute the expectation value ⟨ψ|O|φ⟩.
    fn braket<B: OpsFor<Self::Storage>>(
        backend: &B,
        psi: &Mps<Self::Storage, Self>,
        op: &Mpo<Self::Storage, Self>,
        phi: &Mps<Self::Storage, Self>,
    ) -> T;

    /// Apply an MPO to an MPS: O|ψ⟩ via the streaming-naive algorithm.
    ///
    /// `forward_cap = None` keeps the forward pass on its QR-only branch
    /// (lossless streaming naive). `forward_cap = Some(k)` falls to a
    /// truncated SVD with cap `k * chi_max` when the natural per-site
    /// forward rank exceeds it.
    fn apply<B: OpsFor<Self::Storage>>(
        backend: &B,
        op: &Mpo<Self::Storage, Self>,
        psi: &Mps<Self::Storage, Self>,
        params: Option<&TruncateParams>,
        forward_cap: Option<NonZeroUsize>,
    ) -> Mps<Self::Storage, Self>;
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T: Scalar> MpsOps<T> for DenseLayout {
    type Storage = DenseStorage<T>;

    fn canonicalize<B: OpsFor<Self::Storage>>(
        backend: &B,
        chain: &mut impl TensorChain<DenseStorage<T>, DenseLayout>,
        center: usize,
    ) {
        super::canonicalize::canonicalize_dense(backend, chain, center);
    }

    fn truncate<B: OpsFor<Self::Storage>>(
        backend: &B,
        chain: &mut impl TensorChain<DenseStorage<T>, DenseLayout>,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        super::truncate::truncate_dense(backend, chain, params)
    }

    fn inner<B: OpsFor<Self::Storage>>(
        backend: &B,
        psi: &Mps<DenseStorage<T>, DenseLayout>,
        phi: &Mps<DenseStorage<T>, DenseLayout>,
    ) -> T {
        super::inner::inner_dense(backend, psi, phi)
    }

    fn norm<B: OpsFor<Self::Storage>>(
        backend: &B,
        psi: &Mps<DenseStorage<T>, DenseLayout>,
    ) -> T::Real {
        super::inner::norm_dense(backend, psi)
    }

    fn braket<B: OpsFor<Self::Storage>>(
        backend: &B,
        psi: &Mps<DenseStorage<T>, DenseLayout>,
        op: &Mpo<DenseStorage<T>, DenseLayout>,
        phi: &Mps<DenseStorage<T>, DenseLayout>,
    ) -> T {
        super::inner::braket_dense(backend, psi, op, phi)
    }

    fn apply<B: OpsFor<Self::Storage>>(
        backend: &B,
        op: &Mpo<DenseStorage<T>, DenseLayout>,
        psi: &Mps<DenseStorage<T>, DenseLayout>,
        params: Option<&TruncateParams>,
        forward_cap: Option<NonZeroUsize>,
    ) -> Mps<DenseStorage<T>, DenseLayout> {
        super::apply::apply_streaming_naive_dense(backend, op, psi, params, forward_cap)
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T: Scalar, S: Sector> MpsOps<T> for BlockSparseLayout<S> {
    type Storage = BlockSparseStorage<T>;

    fn canonicalize<B: OpsFor<Self::Storage>>(
        backend: &B,
        chain: &mut impl TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        center: usize,
    ) {
        super::canonicalize::canonicalize_bsp(backend, chain, center);
    }

    fn truncate<B: OpsFor<Self::Storage>>(
        backend: &B,
        chain: &mut impl TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        super::truncate::truncate_bsp(backend, chain, params)
    }

    fn inner<B: OpsFor<Self::Storage>>(
        backend: &B,
        psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        phi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    ) -> T {
        super::inner::inner_bsp(backend, psi, phi)
    }

    fn norm<B: OpsFor<Self::Storage>>(
        backend: &B,
        psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    ) -> T::Real {
        super::inner::norm_bsp(backend, psi)
    }

    fn braket<B: OpsFor<Self::Storage>>(
        backend: &B,
        psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        phi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    ) -> T {
        super::inner::braket_bsp(backend, psi, op, phi)
    }

    fn apply<B: OpsFor<Self::Storage>>(
        backend: &B,
        op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        params: Option<&TruncateParams>,
        forward_cap: Option<NonZeroUsize>,
    ) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>> {
        super::apply::apply_streaming_naive_bsp(backend, op, psi, params, forward_cap)
    }
}

// ---------------------------------------------------------------------------
// Unified free functions — type-erase the layout into `L: MpsOps<T>` so
// algorithms can write `canonicalize(backend, chain, c)` without naming
// the storage explicitly.
// ---------------------------------------------------------------------------

/// Position the orthogonality center at `center`.
pub fn canonicalize<T, L, B, C>(backend: &B, chain: &mut C, center: usize)
where
    T: Scalar,
    L: MpsOps<T>,
    B: OpsFor<L::Storage>,
    C: TensorChain<L::Storage, L>,
{
    <L as MpsOps<T>>::canonicalize(backend, chain, center);
}

/// Truncate bond dimensions according to `params`.
pub fn truncate<T, L, B, C>(backend: &B, chain: &mut C, params: &TruncateParams) -> TruncResult<T>
where
    T: Scalar,
    L: MpsOps<T>,
    B: OpsFor<L::Storage>,
    C: TensorChain<L::Storage, L>,
{
    <L as MpsOps<T>>::truncate(backend, chain, params)
}

/// Compute the inner product ⟨ψ|φ⟩.
pub fn inner<T, L, B>(backend: &B, psi: &Mps<L::Storage, L>, phi: &Mps<L::Storage, L>) -> T
where
    T: Scalar,
    L: MpsOps<T>,
    B: OpsFor<L::Storage>,
{
    <L as MpsOps<T>>::inner(backend, psi, phi)
}

/// Compute the norm ‖ψ‖.
pub fn norm<T, L, B>(backend: &B, psi: &Mps<L::Storage, L>) -> T::Real
where
    T: Scalar,
    L: MpsOps<T>,
    B: OpsFor<L::Storage>,
{
    <L as MpsOps<T>>::norm(backend, psi)
}

/// Compute the expectation value ⟨ψ|O|φ⟩.
pub fn braket<T, L, B>(
    backend: &B,
    psi: &Mps<L::Storage, L>,
    op: &Mpo<L::Storage, L>,
    phi: &Mps<L::Storage, L>,
) -> T
where
    T: Scalar,
    L: MpsOps<T>,
    B: OpsFor<L::Storage>,
{
    <L as MpsOps<T>>::braket(backend, psi, op, phi)
}

/// Apply an MPO to an MPS: O|ψ⟩ via the streaming-naive algorithm with the
/// default lossless forward sweep (`forward_cap = None`).
///
/// Equivalent to `apply_with_method(backend, op, psi, params, ApplyMethod::default())`.
pub fn apply<T, L, B>(
    backend: &B,
    op: &Mpo<L::Storage, L>,
    psi: &Mps<L::Storage, L>,
    params: Option<&TruncateParams>,
) -> Mps<L::Storage, L>
where
    T: Scalar,
    L: MpsOps<T>,
    B: OpsFor<L::Storage>,
{
    <L as MpsOps<T>>::apply(backend, op, psi, params, None)
}

/// Apply an MPO to an MPS using the requested algorithm.
///
/// `ApplyMethod::ZipUp` is reserved for the literature Stoudenmire-White
/// single-pass interleaved-truncation algorithm and is not yet
/// implemented; calling with that variant panics.
pub fn apply_with_method<T, L, B>(
    backend: &B,
    op: &Mpo<L::Storage, L>,
    psi: &Mps<L::Storage, L>,
    params: Option<&TruncateParams>,
    method: ApplyMethod,
) -> Mps<L::Storage, L>
where
    T: Scalar,
    L: MpsOps<T>,
    B: OpsFor<L::Storage>,
{
    match method {
        ApplyMethod::StreamingNaive { forward_cap } => {
            <L as MpsOps<T>>::apply(backend, op, psi, params, forward_cap)
        }
        ApplyMethod::ZipUp => unimplemented!(
            "ApplyMethod::ZipUp is reserved for the literature Stoudenmire-White \
             single-pass interleaved-truncation algorithm and is not yet \
             implemented; use ApplyMethod::StreamingNaive for the streaming \
             naive variant",
        ),
    }
}
