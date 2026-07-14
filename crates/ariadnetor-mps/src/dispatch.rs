//! Chain-keyed dispatch for MPS operations over different storage /
//! layout flavors.
//!
//! [`MpsOps`] is keyed on the chain type itself
//! ([`Mps<DenseStorage<T>, DenseLayout>`](Mps) and
//! [`Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>`](Mps)); each
//! implementation routes to its storage-specific kernel. The trait is
//! **sealed** (its `sealed::Sealed` supertrait is crate-private), so it
//! cannot be implemented downstream and the storage / layout taxonomy is
//! reachable only as its sealed associated types rather than as free
//! `Storage` / `Layout` bounds on a public surface.
//!
//! The public entry points are the multi-arg free functions below
//! ([`inner`], [`braket`], [`apply`], [`apply_with_method`]) plus the
//! single-chain inherent methods on [`Mps`] (`canonicalize`, `truncate`,
//! `norm`). Both are generic over `Mps<St, L>` and dispatch through the
//! sealed trait; neither names a storage or layout primitive on its own
//! bounds beyond the chain's own type parameters.
//!
//! # Operation authority
//!
//! Every operation takes its compute backend explicitly at the call site
//! and dispatches all kernels through that handle via the explicit-backend
//! (`*_with_backend`) linalg paths. The backend is bound by
//! [`OpsFor<Self::Storage>`](ariadnetor_tensor::OpsFor) — the same capability
//! gate the linalg surface enforces — so only a backend that has declared
//! it operates on this chain's storage can be supplied. The chain itself
//! carries no backend, so there is a single, unambiguous authority per
//! call. Block-sparse layout-order safety is enforced inside the linalg
//! twins (their release-active entry checks compare each operand's layout
//! order against the supplied backend's preferred order); dense paths
//! self-normalize.

use std::num::NonZeroUsize;

use ariadnetor_core::Scalar;
use ariadnetor_tensor::{
    BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, OpsFor, Sector, Storage,
    StorageFor, TensorLayout,
};

use super::types::{
    ApplyMethod, Mpo, Mps, SuccessiveRandomizedParams, TruncResult, TruncateParams, VariationalInit,
};

mod sealed {
    use ariadnetor_core::Scalar;
    use ariadnetor_tensor::{
        BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, Sector,
    };

    use crate::types::Mps;

    /// Crate-private supertrait that seals [`super::MpsOps`] against
    /// downstream implementation. Carries no associated surface, so the
    /// public trait projects no taxonomy through it.
    pub trait Sealed {}

    impl<T: Scalar> Sealed for Mps<DenseStorage<T>, DenseLayout> {}
    impl<T: Scalar, S: Sector> Sealed for Mps<BlockSparseStorage<T>, BlockSparseLayout<S>> {}
}

/// Chain-keyed dispatch trait for MPS operations.
///
/// Implemented for the two concrete `Mps` chain types, each pairing a
/// storage / layout via [`MpsOps::Storage`] / [`MpsOps::Layout`] and
/// routing each operation to its storage-specific kernel. Sealed: only the
/// in-crate impls exist, and the trait carries the storage / layout taxa as
/// associated types rather than exposing them on a `pub` bound surface.
///
/// The methods are the kernels the public free functions ([`inner`],
/// [`braket`], [`apply`]) and the [`Mps`] inherent methods (`canonicalize`,
/// `truncate`, `norm`) forward to; they are not called directly.
pub trait MpsOps<T: Scalar>: sealed::Sealed {
    /// Layout type paired with this chain. Needed to name the sibling
    /// chain type [`Mpo<Self::Storage, Self::Layout>`](Mpo) in the
    /// multi-arg methods, which `Self = Mps<St, L>` alone cannot recover.
    type Layout: TensorLayout;
    /// Storage type paired with this chain. The `StorageFor<Self::Layout>`
    /// bound lets the methods name the sibling `Mpo<Self::Storage,
    /// Self::Layout>` (whose own bound requires the pairing).
    type Storage: Storage + StorageFor<Self::Layout>;

    /// Compute the inner product ⟨ψ|φ⟩.
    fn inner_k<B: OpsFor<Self::Storage>>(backend: &B, psi: &Self, phi: &Self) -> T;

    /// Compute the norm ‖ψ‖.
    fn norm_k<B: OpsFor<Self::Storage>>(backend: &B, psi: &Self) -> T::Real;

    /// Compute the expectation value ⟨ψ|O|φ⟩.
    fn braket_k<B: OpsFor<Self::Storage>>(
        backend: &B,
        psi: &Self,
        op: &Mpo<Self::Storage, Self::Layout>,
        phi: &Self,
    ) -> T;

    /// Apply an MPO to an MPS: O|ψ⟩ via the streaming-naive algorithm.
    ///
    /// `forward_cap = None` keeps the forward pass on its QR-only branch
    /// (lossless streaming naive). `forward_cap = Some(k)` falls to a
    /// truncated SVD with cap `k * chi_max` when the natural per-site
    /// forward rank exceeds it.
    fn apply_k<B: OpsFor<Self::Storage>>(
        backend: &B,
        op: &Mpo<Self::Storage, Self::Layout>,
        psi: &Self,
        params: Option<&TruncateParams>,
        forward_cap: Option<NonZeroUsize>,
    ) -> Self
    where
        Self: Sized;

    /// Apply an MPO to an MPS via the Stoudenmire-White zip-up algorithm:
    /// right-canonicalize `psi`, then a single forward sweep with per-site
    /// SVD truncated directly to `chi_max` (from `params.svd`) and no
    /// backward pass. `params = None` keeps full rank (lossless).
    fn apply_zipup_k<B: OpsFor<Self::Storage>>(
        backend: &B,
        op: &Mpo<Self::Storage, Self::Layout>,
        psi: &Self,
        params: Option<&TruncateParams>,
    ) -> Self
    where
        Self: Sized;

    /// Apply an MPO to an MPS via the density-matrix algorithm: materialize
    /// the untruncated product, accumulate the `⟨φ|φ⟩` right environment, then
    /// a single forward sweep truncating each bond's reduced density matrix to
    /// `chi_max` (from `params.svd`) via its dominant eigenvectors. `params =
    /// None` keeps full rank (lossless).
    fn apply_density_matrix_k<B: OpsFor<Self::Storage>>(
        backend: &B,
        op: &Mpo<Self::Storage, Self::Layout>,
        psi: &Self,
        params: Option<&TruncateParams>,
    ) -> Self
    where
        Self: Sized;

    /// Apply an MPO to an MPS via single-site variational fitting: seed from
    /// zip-up or density-matrix (`init`, truncated to `params.svd.chi_max`),
    /// then DMRG-style single-site sweeps whose local update is the `⟨φ|W|ψ⟩`
    /// environment projection, iterated until the center overlap's relative
    /// change is at or below `tol` or `max_sweeps` cycles run.
    ///
    /// Host-pinned: builds on the host-resident `BraketEnvs` primitive, so
    /// `backend` is not consulted (the concrete impls run on `Host`).
    fn apply_variational_k<B: OpsFor<Self::Storage>>(
        backend: &B,
        op: &Mpo<Self::Storage, Self::Layout>,
        psi: &Self,
        params: Option<&TruncateParams>,
        init: VariationalInit,
        max_sweeps: usize,
        tol: f64,
    ) -> Self
    where
        Self: Sized;

    /// Apply an MPO to an MPS via successive randomized compression (SRC):
    /// a single right-to-left randomized-QB sweep with adaptive or fixed-rank
    /// bond selection (see [`ApplyMethod::SuccessiveRandomized`]). Dense-only:
    /// the block-sparse impl panics because the Gaussian sketch mixes
    /// symmetry sectors.
    fn apply_successive_randomized_k<B: OpsFor<Self::Storage>>(
        backend: &B,
        op: &Mpo<Self::Storage, Self::Layout>,
        psi: &Self,
        params: Option<&TruncateParams>,
        src: SuccessiveRandomizedParams,
    ) -> Self
    where
        Self: Sized;

    /// Position the orthogonality center at `center`.
    fn canon_k<B: OpsFor<Self::Storage>>(backend: &B, chain: &mut Self, center: usize);

    /// Truncate bond dimensions according to `params`.
    fn trunc_k<B: OpsFor<Self::Storage>>(
        backend: &B,
        chain: &mut Self,
        params: &TruncateParams,
    ) -> TruncResult<T>;
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T: Scalar> MpsOps<T> for Mps<DenseStorage<T>, DenseLayout> {
    type Storage = DenseStorage<T>;
    type Layout = DenseLayout;

    fn inner_k<B: OpsFor<DenseStorage<T>>>(backend: &B, psi: &Self, phi: &Self) -> T {
        super::inner::inner_dense(backend, psi, phi)
    }

    fn norm_k<B: OpsFor<DenseStorage<T>>>(backend: &B, psi: &Self) -> T::Real {
        super::inner::norm_dense(backend, psi)
    }

    fn braket_k<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        psi: &Self,
        op: &Mpo<DenseStorage<T>, DenseLayout>,
        phi: &Self,
    ) -> T {
        super::inner::braket_dense(backend, psi, op, phi)
    }

    fn apply_k<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        op: &Mpo<DenseStorage<T>, DenseLayout>,
        psi: &Self,
        params: Option<&TruncateParams>,
        forward_cap: Option<NonZeroUsize>,
    ) -> Self {
        super::apply::apply_streaming_naive_dense(backend, op, psi, params, forward_cap)
    }

    fn apply_zipup_k<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        op: &Mpo<DenseStorage<T>, DenseLayout>,
        psi: &Self,
        params: Option<&TruncateParams>,
    ) -> Self {
        super::apply::apply_zipup_dense(backend, op, psi, params)
    }

    fn apply_density_matrix_k<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        op: &Mpo<DenseStorage<T>, DenseLayout>,
        psi: &Self,
        params: Option<&TruncateParams>,
    ) -> Self {
        super::apply::apply_density_matrix_dense(backend, op, psi, params)
    }

    // Host-pinned: `_backend` is ignored — the kernel runs on `Host` (see
    // `ApplyMethod::Variational`).
    fn apply_variational_k<B: OpsFor<DenseStorage<T>>>(
        _backend: &B,
        op: &Mpo<DenseStorage<T>, DenseLayout>,
        psi: &Self,
        params: Option<&TruncateParams>,
        init: VariationalInit,
        max_sweeps: usize,
        tol: f64,
    ) -> Self {
        super::apply::apply_variational_dense(op, psi, params, init, max_sweeps, tol)
    }

    fn apply_successive_randomized_k<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        op: &Mpo<DenseStorage<T>, DenseLayout>,
        psi: &Self,
        params: Option<&TruncateParams>,
        src: SuccessiveRandomizedParams,
    ) -> Self {
        super::apply::apply_successive_randomized_dense(backend, op, psi, params, src)
    }

    fn canon_k<B: OpsFor<DenseStorage<T>>>(backend: &B, chain: &mut Self, center: usize) {
        super::canonicalize::canonicalize_dense(backend, chain, center);
    }

    fn trunc_k<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        chain: &mut Self,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        super::truncate::truncate_dense(backend, chain, params)
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T: Scalar, S: Sector> MpsOps<T> for Mps<BlockSparseStorage<T>, BlockSparseLayout<S>> {
    type Storage = BlockSparseStorage<T>;
    type Layout = BlockSparseLayout<S>;

    fn inner_k<B: OpsFor<BlockSparseStorage<T>>>(backend: &B, psi: &Self, phi: &Self) -> T {
        super::inner::inner_bsp(backend, psi, phi)
    }

    fn norm_k<B: OpsFor<BlockSparseStorage<T>>>(backend: &B, psi: &Self) -> T::Real {
        super::inner::norm_bsp(backend, psi)
    }

    fn braket_k<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        psi: &Self,
        op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        phi: &Self,
    ) -> T {
        super::inner::braket_bsp(backend, psi, op, phi)
    }

    fn apply_k<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        psi: &Self,
        params: Option<&TruncateParams>,
        forward_cap: Option<NonZeroUsize>,
    ) -> Self {
        super::apply::apply_streaming_naive_bsp(backend, op, psi, params, forward_cap)
    }

    fn apply_zipup_k<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        psi: &Self,
        params: Option<&TruncateParams>,
    ) -> Self {
        super::apply::apply_zipup_bsp(backend, op, psi, params)
    }

    fn apply_density_matrix_k<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        psi: &Self,
        params: Option<&TruncateParams>,
    ) -> Self {
        super::apply::apply_density_matrix_bsp(backend, op, psi, params)
    }

    // Host-pinned: `_backend` is ignored — the kernel runs on `Host` (see
    // `ApplyMethod::Variational`).
    fn apply_variational_k<B: OpsFor<BlockSparseStorage<T>>>(
        _backend: &B,
        op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        psi: &Self,
        params: Option<&TruncateParams>,
        init: VariationalInit,
        max_sweeps: usize,
        tol: f64,
    ) -> Self {
        super::apply::apply_variational_bsp(op, psi, params, init, max_sweeps, tol)
    }

    fn apply_successive_randomized_k<B: OpsFor<BlockSparseStorage<T>>>(
        _backend: &B,
        _op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        _psi: &Self,
        _params: Option<&TruncateParams>,
        _src: SuccessiveRandomizedParams,
    ) -> Self {
        panic!(
            "ApplyMethod::SuccessiveRandomized is dense-only: the Gaussian sketch \
             mixes symmetry sectors, so a block-sparse product would not preserve \
             the chain's quantum-number structure"
        );
    }

    fn canon_k<B: OpsFor<BlockSparseStorage<T>>>(backend: &B, chain: &mut Self, center: usize) {
        super::canonicalize::canonicalize_bsp(backend, chain, center);
    }

    fn trunc_k<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        chain: &mut Self,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        super::truncate::truncate_bsp(backend, chain, params)
    }
}

// ---------------------------------------------------------------------------
// Single-chain inherent methods on `Mps`. Keyed on the chain, they dispatch
// through the sealed trait without naming a storage / layout primitive on
// their public bounds beyond the chain's own type parameters.
// ---------------------------------------------------------------------------

impl<St, L> Mps<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Position the orthogonality center at `center`.
    pub fn canonicalize<T, B>(&mut self, backend: &B, center: usize)
    where
        T: Scalar,
        Mps<St, L>: MpsOps<T, Storage = St, Layout = L>,
        B: OpsFor<St>,
    {
        <Self as MpsOps<T>>::canon_k(backend, self, center);
    }

    /// Truncate bond dimensions according to `params`.
    pub fn truncate<T, B>(&mut self, backend: &B, params: &TruncateParams) -> TruncResult<T>
    where
        T: Scalar,
        Mps<St, L>: MpsOps<T, Storage = St, Layout = L>,
        B: OpsFor<St>,
    {
        <Self as MpsOps<T>>::trunc_k(backend, self, params)
    }

    /// Compute the norm ‖ψ‖.
    pub fn norm<T, B>(&self, backend: &B) -> T::Real
    where
        T: Scalar,
        Mps<St, L>: MpsOps<T, Storage = St, Layout = L>,
        B: OpsFor<St>,
    {
        <Self as MpsOps<T>>::norm_k(backend, self)
    }
}

// ---------------------------------------------------------------------------
// Multi-arg public free functions — dispatch via the sealed `MpsOps` trait
// so callers write `inner(backend, psi, phi)` over `Mps<St, L>` without
// naming the storage explicitly.
// ---------------------------------------------------------------------------

/// Compute the inner product ⟨ψ|φ⟩.
pub fn inner<T, St, L, B>(backend: &B, psi: &Mps<St, L>, phi: &Mps<St, L>) -> T
where
    T: Scalar,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    Mps<St, L>: MpsOps<T, Storage = St, Layout = L>,
    B: OpsFor<St>,
{
    <Mps<St, L> as MpsOps<T>>::inner_k(backend, psi, phi)
}

/// Compute the expectation value ⟨ψ|O|φ⟩.
pub fn braket<T, St, L, B>(backend: &B, psi: &Mps<St, L>, op: &Mpo<St, L>, phi: &Mps<St, L>) -> T
where
    T: Scalar,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    Mps<St, L>: MpsOps<T, Storage = St, Layout = L>,
    B: OpsFor<St>,
{
    <Mps<St, L> as MpsOps<T>>::braket_k(backend, psi, op, phi)
}

/// Apply an MPO to an MPS: O|ψ⟩ via the streaming-naive algorithm with the
/// default lossless forward sweep (`forward_cap = None`).
///
/// Equivalent to `apply_with_method(backend, op, psi, params, ApplyMethod::default())`.
pub fn apply<T, St, L, B>(
    backend: &B,
    op: &Mpo<St, L>,
    psi: &Mps<St, L>,
    params: Option<&TruncateParams>,
) -> Mps<St, L>
where
    T: Scalar,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    Mps<St, L>: MpsOps<T, Storage = St, Layout = L>,
    B: OpsFor<St>,
{
    <Mps<St, L> as MpsOps<T>>::apply_k(backend, op, psi, params, None)
}

/// Apply an MPO to an MPS using the requested algorithm.
///
/// - `ApplyMethod::StreamingNaive` runs the per-site product with a streaming
///   forward QR/SVD sweep followed by an optional `canonicalize` + `truncate`.
/// - `ApplyMethod::ZipUp` selects the Stoudenmire-White single-pass zip-up
///   algorithm (right-canonicalize, then one forward sweep with per-site
///   truncation to `chi_max` and no backward pass).
/// - `ApplyMethod::DensityMatrix` materializes the untruncated product,
///   accumulates the `⟨φ|φ⟩` right environment, then a single forward sweep
///   truncating each bond's reduced density matrix to `chi_max` via its
///   dominant eigenvectors.
/// - `ApplyMethod::Variational` seeds from zip-up or density-matrix and refines
///   the fit at the fixed seed bond via single-site DMRG-style sweeps. This
///   method is **host-pinned**: it builds on the host-resident `BraketEnvs`
///   primitive, so `backend` is not consulted for it.
/// - `ApplyMethod::SuccessiveRandomized` computes the compressed product
///   directly via a single right-to-left randomized-QB sweep with adaptive
///   or fixed-rank bond selection. **Dense-only** (panics on block-sparse
///   chains).
pub fn apply_with_method<T, St, L, B>(
    backend: &B,
    op: &Mpo<St, L>,
    psi: &Mps<St, L>,
    params: Option<&TruncateParams>,
    method: ApplyMethod,
) -> Mps<St, L>
where
    T: Scalar,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    Mps<St, L>: MpsOps<T, Storage = St, Layout = L>,
    B: OpsFor<St>,
{
    match method {
        ApplyMethod::StreamingNaive { forward_cap } => {
            <Mps<St, L> as MpsOps<T>>::apply_k(backend, op, psi, params, forward_cap)
        }
        ApplyMethod::ZipUp => <Mps<St, L> as MpsOps<T>>::apply_zipup_k(backend, op, psi, params),
        ApplyMethod::DensityMatrix => {
            <Mps<St, L> as MpsOps<T>>::apply_density_matrix_k(backend, op, psi, params)
        }
        ApplyMethod::Variational {
            init,
            max_sweeps,
            tol,
        } => <Mps<St, L> as MpsOps<T>>::apply_variational_k(
            backend, op, psi, params, init, max_sweeps, tol,
        ),
        ApplyMethod::SuccessiveRandomized(src) => {
            <Mps<St, L> as MpsOps<T>>::apply_successive_randomized_k(backend, op, psi, params, src)
        }
    }
}
