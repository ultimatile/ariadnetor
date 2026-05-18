//! Dispatch traits for MPS operations.
//!
//! - [`MpsOpsRepr`] is implemented for `R: TensorRepr` (i.e.
//!   [`Dense<T>`] and [`BlockSparse<T, S>`]). It routes operations on
//!   [`MpsRepr`] / [`MpoRepr`] chains to the `*_repr` algorithm
//!   bodies in [`super`].
//! - [`MpsOps<L>`] is implemented for `St: Storage + StorageFor<L>`
//!   (i.e. [`DenseStorage<T>`] with `L = DenseLayout` and
//!   [`BlockSparseStorage<T>`] with `L = BlockSparseLayout<S>`). It
//!   routes operations on [`Mps`] / [`Mpo`] chains to the
//!   bare-named algorithm bodies in [`super`].

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_tensor::{
    BlockSparse, BlockSparseLayout, BlockSparseStorage, Dense, DenseLayout, DenseStorage, Sector,
    Storage, StorageFor, TensorLayout, TensorRepr,
};

use super::chain::{TensorChain, TensorChainRepr};
use super::types::{ApplyMethod, Mpo, MpoRepr, Mps, MpsRepr, TruncResult, TruncateParams};

// ============================================================================
// `R: TensorRepr` dispatch — `MpsRepr` / `MpoRepr` chains
// ============================================================================

/// Dispatch trait for MPS / MPO operations whose chain sites are
/// `R: TensorRepr`.
///
/// Implemented for [`Dense<T>`] and [`BlockSparse<T, S>`]; routes each
/// operation on [`MpsRepr`] / [`MpoRepr`] chains to its
/// storage-specific implementation.
pub trait MpsOpsRepr: TensorRepr + Sized {
    /// Position the orthogonality center at `center`.
    fn canonicalize<B: ComputeBackend>(chain: &mut impl TensorChainRepr<Self, B>, center: usize);

    /// Truncate bond dimensions according to `params`.
    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChainRepr<Self, B>,
        params: &TruncateParams,
    ) -> TruncResult<Self::Elem>;

    /// Compute the inner product ⟨ψ|φ⟩.
    fn inner<B: ComputeBackend>(psi: &MpsRepr<Self, B>, phi: &MpsRepr<Self, B>) -> Self::Elem;

    /// Compute the norm ‖ψ‖.
    fn norm<B: ComputeBackend>(psi: &MpsRepr<Self, B>) -> <Self::Elem as Scalar>::Real;

    /// Compute the expectation value ⟨ψ|O|φ⟩.
    fn braket<B: ComputeBackend>(
        psi: &MpsRepr<Self, B>,
        op: &MpoRepr<Self, B>,
        phi: &MpsRepr<Self, B>,
    ) -> Self::Elem;

    /// Apply an MPO to an MPS: O|ψ⟩.
    fn apply<B: ComputeBackend>(
        op: &MpoRepr<Self, B>,
        psi: &MpsRepr<Self, B>,
        params: Option<&TruncateParams>,
    ) -> MpsRepr<Self, B>;

    /// Apply an MPO to an MPS via the zip-up algorithm.
    ///
    /// Implementations may panic when zip-up is not yet supported for
    /// the underlying storage type — callers that need a portable
    /// fallback should route through [`apply_with_method_repr`] with
    /// [`ApplyMethod::Naive`].
    fn apply_zipup<B: ComputeBackend>(
        op: &MpoRepr<Self, B>,
        psi: &MpsRepr<Self, B>,
        params: Option<&TruncateParams>,
    ) -> MpsRepr<Self, B>;
}

impl<T: Scalar> MpsOpsRepr for Dense<T> {
    fn canonicalize<B: ComputeBackend>(chain: &mut impl TensorChainRepr<Self, B>, center: usize) {
        super::canonicalize::canonicalize_dense_repr(chain, center);
    }

    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChainRepr<Self, B>,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        super::truncate::truncate_dense_repr(chain, params)
    }

    fn inner<B: ComputeBackend>(psi: &MpsRepr<Self, B>, phi: &MpsRepr<Self, B>) -> T {
        super::inner::inner_dense_repr(psi, phi)
    }

    fn norm<B: ComputeBackend>(psi: &MpsRepr<Self, B>) -> T::Real {
        super::inner::norm_dense_repr(psi)
    }

    fn braket<B: ComputeBackend>(
        psi: &MpsRepr<Self, B>,
        op: &MpoRepr<Self, B>,
        phi: &MpsRepr<Self, B>,
    ) -> T {
        super::inner::braket_dense_repr(psi, op, phi)
    }

    fn apply<B: ComputeBackend>(
        op: &MpoRepr<Self, B>,
        psi: &MpsRepr<Self, B>,
        params: Option<&TruncateParams>,
    ) -> MpsRepr<Self, B> {
        super::apply::apply_dense_repr(op, psi, params)
    }

    fn apply_zipup<B: ComputeBackend>(
        op: &MpoRepr<Self, B>,
        psi: &MpsRepr<Self, B>,
        params: Option<&TruncateParams>,
    ) -> MpsRepr<Self, B> {
        super::apply::apply_zipup_dense_repr(op, psi, params)
    }
}

impl<T: Scalar, S: Sector> MpsOpsRepr for BlockSparse<T, S> {
    fn canonicalize<B: ComputeBackend>(chain: &mut impl TensorChainRepr<Self, B>, center: usize) {
        super::canonicalize::canonicalize_bsp_repr(chain, center);
    }

    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChainRepr<Self, B>,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        super::truncate::truncate_bsp_repr(chain, params)
    }

    fn inner<B: ComputeBackend>(psi: &MpsRepr<Self, B>, phi: &MpsRepr<Self, B>) -> T {
        super::inner::inner_bsp_repr(psi, phi)
    }

    fn norm<B: ComputeBackend>(psi: &MpsRepr<Self, B>) -> T::Real {
        super::inner::norm_bsp_repr(psi)
    }

    fn braket<B: ComputeBackend>(
        psi: &MpsRepr<Self, B>,
        op: &MpoRepr<Self, B>,
        phi: &MpsRepr<Self, B>,
    ) -> T {
        super::inner::braket_bsp_repr(psi, op, phi)
    }

    fn apply<B: ComputeBackend>(
        op: &MpoRepr<Self, B>,
        psi: &MpsRepr<Self, B>,
        params: Option<&TruncateParams>,
    ) -> MpsRepr<Self, B> {
        super::apply::apply_bsp_repr(op, psi, params)
    }

    fn apply_zipup<B: ComputeBackend>(
        op: &MpoRepr<Self, B>,
        psi: &MpsRepr<Self, B>,
        params: Option<&TruncateParams>,
    ) -> MpsRepr<Self, B> {
        super::apply::apply_zipup_bsp_repr(op, psi, params)
    }
}

// ----------------------------------------------------------------------------
// `R: TensorRepr` free functions
// ----------------------------------------------------------------------------

/// Position the orthogonality center at `center` (`*Repr` chains).
pub fn canonicalize_repr<R: MpsOpsRepr, B: ComputeBackend>(
    chain: &mut impl TensorChainRepr<R, B>,
    center: usize,
) {
    R::canonicalize(chain, center);
}

/// Truncate bond dimensions according to `params` (`*Repr` chains).
pub fn truncate_repr<R: MpsOpsRepr, B: ComputeBackend>(
    chain: &mut impl TensorChainRepr<R, B>,
    params: &TruncateParams,
) -> TruncResult<R::Elem> {
    R::truncate(chain, params)
}

/// Compute the inner product ⟨ψ|φ⟩ (`*Repr` chains).
pub fn inner_repr<R: MpsOpsRepr, B: ComputeBackend>(
    psi: &MpsRepr<R, B>,
    phi: &MpsRepr<R, B>,
) -> R::Elem {
    R::inner(psi, phi)
}

/// Compute the norm ‖ψ‖ (`*Repr` chains).
pub fn norm_repr<R: MpsOpsRepr, B: ComputeBackend>(
    psi: &MpsRepr<R, B>,
) -> <R::Elem as Scalar>::Real {
    R::norm(psi)
}

/// Compute the expectation value ⟨ψ|O|φ⟩ (`*Repr` chains).
pub fn braket_repr<R: MpsOpsRepr, B: ComputeBackend>(
    psi: &MpsRepr<R, B>,
    op: &MpoRepr<R, B>,
    phi: &MpsRepr<R, B>,
) -> R::Elem {
    R::braket(psi, op, phi)
}

/// Apply an MPO to an MPS: O|ψ⟩ (`*Repr` chains).
///
/// Equivalent to [`apply_with_method_repr`] with [`ApplyMethod::Naive`].
pub fn apply_repr<R: MpsOpsRepr, B: ComputeBackend>(
    op: &MpoRepr<R, B>,
    psi: &MpsRepr<R, B>,
    params: Option<&TruncateParams>,
) -> MpsRepr<R, B> {
    R::apply(op, psi, params)
}

/// Apply an MPO to an MPS using the requested algorithm (`*Repr`
/// chains).
///
/// # Panics
///
/// Panics if the chosen `method` is not supported for the storage type
/// (e.g. [`ApplyMethod::ZipUp`] on [`BlockSparse`]).
pub fn apply_with_method_repr<R: MpsOpsRepr, B: ComputeBackend>(
    op: &MpoRepr<R, B>,
    psi: &MpsRepr<R, B>,
    params: Option<&TruncateParams>,
    method: ApplyMethod,
) -> MpsRepr<R, B> {
    match method {
        ApplyMethod::Naive => R::apply(op, psi, params),
        ApplyMethod::ZipUp => R::apply_zipup(op, psi, params),
    }
}

// ============================================================================
// `St: Storage + StorageFor<L>` dispatch — `Mps` / `Mpo` chains
// ============================================================================

/// Dispatch trait for MPS / MPO operations over the storage / layout split.
///
/// Implemented for [`DenseStorage<T>`] (with `L = DenseLayout`) and
/// [`BlockSparseStorage<T>`] (with `L = BlockSparseLayout<S>`). Algorithms
/// written against `St: MpsOps<L>` work with both flavors without duplication.
pub trait MpsOps<L: TensorLayout>: Storage + StorageFor<L> + Sized {
    /// Scalar element type yielded by inner-product / norm operations.
    type Elem: Scalar;

    /// Position the orthogonality center at `center`.
    fn canonicalize<B: ComputeBackend>(chain: &mut impl TensorChain<Self, L, B>, center: usize);

    /// Truncate bond dimensions according to `params`.
    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChain<Self, L, B>,
        params: &TruncateParams,
    ) -> TruncResult<Self::Elem>;

    /// Compute the inner product ⟨ψ|φ⟩.
    fn inner<B: ComputeBackend>(psi: &Mps<Self, L, B>, phi: &Mps<Self, L, B>) -> Self::Elem;

    /// Compute the norm ‖ψ‖.
    fn norm<B: ComputeBackend>(psi: &Mps<Self, L, B>) -> <Self::Elem as Scalar>::Real;

    /// Compute the expectation value ⟨ψ|O|φ⟩.
    fn braket<B: ComputeBackend>(
        psi: &Mps<Self, L, B>,
        op: &Mpo<Self, L, B>,
        phi: &Mps<Self, L, B>,
    ) -> Self::Elem;

    /// Apply an MPO to an MPS: O|ψ⟩.
    fn apply<B: ComputeBackend>(
        op: &Mpo<Self, L, B>,
        psi: &Mps<Self, L, B>,
        params: Option<&TruncateParams>,
    ) -> Mps<Self, L, B>;

    /// Apply an MPO to an MPS via the zip-up algorithm.
    ///
    /// Implementations may panic when zip-up is not yet supported for
    /// the underlying storage type.
    fn apply_zipup<B: ComputeBackend>(
        op: &Mpo<Self, L, B>,
        psi: &Mps<Self, L, B>,
        params: Option<&TruncateParams>,
    ) -> Mps<Self, L, B>;
}

impl<T: Scalar> MpsOps<DenseLayout> for DenseStorage<T> {
    type Elem = T;

    fn canonicalize<B: ComputeBackend>(
        chain: &mut impl TensorChain<Self, DenseLayout, B>,
        center: usize,
    ) {
        super::canonicalize::canonicalize_dense(chain, center);
    }

    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChain<Self, DenseLayout, B>,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        super::truncate_data::truncate_dense(chain, params)
    }

    fn inner<B: ComputeBackend>(
        psi: &Mps<Self, DenseLayout, B>,
        phi: &Mps<Self, DenseLayout, B>,
    ) -> T {
        super::inner::inner_dense(psi, phi)
    }

    fn norm<B: ComputeBackend>(psi: &Mps<Self, DenseLayout, B>) -> T::Real {
        super::inner::norm_dense(psi)
    }

    fn braket<B: ComputeBackend>(
        psi: &Mps<Self, DenseLayout, B>,
        op: &Mpo<Self, DenseLayout, B>,
        phi: &Mps<Self, DenseLayout, B>,
    ) -> T {
        super::inner::braket_dense(psi, op, phi)
    }

    fn apply<B: ComputeBackend>(
        op: &Mpo<Self, DenseLayout, B>,
        psi: &Mps<Self, DenseLayout, B>,
        params: Option<&TruncateParams>,
    ) -> Mps<Self, DenseLayout, B> {
        super::apply_data::apply_dense(op, psi, params)
    }

    fn apply_zipup<B: ComputeBackend>(
        op: &Mpo<Self, DenseLayout, B>,
        psi: &Mps<Self, DenseLayout, B>,
        params: Option<&TruncateParams>,
    ) -> Mps<Self, DenseLayout, B> {
        super::apply_data::apply_zipup_dense(op, psi, params)
    }
}

impl<T: Scalar, S: Sector> MpsOps<BlockSparseLayout<S>> for BlockSparseStorage<T> {
    type Elem = T;

    fn canonicalize<B: ComputeBackend>(
        chain: &mut impl TensorChain<Self, BlockSparseLayout<S>, B>,
        center: usize,
    ) {
        super::canonicalize::canonicalize_bsp(chain, center);
    }

    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChain<Self, BlockSparseLayout<S>, B>,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        super::truncate_data::truncate_bsp(chain, params)
    }

    fn inner<B: ComputeBackend>(
        psi: &Mps<Self, BlockSparseLayout<S>, B>,
        phi: &Mps<Self, BlockSparseLayout<S>, B>,
    ) -> T {
        super::inner::inner_bsp(psi, phi)
    }

    fn norm<B: ComputeBackend>(psi: &Mps<Self, BlockSparseLayout<S>, B>) -> T::Real {
        super::inner::norm_bsp(psi)
    }

    fn braket<B: ComputeBackend>(
        psi: &Mps<Self, BlockSparseLayout<S>, B>,
        op: &Mpo<Self, BlockSparseLayout<S>, B>,
        phi: &Mps<Self, BlockSparseLayout<S>, B>,
    ) -> T {
        super::inner::braket_bsp(psi, op, phi)
    }

    fn apply<B: ComputeBackend>(
        op: &Mpo<Self, BlockSparseLayout<S>, B>,
        psi: &Mps<Self, BlockSparseLayout<S>, B>,
        params: Option<&TruncateParams>,
    ) -> Mps<Self, BlockSparseLayout<S>, B> {
        super::apply_data::apply_bsp(op, psi, params)
    }

    fn apply_zipup<B: ComputeBackend>(
        op: &Mpo<Self, BlockSparseLayout<S>, B>,
        psi: &Mps<Self, BlockSparseLayout<S>, B>,
        params: Option<&TruncateParams>,
    ) -> Mps<Self, BlockSparseLayout<S>, B> {
        super::apply_data::apply_zipup_bsp(op, psi, params)
    }
}

// ----------------------------------------------------------------------------
// `Mps` / `Mpo` free functions
// ----------------------------------------------------------------------------

/// Position the orthogonality center at `center`.
pub fn canonicalize<St, L, B>(chain: &mut impl TensorChain<St, L, B>, center: usize)
where
    St: MpsOps<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    St::canonicalize(chain, center);
}

/// Truncate bond dimensions according to `params`.
pub fn truncate<St, L, B>(
    chain: &mut impl TensorChain<St, L, B>,
    params: &TruncateParams,
) -> TruncResult<St::Elem>
where
    St: MpsOps<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    St::truncate(chain, params)
}

/// Compute the inner product ⟨ψ|φ⟩.
pub fn inner<St, L, B>(psi: &Mps<St, L, B>, phi: &Mps<St, L, B>) -> St::Elem
where
    St: MpsOps<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    St::inner(psi, phi)
}

/// Compute the norm ‖ψ‖.
pub fn norm<St, L, B>(psi: &Mps<St, L, B>) -> <St::Elem as Scalar>::Real
where
    St: MpsOps<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    St::norm(psi)
}

/// Compute the expectation value ⟨ψ|O|φ⟩.
pub fn braket<St, L, B>(psi: &Mps<St, L, B>, op: &Mpo<St, L, B>, phi: &Mps<St, L, B>) -> St::Elem
where
    St: MpsOps<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    St::braket(psi, op, phi)
}

/// Apply an MPO to an MPS: O|ψ⟩.
///
/// Equivalent to [`apply_with_method`] with [`ApplyMethod::Naive`].
pub fn apply<St, L, B>(
    op: &Mpo<St, L, B>,
    psi: &Mps<St, L, B>,
    params: Option<&TruncateParams>,
) -> Mps<St, L, B>
where
    St: MpsOps<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    St::apply(op, psi, params)
}

/// Apply an MPO to an MPS using the requested algorithm.
///
/// See [`ApplyMethod`] for the trade-offs between variants.
///
/// # Panics
///
/// Panics if the chosen `method` is not supported for the storage type
/// (e.g. [`ApplyMethod::ZipUp`] on [`BlockSparseStorage`]).
pub fn apply_with_method<St, L, B>(
    op: &Mpo<St, L, B>,
    psi: &Mps<St, L, B>,
    params: Option<&TruncateParams>,
    method: ApplyMethod,
) -> Mps<St, L, B>
where
    St: MpsOps<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    match method {
        ApplyMethod::Naive => St::apply(op, psi, params),
        ApplyMethod::ZipUp => St::apply_zipup(op, psi, params),
    }
}
