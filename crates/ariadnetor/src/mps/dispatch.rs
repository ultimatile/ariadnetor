//! Dispatch trait for MPS operations over different storage types.
//!
//! [`MpsOps`] enables algorithms (DMRG, TDVP, etc.) to be written once,
//! generic over `Dense<T>` and `BlockSparse<T, S>`.

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_tensor::{BlockSparse, Dense, Sector};

use arnet_tensor::TensorRepr;

use super::chain::TensorChain;
use super::types::{Mpo, Mps, TruncResult, TruncateParams};

/// Dispatch trait for MPS/MPO operations.
///
/// Implemented for [`Dense<T>`] and [`BlockSparse<T, S>`], routing each
/// operation to its storage-specific implementation. Algorithms written
/// against `R: MpsOps` work with both storage types without duplication.
pub trait MpsOps: TensorRepr<Elem: Scalar> + Sized {
    /// Position the orthogonality center at `center`.
    fn canonicalize<B: ComputeBackend>(chain: &mut impl TensorChain<Self, B>, center: usize);

    /// Truncate bond dimensions according to `params`.
    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChain<Self, B>,
        params: &TruncateParams,
    ) -> TruncResult<Self::Elem>;

    /// Compute the inner product ⟨ψ|φ⟩.
    fn inner<B: ComputeBackend>(psi: &Mps<Self, B>, phi: &Mps<Self, B>) -> Self::Elem;

    /// Compute the norm ‖ψ‖.
    fn norm<B: ComputeBackend>(psi: &Mps<Self, B>) -> <Self::Elem as Scalar>::Real;

    /// Compute the expectation value ⟨ψ|O|φ⟩.
    fn braket<B: ComputeBackend>(
        psi: &Mps<Self, B>,
        op: &Mpo<Self, B>,
        phi: &Mps<Self, B>,
    ) -> Self::Elem;

    /// Apply an MPO to an MPS: O|ψ⟩.
    fn apply<B: ComputeBackend>(
        op: &Mpo<Self, B>,
        psi: &Mps<Self, B>,
        params: Option<&TruncateParams>,
    ) -> Mps<Self, B>;
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T: Scalar> MpsOps for Dense<T> {
    fn canonicalize<B: ComputeBackend>(chain: &mut impl TensorChain<Self, B>, center: usize) {
        super::canonicalize::canonicalize_dense(chain, center);
    }

    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChain<Self, B>,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        super::truncate::truncate_dense(chain, params)
    }

    fn inner<B: ComputeBackend>(psi: &Mps<Self, B>, phi: &Mps<Self, B>) -> T {
        super::inner::inner_dense(psi, phi)
    }

    fn norm<B: ComputeBackend>(psi: &Mps<Self, B>) -> T::Real {
        super::inner::norm_dense(psi)
    }

    fn braket<B: ComputeBackend>(psi: &Mps<Self, B>, op: &Mpo<Self, B>, phi: &Mps<Self, B>) -> T {
        super::inner::braket_dense(psi, op, phi)
    }

    fn apply<B: ComputeBackend>(
        op: &Mpo<Self, B>,
        psi: &Mps<Self, B>,
        params: Option<&TruncateParams>,
    ) -> Mps<Self, B> {
        super::apply::apply_dense(op, psi, params)
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T: Scalar, S: Sector> MpsOps for BlockSparse<T, S> {
    fn canonicalize<B: ComputeBackend>(chain: &mut impl TensorChain<Self, B>, center: usize) {
        super::canonicalize::canonicalize_bsp(chain, center);
    }

    fn truncate<B: ComputeBackend>(
        chain: &mut impl TensorChain<Self, B>,
        params: &TruncateParams,
    ) -> TruncResult<T> {
        super::truncate::truncate_bsp(chain, params)
    }

    fn inner<B: ComputeBackend>(psi: &Mps<Self, B>, phi: &Mps<Self, B>) -> T {
        super::inner::inner_bsp(psi, phi)
    }

    fn norm<B: ComputeBackend>(psi: &Mps<Self, B>) -> T::Real {
        super::inner::norm_bsp(psi)
    }

    fn braket<B: ComputeBackend>(psi: &Mps<Self, B>, op: &Mpo<Self, B>, phi: &Mps<Self, B>) -> T {
        super::inner::braket_bsp(psi, op, phi)
    }

    fn apply<B: ComputeBackend>(
        op: &Mpo<Self, B>,
        psi: &Mps<Self, B>,
        params: Option<&TruncateParams>,
    ) -> Mps<Self, B> {
        super::apply::apply_bsp(op, psi, params)
    }
}

// ---------------------------------------------------------------------------
// Unified free functions
// ---------------------------------------------------------------------------

/// Position the orthogonality center at `center`.
pub fn canonicalize<R: MpsOps, B: ComputeBackend>(
    chain: &mut impl TensorChain<R, B>,
    center: usize,
) {
    R::canonicalize(chain, center);
}

/// Truncate bond dimensions according to `params`.
pub fn truncate<R: MpsOps, B: ComputeBackend>(
    chain: &mut impl TensorChain<R, B>,
    params: &TruncateParams,
) -> TruncResult<R::Elem> {
    R::truncate(chain, params)
}

/// Compute the inner product ⟨ψ|φ⟩.
pub fn inner<R: MpsOps, B: ComputeBackend>(psi: &Mps<R, B>, phi: &Mps<R, B>) -> R::Elem {
    R::inner(psi, phi)
}

/// Compute the norm ‖ψ‖.
pub fn norm<R: MpsOps, B: ComputeBackend>(psi: &Mps<R, B>) -> <R::Elem as Scalar>::Real {
    R::norm(psi)
}

/// Compute the expectation value ⟨ψ|O|φ⟩.
pub fn braket<R: MpsOps, B: ComputeBackend>(
    psi: &Mps<R, B>,
    op: &Mpo<R, B>,
    phi: &Mps<R, B>,
) -> R::Elem {
    R::braket(psi, op, phi)
}

/// Apply an MPO to an MPS: O|ψ⟩.
pub fn apply<R: MpsOps, B: ComputeBackend>(
    op: &Mpo<R, B>,
    psi: &Mps<R, B>,
    params: Option<&TruncateParams>,
) -> Mps<R, B> {
    R::apply(op, psi, params)
}
