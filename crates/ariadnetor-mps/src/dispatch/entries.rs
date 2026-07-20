//! Multi-arg public free functions — dispatch via the sealed [`MpsOps`]
//! trait so callers write `inner(backend, psi, phi)` over `Mps<St, L>`
//! without naming the storage explicitly. Single-chain operations live as
//! inherent methods on [`Mps`] in the parent module instead.

use ariadnetor_core::Scalar;
use ariadnetor_tensor::{OpsFor, Storage, StorageFor, TensorLayout};

use super::MpsOps;
use crate::types::{
    ApplyError, ApplyMethod, Mpo, Mps, SuccessiveRandomizedParams, SumTerm, TruncateParams,
};

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

/// Apply an MPO to an MPS with the default method: O|ψ⟩ via the
/// streaming-naive algorithm with the default lossless forward sweep
/// (`forward_cap = None`).
///
/// Computes the same state as
/// `apply_with_method(backend, op, psi, params, ApplyMethod::default())`,
/// but stays infallible: the current default method has no failure path,
/// so there is no error to surface. Should the default ever change to a
/// fallible method, this signature changes with it.
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
///
/// # Errors
///
/// Returns [`ApplyError::NonFinite`] when the computation was poisoned by
/// non-finite values (NaN/inf) and the poison reached a result boundary.
/// The scan runs before the optional `canonicalize` + `truncate`
/// finishing pass, so `Ok` certifies the state as assembled, not the
/// finishing pass's output (see [`ApplyError::NonFinite`] for the exact
/// contract). Currently only `ApplyMethod::SuccessiveRandomized` performs
/// this check; the other methods have no failure path and always return
/// `Ok`.
pub fn apply_with_method<T, St, L, B>(
    backend: &B,
    op: &Mpo<St, L>,
    psi: &Mps<St, L>,
    params: Option<&TruncateParams>,
    method: ApplyMethod,
) -> Result<Mps<St, L>, ApplyError>
where
    T: Scalar,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    Mps<St, L>: MpsOps<T, Storage = St, Layout = L>,
    B: OpsFor<St>,
{
    match method {
        ApplyMethod::StreamingNaive { forward_cap } => Ok(<Mps<St, L> as MpsOps<T>>::apply_k(
            backend,
            op,
            psi,
            params,
            forward_cap,
        )),
        ApplyMethod::ZipUp => Ok(<Mps<St, L> as MpsOps<T>>::apply_zipup_k(
            backend, op, psi, params,
        )),
        ApplyMethod::DensityMatrix => Ok(<Mps<St, L> as MpsOps<T>>::apply_density_matrix_k(
            backend, op, psi, params,
        )),
        ApplyMethod::Variational {
            init,
            max_sweeps,
            tol,
        } => Ok(<Mps<St, L> as MpsOps<T>>::apply_variational_k(
            backend, op, psi, params, init, max_sweeps, tol,
        )),
        ApplyMethod::SuccessiveRandomized(src) => {
            <Mps<St, L> as MpsOps<T>>::apply_successive_randomized_k(backend, op, psi, params, src)
        }
    }
}

/// Apply a coefficient-weighted sum of MPO-MPS products
/// `η ≈ Σ_t coeffs[t] · H_t ψ_t` via successive randomized compression
/// (SRC): one right-to-left randomized-QB sweep over all terms at once,
/// sharing each site's Gaussian sketch block across terms, with adaptive
/// or fixed-rank bond selection per `src`. The product MPS of the sum is
/// never materialized, and neither is any per-term product. Sharing the
/// sketch is what makes the summed panel a sketch of the summed state, so
/// the per-term compression guarantees of
/// [`ApplyMethod::SuccessiveRandomized`] carry over to the sum.
///
/// The terms must have equal chain lengths and, per site, equal MPO
/// output (bra) dimensions — the legs the shared sketch acts on. Within
/// each term the MPO ket dimension must match its MPS physical dimension;
/// input spaces may differ across terms. The default per-bond cap
/// (`src.max_dim = None`) is the sum over terms of the products of
/// maximum MPO and MPS bond dimensions, in both stopping modes.
/// Zero-weighted terms are validated but otherwise behave as absent
/// (pruned before the sweep, so a non-finite element in a disabled term
/// cannot poison the result); an all-zero coefficient list yields the
/// bond-dimension-1 zero state.
///
/// When `params` is `Some`, the standard `canonicalize` + `truncate`
/// finishing pass runs after the sweep; otherwise the result is left in
/// `Mixed { center: 0 }`. A single coefficient-one term reproduces
/// [`apply_with_method`] with `ApplyMethod::SuccessiveRandomized`
/// bit-identically at equal seeds.
///
/// # Panics
///
/// Panics on block-sparse chains (the Gaussian sketch mixes symmetry
/// sectors, so this entry is dense-only), on an empty term list,
/// zero-length chains, mismatched chain lengths, a coefficient count
/// differing from the term count, non-finite coefficients, incompatible
/// site dimensions (see above), and on the parameter violations
/// documented in [`SuccessiveRandomizedParams`].
///
/// # Errors
///
/// Returns [`ApplyError::NonFinite`] when a non-finite element reaches a
/// result boundary of the sweep (a growth round's summed sketch panel or
/// an assembled site tensor). See
/// [`ApplyMethod::SuccessiveRandomized`] for the exact contract.
pub fn apply_sum_successive_randomized<T, St, L, B>(
    backend: &B,
    terms: &[SumTerm<'_, St, L>],
    coeffs: &[T],
    params: Option<&TruncateParams>,
    src: SuccessiveRandomizedParams,
) -> Result<Mps<St, L>, ApplyError>
where
    T: Scalar,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    Mps<St, L>: MpsOps<T, Storage = St, Layout = L>,
    B: OpsFor<St>,
{
    <Mps<St, L> as MpsOps<T>>::apply_sum_successive_randomized_k(
        backend, terms, coeffs, params, src,
    )
}
