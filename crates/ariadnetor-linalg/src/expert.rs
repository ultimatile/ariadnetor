//! Expert layer: per-call [`ExecPolicy`] control.
//!
//! The default operation surface (the `*_with_backend` free functions and the
//! `DenseHostOps` / `BlockSparseHostOps` methods) auto-selects a parallelism
//! policy per call by consulting the backend's `par_for_*` hooks. This module
//! is the escape hatch for callers that need to pin the policy explicitly —
//! `Sequential` to dodge faer's small-matrix parallel slowdown, or
//! `Parallel(n)` to opt a large problem into threads the auto-heuristic would
//! leave sequential.
//!
//! These operations are defined here directly under bare names
//! (`expert::permute`, not `expert::permute_with_policy`): the `expert::` path
//! and the explicit `ExecPolicy` argument already mark the call as the
//! policy-pinned form, so the suffix would be redundant.
//!
//! Only a name that exists solely to be re-aliased to its public name is
//! dropped this way. Two internal tiers legitimately keep `_with_policy` and
//! are deliberately not surfaced here:
//!
//! - The [`LinalgDecompose`] `*_with_policy` trait methods are genuine
//!   siblings of the auto-policy `svd` / `trunc_svd` / `qr` / `lq` methods on
//!   the same trait; the bare name is taken by the auto method, so the
//!   explicit method needs a distinct one. This is not a re-alias.
//! - The `*_with_policy_dense` `pub(crate)` kernels are a different tier — the
//!   joined-data form the dense ops here delegate to — not a re-alias.
//!
//! The four decompositions (`svd` / `trunc_svd` / `qr` / `lq`) dispatch over
//! layout via [`LinalgDecompose`], so their `expert` forms serve both Dense
//! and BlockSparse from one bare name. This is also the only public entry that
//! pins an [`ExecPolicy`] on a block-sparse decomposition; the auto-policy
//! crate-root forms keep block-sparse on `Sequential`.

use arnet_core::Scalar;
use arnet_core::backend::ExecPolicy;
use arnet_tensor::{DenseStorage, DenseTensor, OpsFor, Tensor};

use crate::contract::contract_with_policy_dense;
use crate::eigen::{EigResult, EighResult, eig_with_policy_dense, eigh_with_policy_dense};
use crate::error::LinalgError;
use crate::solve::solve_with_policy_dense;
use crate::transpose::transpose_inner;
use crate::{LinalgDecompose, TruncSvdParams};

/// Axis permutation with an explicit backend and caller-specified execution
/// policy.
///
/// Expert-layer counterpart of [`crate::permute_with_backend`]; that entry
/// point consults `backend.par_for_transpose`, while this one takes `policy`
/// directly. The backend is supplied at the call site and the tensor's own
/// backend is never consulted.
pub fn permute<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    perm: &[usize],
    policy: ExecPolicy,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = transpose_inner(backend, tensor.data(), perm, false, policy)?;
    Ok(DenseTensor::from_data(result))
}

/// Pure tensor contraction with an explicit backend and caller-specified
/// execution policy for the main GEMM.
///
/// `policy` overrides the main GEMM's `ExecPolicy` only; internal transposes
/// self-tune via `backend.par_for_transpose`. This is a per-kernel override,
/// not a scope-wide thread budget — `Sequential` does not force the whole
/// contraction sequential.
///
/// The backend is supplied at the call site and neither operand's own backend
/// is consulted. Output is returned in `backend.preferred_order()`, consistent
/// with the decomposition functions.
///
/// Expert-layer counterpart of [`crate::contract_with_backend`].
pub fn contract<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    lhs: &DenseTensor<T>,
    rhs: &DenseTensor<T>,
    notation: &str,
    policy: ExecPolicy,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = contract_with_policy_dense(backend, lhs.data(), rhs.data(), notation, policy)?;
    Ok(DenseTensor::from_data(result))
}

/// Linear solve with an explicit backend and caller-specified execution
/// policy.
///
/// Expert-layer counterpart of [`crate::solve_with_backend`]; that entry point
/// consults `backend.par_for_solve`, while this one takes `policy` directly.
/// The backend is supplied at the call site and neither operand's own backend
/// is consulted.
pub fn solve<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    a: &DenseTensor<T>,
    b: &DenseTensor<T>,
    nrow_a: usize,
    policy: ExecPolicy,
) -> Result<DenseTensor<T>, LinalgError> {
    let result = solve_with_policy_dense(backend, a.data(), b.data(), nrow_a, policy)?;
    Ok(DenseTensor::from_data(result))
}

/// Self-adjoint eigenvalue decomposition with an explicit backend and
/// caller-specified execution policy.
///
/// Expert-layer counterpart of [`crate::eigh_with_backend`]; that entry point
/// consults `backend.par_for_eigh`, while this one takes `policy` directly.
/// The backend is supplied at the call site and the tensor's own backend is
/// never consulted.
pub fn eigh<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<EighResult<T>, LinalgError> {
    let (w, v) = eigh_with_policy_dense(backend, tensor.data(), nrow, policy)?;
    Ok((DenseTensor::from_data(w), DenseTensor::from_data(v)))
}

/// General eigenvalue decomposition with an explicit backend and
/// caller-specified execution policy.
///
/// Expert-layer counterpart of [`crate::eig_with_backend`]; that entry point
/// consults `backend.par_for_eig`, while this one takes `policy` directly. The
/// backend is supplied at the call site and the tensor's own backend is never
/// consulted.
pub fn eig<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    tensor: &DenseTensor<T>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<EigResult<T>, LinalgError> {
    let (w, v) = eig_with_policy_dense(backend, tensor.data(), nrow, policy)?;
    Ok((DenseTensor::from_data(w), DenseTensor::from_data(v)))
}

/// Thin SVD of a tensor reshaped as a matrix, with a caller-specified
/// execution policy.
///
/// Dispatches over layout via [`LinalgDecompose`], so one call serves both
/// Dense and BlockSparse. Expert-layer counterpart of the auto-policy
/// [`crate::svd`].
pub fn svd<T, L, B>(
    backend: &B,
    t: &Tensor<L::Storage, L>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<L::SvdOutput, LinalgError>
where
    T: Scalar,
    L: LinalgDecompose<T>,
    B: OpsFor<L::Storage>,
{
    L::svd_with_policy(backend, t, nrow, policy)
}

/// Truncated SVD of a tensor reshaped as a matrix, with a caller-specified
/// execution policy.
///
/// Dispatches over layout via [`LinalgDecompose`], so one call serves both
/// Dense and BlockSparse. Expert-layer counterpart of the auto-policy
/// [`crate::trunc_svd`].
pub fn trunc_svd<T, L, B>(
    backend: &B,
    t: &Tensor<L::Storage, L>,
    nrow: usize,
    params: &TruncSvdParams,
    policy: ExecPolicy,
) -> Result<L::TruncSvdOutput, LinalgError>
where
    T: Scalar,
    L: LinalgDecompose<T>,
    B: OpsFor<L::Storage>,
{
    L::trunc_svd_with_policy(backend, t, nrow, params, policy)
}

/// Thin QR of a tensor reshaped as a matrix, with a caller-specified execution
/// policy.
///
/// Dispatches over layout via [`LinalgDecompose`], so one call serves both
/// Dense and BlockSparse. Expert-layer counterpart of the auto-policy
/// [`crate::qr`].
pub fn qr<T, L, B>(
    backend: &B,
    t: &Tensor<L::Storage, L>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<L::QrOutput, LinalgError>
where
    T: Scalar,
    L: LinalgDecompose<T>,
    B: OpsFor<L::Storage>,
{
    L::qr_with_policy(backend, t, nrow, policy)
}

/// Thin LQ of a tensor reshaped as a matrix, with a caller-specified execution
/// policy.
///
/// Dispatches over layout via [`LinalgDecompose`], so one call serves both
/// Dense and BlockSparse. Expert-layer counterpart of the auto-policy
/// [`crate::lq`].
pub fn lq<T, L, B>(
    backend: &B,
    t: &Tensor<L::Storage, L>,
    nrow: usize,
    policy: ExecPolicy,
) -> Result<L::LqOutput, LinalgError>
where
    T: Scalar,
    L: LinalgDecompose<T>,
    B: OpsFor<L::Storage>,
{
    L::lq_with_policy(backend, t, nrow, policy)
}
