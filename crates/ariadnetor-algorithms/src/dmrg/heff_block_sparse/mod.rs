//! BlockSparse / U(1) variant of the 2-site DMRG local update.
//!
//! Mirrors [`super::heff`] (the Dense path) for a
//! `BlockSparseTensorData<T, S>`-backed chain. The effective
//! Hamiltonian built from `(left(i), W[i], W[i+1], right(i+2))` is
//! exposed as a [`crate::krylov::LinearOp<T>`] so the existing Dense
//! Krylov solvers (Lanczos by default, ARPACK behind the `arpack`
//! feature) drive it without a separate native BlockSparse Krylov
//! path.
//!
//! ## Flat-buffer adapter
//!
//! `LinearOp<T>` operates on `DenseTensor<T, B>` flat vectors. The
//! BlockSparse Heff implements `apply(&DenseTensor<T, B>) -> DenseTensor<T, B>` via:
//!
//! 1. **Scatter** the flat input into a populated
//!    `BlockSparseTensorData` 2-site tensor whose indices and flux
//!    match the psi template derived from the MPS sites at
//!    `(site, site+1)`.
//! 2. **Contract** through the env / W tensors using
//!    [`arnet::contract_block_sparse_with_backend`] in four steps. The
//!    axis convention mirrors `arnet_mps::inner::braket_bsp` and
//!    the Phase 6.1 `extend_*_step` kernels; the natural output
//!    order `lhs_free | rhs_free` ends in
//!    `[chi_l, d_i, d_{i+1}, chi_r]`, matching the input shape with
//!    no axis swap.
//! 3. **Gather** the rank-4 result back into a flat `DenseTensor<T, B>`
//!    of the same length, walking the psi template's
//!    [`BlockSparseTensorData::block_metas`] and looking up each block
//!    in the contracted output by coordinate.
//!
//! Symmetry preservation is structural: the psi template only
//! allocates flux-allowed blocks, and `contract_block_sparse`
//! propagates flux as `lhs.flux().fuse(rhs.flux())`. With env /
//! MPO fluxes pre-validated to identity at the entry point, the
//! matvec output's flux equals `psi.flux()` by construction.

mod operator;
mod validation;

pub use operator::EffectiveHamiltonian2SiteBlockSparse;

use std::sync::Arc;

use arnet::{
    BlockSingularValues, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, ComputeBackend,
    NativeBackend, Scalar, Sector, TruncSvdParams, trunc_svd_block_sparse_with_backend,
};
use arnet_mps::{Mpo, Mps};

#[cfg(feature = "arpack")]
use crate::krylov::arpack_smallest;
use crate::krylov::lanczos_smallest;

use super::env::DmrgEnvs;
use super::heff_error::DmrgHeffError;
use super::solver::{DmrgScalar, LocalEigensolverParams};

/// Result of a single BlockSparse 2-site DMRG step: smallest local
/// eigenpair plus the truncated-SVD split of its eigenvector.
///
/// `u`, `s`, `vt` are returned **separately** so the caller (the
/// BlockSparse sweep driver
/// [`super::sweep_2site`]) decides which side
/// absorbs `S`. Mirrors [`super::heff::TwoSiteStepResult`] for the
/// Dense path.
///
/// `Debug` is not derived because `BlockSparse: !Debug`; tests that
/// need to inspect the result destructure its fields directly.
pub struct TwoSiteStepResultBlockSparse<T: Scalar, S: Sector, B: ComputeBackend = NativeBackend> {
    pub eigenvalue: T::Real,
    pub residual: T::Real,
    pub iters: usize,
    /// `true` iff the local eigensolver succeeded.
    pub converged: bool,
    /// Left singular vectors. Legs `[chi_l, d_i, bond(In)]`,
    /// `flux = identity()`. Left-canonical at axes `(chi_l, d_i)`.
    pub u: BlockSparseTensor<T, S, B>,
    /// Singular values per fused sector (descending within each
    /// sector).
    pub s: BlockSingularValues<<T as Scalar>::Real, S>,
    /// Right singular vectors. Legs `[bond(Out), d_{i+1}, chi_r]`,
    /// `flux = psi_flux`. Right-canonical at axes `(d_{i+1}, chi_r)`.
    pub vt: BlockSparseTensor<T, S, B>,
    /// Frobenius norm of the discarded singular values.
    pub trunc_err: T::Real,
}

/// Run a single 2-site DMRG step at sites `(site, site+1)` on a
/// BlockSparse-backed chain. Mirrors [`super::heff::dmrg_2site_step`]
/// for the BlockSparse / U(1) path.
///
/// Builds the local effective Hamiltonian, drives the selected
/// local eigensolver (per [`LocalEigensolverParams`] — Lanczos by
/// default, ARPACK behind the `arpack` feature) via the flat-buffer
/// adapter, then splits the optimized two-site block via
/// [`trunc_svd_block_sparse_with_backend`] with `nrow = 2`. Caller
/// advances envs separately.
///
/// # Errors
///
/// - [`DmrgHeffError::InvalidSite`], [`DmrgHeffError::LengthMismatch`],
///   [`DmrgHeffError::StaleEnv`], [`DmrgHeffError::ShapeMismatch`],
///   [`DmrgHeffError::InvalidEigensolverParams`],
///   [`DmrgHeffError::Contract`] — same semantics as
///   [`super::heff::dmrg_2site_step`].
/// - [`DmrgHeffError::QnMismatch`] — BlockSparse-specific QN /
///   Direction / sector / per-site-flux compatibility check
///   failed. The matvec body's `.expect` calls cannot fire on user
///   input because every contract pair, MPO well-formedness
///   property, env-template-compatibility property, and identity-flux
///   precondition is validated up front.
pub fn dmrg_2site_step_block_sparse<T, S, B>(
    envs: &DmrgEnvs<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    mps: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    mpo: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    site: usize,
    eigensolver: &LocalEigensolverParams,
    trunc: &TruncSvdParams,
) -> Result<TwoSiteStepResultBlockSparse<T, S, B>, DmrgHeffError>
where
    T: DmrgScalar,
    T::Real: Scalar<Real = T::Real>,
    S: Sector,
    B: ComputeBackend,
{
    let v = validation::validate_inputs(envs, mps, mpo, site, eigensolver)?;

    let heff = EffectiveHamiltonian2SiteBlockSparse::new(
        v.left,
        v.w_i,
        v.w_ip1,
        v.right,
        v.mps_i,
        v.mps_ip1,
        Arc::clone(&v.backend),
    )?;
    let dim = heff.dim();
    if dim == 0 {
        // Per-axis `total_dim() >= 1` checks in `validate_inputs`
        // ensure individual outer axes are non-empty, but the
        // combined `psi_flux = mps_i.flux ⊕ mps_ip1.flux` may
        // still be unreachable on the (axis_0 × axis_1 × axis_1
        // × axis_2) sector lattice — in which case the
        // `BlockSparseTensorData::zeros(...)` template allocates
        // zero blocks. Without this check the underlying solver's
        // `dim >= 1` precondition fires (Lanczos panics, ARPACK
        // rejects with `InvalidParam`) on otherwise valid user
        // input. Doing the check here (post `new`) avoids a second
        // template allocation that a validation-time check would
        // have required.
        return Err(DmrgHeffError::QnMismatch {
            site,
            field: "psi_template",
            detail: format!(
                "no flux-allowed (q_l, q_p, q_p, q_r) tuple satisfies psi_flux = {:?} \
                 given MPS[i].axis 0 = {:?}, MPS[i].axis 1 = {:?}, \
                 MPS[i+1].axis 1 = {:?}, MPS[i+1].axis 2 = {:?}",
                heff.psi_flux(),
                v.mps_i.indices()[0].blocks(),
                v.mps_i.indices()[1].blocks(),
                v.mps_ip1.indices()[1].blocks(),
                v.mps_ip1.indices()[2].blocks(),
            ),
        });
    }
    let (eigenvalue, eigenvector, iters, converged, residual) = match eigensolver {
        LocalEigensolverParams::Lanczos(p) => {
            let lan = lanczos_smallest::<T, _>(&heff, dim, p);
            (
                lan.eigenvalue,
                lan.eigenvector,
                lan.iters,
                lan.converged,
                lan.residual,
            )
        }
        #[cfg(feature = "arpack")]
        LocalEigensolverParams::Arpack(p) => {
            let res = arpack_smallest::<T, _>(&heff, dim, p)?;
            // See `super::heff::dmrg_2site_step` ARPACK arm for the
            // rationale: the step-level converged flag tracks
            // ARPACK's relative-tol stopping (Ok return), not the
            // absolute-tol divergence indicator that ARPACK exposes
            // as `ArpackResult.converged`.
            (
                res.eigenvalue,
                res.eigenvector,
                res.iters,
                true,
                res.residual,
            )
        }
    };

    let psi_4d = operator::scatter_flat_to_template(
        eigenvector.data_slice(),
        &heff.psi_template,
        &heff.block_offsets,
        &heff.block_coords,
    );
    let (u, s, vt, trunc_err) = trunc_svd_block_sparse_with_backend(&v.backend, &psi_4d, 2, trunc)?;

    Ok(TwoSiteStepResultBlockSparse {
        eigenvalue,
        residual,
        iters,
        converged,
        u,
        s,
        vt,
        trunc_err,
    })
}
