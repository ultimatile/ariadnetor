//! Chain-keyed dispatch for the 2-site DMRG sweep driver.
//!
//! [`DmrgOps`] is keyed on the [`Mps`] chain (sharing its [`MpsOps`]
//! supertrait so the storage / layout taxa come from one place) and lets
//! [`super::sweep::sweep_2site`] be written once over both the Dense and
//! BlockSparse paths. Each implementation is a thin delegation to the
//! existing storage-specific kernels, with no logic duplication. Sealed
//! transitively through `MpsOps` (itself sealed), so it cannot be
//! implemented downstream.
//!
//! The per-storage step result (`TwoSiteStepResult` /
//! `TwoSiteStepResultBlockSparse`) diverges in its `s` field and cannot
//! collapse to one chain-agnostic struct. Rather than expose it as a `pub`
//! associated type, [`DmrgOps::full_step_k`] runs the whole 2-site step
//! (solve, project diagnostics, absorb `S`) in one call, keeping the step
//! result an impl-internal local and returning only public types.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{LinalgError, TruncSvdParams, diagonal_scale};
use ariadnetor_mps::{Mpo, Mps, MpsOps};
use ariadnetor_tensor::{
    BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, Host, Sector, Storage,
    StorageFor, Tensor, TensorLayout,
};

use super::env::DmrgEnvs;
use super::heff::dmrg_2site_step;
use super::heff_block_sparse::dmrg_2site_step_block_sparse;
use super::heff_error::DmrgHeffError;
use super::solver::{DmrgScalar, LocalEigensolverParams};
use super::sweep::SweepDirection;

/// Post-absorb site tensors + new bond dimension returned by
/// [`DmrgOps::full_step_k`], paired there with the diagnostic scalars.
///
/// The matching diagnostics (eigenvalue / residual / trunc-err / iters /
/// converged) are typed on `T::Real` and returned alongside this struct
/// rather than held on it; folding them in would force a third scalar
/// parameter, so the struct stays keyed on `(St, L)` only.
pub struct AbsorbedStep<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Post-absorb tensor to write into MPS site `i`.
    pub site_i: Tensor<St, L>,
    /// Post-absorb tensor to write into MPS site `i + 1`.
    pub site_ip1: Tensor<St, L>,
    /// Bond dimension of the new shared bond between sites `i` and
    /// `i + 1`. For BlockSparse / U(1), summed over retained sectors.
    pub bond_dim: usize,
}

/// Diagnostic scalars projected from a 2-site step, in the order
/// `(eigenvalue, residual, trunc_err, iters, converged)`.
type StepDiagnostics<R> = (R, R, R, usize, bool);

/// Successful / failed outcome of [`DmrgOps::full_step_k`]: the absorbed
/// site tensors plus diagnostics, or a [`FullStepError`].
type FullStepOutput<St, L, R> = Result<(AbsorbedStep<St, L>, StepDiagnostics<R>), FullStepError>;

/// Error raised by [`DmrgOps::full_step_k`], distinguishing the local
/// eigensolve failure from the post-solve S-absorb failure so the sweep
/// driver can keep its `Step` / `Scale` breadcrumbs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FullStepError {
    /// The local effective-Hamiltonian solve (`dmrg_2site_step` /
    /// `dmrg_2site_step_block_sparse`) failed.
    #[error("DMRG local 2-site step failed")]
    Heff(#[source] DmrgHeffError),
    /// The post-solve S-absorb (`diagonal_scale`) failed.
    #[error("DMRG S-absorb (diagonal scale) failed")]
    Scale(#[source] LinalgError),
}

/// Chain-keyed dispatch trait for the 2-site DMRG sweep driver.
///
/// Keyed on the [`Mps`] chain with [`MpsOps<T>`] as supertrait (same
/// `Self`), so the storage / layout taxa are the chain's own via
/// [`MpsOps::Storage`] / [`MpsOps::Layout`]. The env subsystem shares that
/// storage flavor; the sweep driver binds [`super::env::DmrgEnvOps`] on
/// the matching [`DmrgEnvs`] chain at its call site, so the
/// storage-coincidence the former layout-keyed super-bound expressed is
/// now carried by the two chains sharing `(St, L)`.
pub trait DmrgOps<T: Scalar>: MpsOps<T> {
    /// Build local `H_eff`, drive the local eigensolver, project the
    /// `T::Real` diagnostics, and absorb `S` — fused into one step
    /// returning only public types. The order is preserved: solve →
    /// project diagnostics → absorb, so the scalars are read before the
    /// step result is consumed.
    ///
    /// Host-pinned: the explicit-backend scaling path runs on
    /// [`Host::shared`], so the method takes no backend at the call site.
    /// DMRG is host-pinned in the CPU-only Stage B scope.
    fn full_step_k(
        &self,
        envs: &DmrgEnvs<<Self as MpsOps<T>>::Storage, <Self as MpsOps<T>>::Layout>,
        mpo: &Mpo<<Self as MpsOps<T>>::Storage, <Self as MpsOps<T>>::Layout>,
        site: usize,
        eigensolver: &LocalEigensolverParams,
        trunc: &TruncSvdParams,
        direction: SweepDirection,
    ) -> FullStepOutput<<Self as MpsOps<T>>::Storage, <Self as MpsOps<T>>::Layout, T::Real>;
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T> DmrgOps<T> for Mps<DenseStorage<T>, DenseLayout>
where
    T: DmrgScalar,
    T::Real: Scalar<Real = T::Real>,
{
    fn full_step_k(
        &self,
        envs: &DmrgEnvs<DenseStorage<T>, DenseLayout>,
        mpo: &Mpo<DenseStorage<T>, DenseLayout>,
        site: usize,
        eigensolver: &LocalEigensolverParams,
        trunc: &TruncSvdParams,
        direction: SweepDirection,
    ) -> FullStepOutput<DenseStorage<T>, DenseLayout, T::Real> {
        let result = dmrg_2site_step(envs, self, mpo, site, eigensolver, trunc)
            .map_err(FullStepError::Heff)?;
        // Project diagnostics before the result is consumed by the absorb.
        let diagnostics = (
            result.eigenvalue,
            result.residual,
            result.trunc_err,
            result.iters,
            result.converged,
        );
        let backend = Host::shared();
        let bond_dim = result.s.shape()[0];
        let (site_i, site_ip1) = match direction {
            SweepDirection::LeftToRight => {
                let s_vt = diagonal_scale(backend.as_ref(), &result.vt, result.s.data_slice(), 0)
                    .map_err(FullStepError::Scale)?;
                (result.u, s_vt)
            }
            SweepDirection::RightToLeft => {
                let u_s = diagonal_scale(backend.as_ref(), &result.u, result.s.data_slice(), 2)
                    .map_err(FullStepError::Scale)?;
                (u_s, result.vt)
            }
        };
        Ok((
            AbsorbedStep {
                site_i,
                site_ip1,
                bond_dim,
            },
            diagnostics,
        ))
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T, S> DmrgOps<T> for Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    T: DmrgScalar,
    T::Real: Scalar<Real = T::Real>,
    S: Sector,
{
    fn full_step_k(
        &self,
        envs: &DmrgEnvs<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        mpo: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        site: usize,
        eigensolver: &LocalEigensolverParams,
        trunc: &TruncSvdParams,
        direction: SweepDirection,
    ) -> FullStepOutput<BlockSparseStorage<T>, BlockSparseLayout<S>, T::Real> {
        let result = dmrg_2site_step_block_sparse(envs, self, mpo, site, eigensolver, trunc)
            .map_err(FullStepError::Heff)?;
        let diagnostics = (
            result.eigenvalue,
            result.residual,
            result.trunc_err,
            result.iters,
            result.converged,
        );
        let backend = Host::shared();
        // Total post-truncation singular values across sectors — the
        // conventional U(1) MPS bond dimension.
        let bond_dim: usize = result.s.values.iter().map(|(_, v)| v.len()).sum();
        let (site_i, site_ip1) = match direction {
            SweepDirection::LeftToRight => {
                let s_vt = diagonal_scale(backend.as_ref(), &result.vt, &result.s, 0)
                    .map_err(FullStepError::Scale)?;
                (result.u, s_vt)
            }
            SweepDirection::RightToLeft => {
                let u_s = diagonal_scale(backend.as_ref(), &result.u, &result.s, 2)
                    .map_err(FullStepError::Scale)?;
                (u_s, result.vt)
            }
        };
        Ok((
            AbsorbedStep {
                site_i,
                site_ip1,
                bond_dim,
            },
            diagnostics,
        ))
    }
}
