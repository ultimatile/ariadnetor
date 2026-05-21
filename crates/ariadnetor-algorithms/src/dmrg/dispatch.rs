//! Layout-keyed dispatch for the 2-site DMRG sweep driver.
//!
//! [`DmrgOps`] is the per-(layout, backend) trait that lets
//! [`super::sweep::sweep_2site`] be written once over both the Dense
//! and BlockSparse paths. It mirrors the `MpsOps` pattern in
//! `arnet-mps`: each implementation is a thin delegation to the
//! existing layout-specific free functions, with no logic duplication.

use arnet::{
    BlockSparseLayout, BlockSparseStorage, ComputeBackend, DenseLayout, DenseStorage, LinalgError,
    NativeBackend, Scalar, Sector, Storage, StorageFor, Tensor, TensorLayout, TruncSvdParams,
    diagonal_scale, diagonal_scale_block_sparse,
};
use arnet_mps::{Mpo, Mps, MpsOps};

use super::env::{DmrgEnvOps, DmrgEnvs};
use super::heff::{TwoSiteStepResult, dmrg_2site_step};
use super::heff_block_sparse::{TwoSiteStepResultBlockSparse, dmrg_2site_step_block_sparse};
use super::heff_error::DmrgHeffError;
use super::solver::{DmrgScalar, LocalEigensolverParams};
use super::sweep::SweepDirection;

/// Per-step output projected to scalar diagnostics + post-S-absorb
/// site tensors + new bond dimension.
pub struct AbsorbedStep<St, L, B>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    /// Post-absorb tensor to write into MPS site `i`.
    pub site_i: Tensor<St, L, B>,
    /// Post-absorb tensor to write into MPS site `i + 1`.
    pub site_ip1: Tensor<St, L, B>,
    /// Bond dimension of the new shared bond between sites `i` and
    /// `i + 1`.
    pub bond_dim: usize,
    /// Smallest eigenvalue of `H_eff` at this step.
    pub eigenvalue: f64, // placeholder; see typed wrapper below
    pub residual: f64,
    pub trunc_err: f64,
    pub iters: usize,
    pub converged: bool,
}

/// Per-(layout, backend) dispatch trait for the 2-site DMRG sweep
/// driver.
///
/// The associated `Storage` types on the `MpsOps<T>` and `DmrgEnvOps<T>`
/// super-traits are assumed to coincide; the env subsystem and the
/// MPS subsystem share one storage flavor per layout. The trait
/// expresses that via the explicit
/// `DmrgEnvOps<T, Storage = <Self as MpsOps<T>>::Storage>` super-bound.
pub trait DmrgOps<T: Scalar, B: ComputeBackend = NativeBackend>:
    MpsOps<T> + DmrgEnvOps<T, Storage = <Self as MpsOps<T>>::Storage> + Sized
{
    /// Layout-typed step result.
    type StepResult;

    /// Build local `H_eff` and drive the chosen local eigensolver.
    fn step(
        envs: &DmrgEnvs<<Self as MpsOps<T>>::Storage, Self, B>,
        mps: &Mps<<Self as MpsOps<T>>::Storage, Self, B>,
        mpo: &Mpo<<Self as MpsOps<T>>::Storage, Self, B>,
        site: usize,
        eigensolver: &LocalEigensolverParams,
        trunc: &TruncSvdParams,
    ) -> Result<Self::StepResult, DmrgHeffError>;

    /// Consume the step result, absorb `S`, project to
    /// [`AbsorbedStep`].
    fn commit_step(
        backend: &B,
        result: Self::StepResult,
        direction: SweepDirection,
    ) -> Result<AbsorbedStep<<Self as MpsOps<T>>::Storage, Self, B>, LinalgError>;

    /// Project diagnostic scalars (the absorbed step's
    /// `eigenvalue / residual / trunc_err` typed as `T::Real`).
    /// Required because [`AbsorbedStep`] holds them as `f64`
    /// placeholders to keep the struct layout backend-agnostic; the
    /// sweep driver pairs the typed scalars with the absorbed
    /// tensors via this projection.
    fn step_scalars(result: &Self::StepResult) -> (T::Real, T::Real, T::Real, usize, bool);
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T, B> DmrgOps<T, B> for DenseLayout
where
    T: DmrgScalar,
    T::Real: Scalar<Real = T::Real>,
    B: ComputeBackend,
{
    type StepResult = TwoSiteStepResult<T, B>;

    fn step(
        envs: &DmrgEnvs<DenseStorage<T>, Self, B>,
        mps: &Mps<DenseStorage<T>, Self, B>,
        mpo: &Mpo<DenseStorage<T>, Self, B>,
        site: usize,
        eigensolver: &LocalEigensolverParams,
        trunc: &TruncSvdParams,
    ) -> Result<Self::StepResult, DmrgHeffError> {
        dmrg_2site_step(envs, mps, mpo, site, eigensolver, trunc)
    }

    fn commit_step(
        _backend: &B,
        result: Self::StepResult,
        direction: SweepDirection,
    ) -> Result<AbsorbedStep<DenseStorage<T>, Self, B>, LinalgError> {
        let bond_dim = result.s.shape()[0];
        // Materialize the singular-value vector into the scalar type
        // expected by `diagonal_scale`. The `s` tensor's data slice
        // is the descending vector ready to consume.
        let s_vec: Vec<T::Real> = result.s.data_slice().to_vec();
        let (site_i, site_ip1) = match direction {
            SweepDirection::LeftToRight => {
                let s_vt = diagonal_scale(&result.vt, &s_vec, 0)?;
                (result.u, s_vt)
            }
            SweepDirection::RightToLeft => {
                let u_s = diagonal_scale(&result.u, &s_vec, 2)?;
                (u_s, result.vt)
            }
        };
        Ok(AbsorbedStep {
            site_i,
            site_ip1,
            bond_dim,
            eigenvalue: 0.0,
            residual: 0.0,
            trunc_err: 0.0,
            iters: 0,
            converged: false,
        })
    }

    fn step_scalars(result: &Self::StepResult) -> (T::Real, T::Real, T::Real, usize, bool) {
        (
            result.eigenvalue,
            result.residual,
            result.trunc_err,
            result.iters,
            result.converged,
        )
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T, S, B> DmrgOps<T, B> for BlockSparseLayout<S>
where
    T: DmrgScalar,
    T::Real: Scalar<Real = T::Real>,
    S: Sector,
    B: ComputeBackend,
{
    type StepResult = TwoSiteStepResultBlockSparse<T, S, B>;

    fn step(
        envs: &DmrgEnvs<BlockSparseStorage<T>, Self, B>,
        mps: &Mps<BlockSparseStorage<T>, Self, B>,
        mpo: &Mpo<BlockSparseStorage<T>, Self, B>,
        site: usize,
        eigensolver: &LocalEigensolverParams,
        trunc: &TruncSvdParams,
    ) -> Result<Self::StepResult, DmrgHeffError> {
        dmrg_2site_step_block_sparse(envs, mps, mpo, site, eigensolver, trunc)
    }

    fn commit_step(
        _backend: &B,
        result: Self::StepResult,
        direction: SweepDirection,
    ) -> Result<AbsorbedStep<BlockSparseStorage<T>, Self, B>, LinalgError> {
        // Total post-truncation singular values across sectors — the
        // conventional U(1) MPS bond dimension.
        let bond_dim: usize = result.s.values.iter().map(|(_, v)| v.len()).sum();
        let (site_i, site_ip1) = match direction {
            SweepDirection::LeftToRight => {
                let s_vt = diagonal_scale_block_sparse(&result.vt, &result.s, 0)?;
                (result.u, s_vt)
            }
            SweepDirection::RightToLeft => {
                let u_s = diagonal_scale_block_sparse(&result.u, &result.s, 2)?;
                (u_s, result.vt)
            }
        };
        Ok(AbsorbedStep {
            site_i,
            site_ip1,
            bond_dim,
            eigenvalue: 0.0,
            residual: 0.0,
            trunc_err: 0.0,
            iters: 0,
            converged: false,
        })
    }

    fn step_scalars(result: &Self::StepResult) -> (T::Real, T::Real, T::Real, usize, bool) {
        (
            result.eigenvalue,
            result.residual,
            result.trunc_err,
            result.iters,
            result.converged,
        )
    }
}
