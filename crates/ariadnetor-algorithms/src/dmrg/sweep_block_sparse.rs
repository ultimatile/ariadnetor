//! BlockSparse / U(1) variant of the 2-site DMRG sweep driver.
//!
//! Mirrors [`super::sweep::dmrg_2site_sweep`] for a
//! `BlockSparse<T, S>`-backed chain. The inner step delegates to
//! [`super::heff_block_sparse::dmrg_2site_step_block_sparse`], which
//! drives Lanczos through a flat-buffer adapter. Singular-value
//! absorb is via [`arnet_linalg::diagonal_scale_block_sparse`] —
//! per-sector scaling that preserves block structure and flux.
//!
//! Sector preservation: the BlockSparse step returns `U` with
//! identity flux and `Vt` with `psi_flux = mps[i].flux ⊕
//! mps[i+1].flux`. The S-absorb is a per-block real scale that does
//! not touch flux, so the chain's total flux is preserved across
//! every step, regardless of sweep direction.

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::{LinalgError, diagonal_scale_block_sparse};
use arnet_mps::{CanonicalForm, Mpo, Mps, TensorChain, braket, norm};
use arnet_tensor::{BlockSparse, Sector};
use num_traits::Zero;

use crate::numeric::try_real_from_f64;

use super::env::DmrgEnvs;
use super::heff::DmrgHeffError;
use super::heff_block_sparse::dmrg_2site_step_block_sparse;
use super::sweep::{
    DmrgResult, DmrgStepRecord, DmrgSweepError, DmrgSweepParams, DmrgSweepRecord, SweepDirection,
    validate_params,
};

/// Run alternating L→R / R→L sweeps on a BlockSparse chain until
/// convergence or `max_sweeps`. Mutates `mps` and `envs` in place;
/// the final MPS state is `CanonicalForm::Mixed { center: 0 }`
/// (R→L runs last).
///
/// The input MPS must be in `CanonicalForm::Right` or
/// `CanonicalForm::Mixed { center: 0 }`. Other forms are rejected
/// because the local effective-Hamiltonian eigenvalue equation
/// returns physical energy directly only when, with active block
/// `(i, i+1)`, sites `(0..i)` are left-canonical and sites
/// `(i+2..N-1)` are right-canonical. The driver starts L→R from
/// `i = 0`, so the binding precondition is right-canonicality of
/// `(2..N-1)`, met exactly by `Right` and `Mixed { center: 0 }`.
/// The driver does not auto-canonicalize — doing so would silently
/// invalidate the caller-supplied `envs`.
pub fn dmrg_2site_sweep_block_sparse<T, S, B>(
    envs: &mut DmrgEnvs<BlockSparse<T, S>, B>,
    mps: &mut Mps<BlockSparse<T, S>, B>,
    mpo: &Mpo<BlockSparse<T, S>, B>,
    params: &DmrgSweepParams,
) -> Result<DmrgResult<T::Real>, DmrgSweepError>
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
    S: Sector,
    B: ComputeBackend,
{
    // ---- Length / size validation -------------------------------
    let n_sites = envs.n_sites();
    if mps.len() != n_sites || mpo.len() != n_sites {
        return Err(DmrgSweepError::LengthMismatch {
            mps: mps.len(),
            mpo: mpo.len(),
            envs: n_sites,
        });
    }
    if n_sites < 2 {
        return Err(DmrgSweepError::TooFewSites { n_sites });
    }

    // ---- Param validation ---------------------------------------
    validate_params(params)?;
    // Casts may fail when `T::Real == f32` and the user supplied a
    // finite value outside f32 range. Same gate as the Dense path.
    let energy_tol_real: T::Real =
        try_real_from_f64::<T>(params.energy_tol).ok_or(DmrgSweepError::InvalidParams {
            detail: "energy_tol is not representable in T::Real",
        })?;
    if try_real_from_f64::<T>(params.lanczos.tol).is_none() {
        return Err(DmrgSweepError::InvalidParams {
            detail: "lanczos.tol is not representable in T::Real",
        });
    }

    // ---- Canonical-form contract --------------------------------
    match mps.canonical_form() {
        CanonicalForm::Right => {}
        CanonicalForm::Mixed { center: 0 } => {}
        other => {
            return Err(DmrgSweepError::MpsNotRightCanonical {
                found: other.clone(),
            });
        }
    }

    let backend: Arc<B> = mps.backend_arc().clone();
    let mut sweeps: Vec<DmrgSweepRecord<T::Real>> = Vec::with_capacity(params.max_sweeps);
    let mut last_energy: Option<T::Real> = None;
    let mut converged = false;
    let mut completed_sweeps = 0usize;

    // ---- Main sweep loop ----------------------------------------
    for sweep_idx in 0..params.max_sweeps {
        let mut steps: Vec<DmrgStepRecord<T::Real>> = Vec::with_capacity(2 * (n_sites - 1));

        // L→R half-sweep.
        for site in 0..n_sites - 1 {
            let record = run_step_bsp(
                envs,
                mps,
                mpo,
                site,
                params,
                sweep_idx,
                SweepDirection::LeftToRight,
                &backend,
            )?;
            steps.push(record);

            // Skip the trailing advance: site = n_sites - 2 is the
            // last L→R step; the next R→L step at the same `site`
            // consumes `left(site)` which is still valid.
            if site < n_sites - 2 {
                envs.advance_left(mps, mpo, site)
                    .map_err(|source| DmrgSweepError::Env {
                        sweep: sweep_idx,
                        direction: SweepDirection::LeftToRight,
                        site,
                        source,
                    })?;
            }
        }

        // R→L half-sweep.
        for site in (0..n_sites - 1).rev() {
            let record = run_step_bsp(
                envs,
                mps,
                mpo,
                site,
                params,
                sweep_idx,
                SweepDirection::RightToLeft,
                &backend,
            )?;
            steps.push(record);

            // Always advance through the trailing `site == 0`
            // boundary so `right[1]` and `left[1]` stay coherent
            // with the freshly-mutated MPS sites — same staleness
            // contract as the Dense path.
            envs.advance_right(mps, mpo, site + 1)
                .map_err(|source| DmrgSweepError::Env {
                    sweep: sweep_idx,
                    direction: SweepDirection::RightToLeft,
                    site,
                    source,
                })?;
        }

        // R→L ends with the orthogonality center at site 0; storage_mut
        // reset the form to Unknown along the way, so re-pin it here.
        mps.set_canonical_form(CanonicalForm::Mixed { center: 0 });

        // ---- Post-sweep diagnostics -----------------------------
        let bra_h_ket = braket(mps, mpo, mps);
        let nrm = norm(mps);
        let nrm_sq: T::Real = nrm * nrm;
        let sweep_energy: T::Real = bra_h_ket.re() / nrm_sq;

        let max_bond = mps.max_bond_dim();
        let mut min_eig = steps[0].eigenvalue;
        let mut max_te = steps[0].trunc_err;
        let mut all_ok = true;
        for s in &steps {
            if s.eigenvalue < min_eig {
                min_eig = s.eigenvalue;
            }
            if s.trunc_err > max_te {
                max_te = s.trunc_err;
            }
            if !s.lanczos_converged {
                all_ok = false;
            }
        }

        sweeps.push(DmrgSweepRecord {
            sweep: sweep_idx,
            sweep_energy,
            min_step_eigenvalue: min_eig,
            max_trunc_err: max_te,
            max_bond,
            all_lanczos_converged: all_ok,
            steps,
        });

        completed_sweeps = sweep_idx + 1;

        // ---- Convergence check ----------------------------------
        if completed_sweeps >= params.min_sweeps
            && let Some(prev) = last_energy
        {
            let delta = sweep_energy - prev;
            let abs_delta = if delta < T::Real::zero() {
                -delta
            } else {
                delta
            };
            if abs_delta <= energy_tol_real && all_ok {
                converged = true;
                break;
            }
        }
        last_energy = Some(sweep_energy);
    }

    let final_energy = sweeps
        .last()
        .map(|s| s.sweep_energy)
        .expect("at least one sweep ran (max_sweeps >= 1 by validation)");

    Ok(DmrgResult {
        energy: final_energy,
        converged,
        n_sweeps: completed_sweeps,
        sweeps,
    })
}

/// Run a single 2-site step at `site` on a BlockSparse chain, then
/// mutate the MPS site tensors at `site` and `site + 1` according to
/// `direction`. Mirror of the Dense `run_step` helper.
#[allow(clippy::too_many_arguments)]
fn run_step_bsp<T, S, B>(
    envs: &DmrgEnvs<BlockSparse<T, S>, B>,
    mps: &mut Mps<BlockSparse<T, S>, B>,
    mpo: &Mpo<BlockSparse<T, S>, B>,
    site: usize,
    params: &DmrgSweepParams,
    sweep_idx: usize,
    direction: SweepDirection,
    backend: &Arc<B>,
) -> Result<DmrgStepRecord<T::Real>, DmrgSweepError>
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
    S: Sector,
    B: ComputeBackend,
{
    let result = dmrg_2site_step_block_sparse(envs, mps, mpo, site, &params.lanczos, &params.trunc)
        .map_err(|source: DmrgHeffError| DmrgSweepError::Step {
            sweep: sweep_idx,
            direction,
            site,
            source,
        })?;

    // Total post-truncation singular values across sectors — the
    // conventional U(1) MPS bond dim.
    let bond_dim: usize = result.s.values.iter().map(|(_, v)| v.len()).sum();

    let scale_err = |source: LinalgError| DmrgSweepError::Scale {
        sweep: sweep_idx,
        direction,
        site,
        source,
    };

    match direction {
        SweepDirection::LeftToRight => {
            // site i ← U (left-isometric per fused sector)
            // site i+1 ← S·Vt (axis 0 = bond(Out), carries S right)
            let s_vt = diagonal_scale_block_sparse(&**backend, &result.vt, &result.s, 0)
                .map_err(scale_err)?;
            *mps.storage_mut(site) = result.u;
            *mps.storage_mut(site + 1) = s_vt;
        }
        SweepDirection::RightToLeft => {
            // site i   ← U·S  (axis 2 = bond(In), carries S left)
            // site i+1 ← Vt   (right-isometric per fused sector)
            let u_s = diagonal_scale_block_sparse(&**backend, &result.u, &result.s, 2)
                .map_err(scale_err)?;
            *mps.storage_mut(site) = u_s;
            *mps.storage_mut(site + 1) = result.vt;
        }
    }

    Ok(DmrgStepRecord {
        sweep: sweep_idx,
        direction,
        site,
        eigenvalue: result.eigenvalue,
        residual: result.residual,
        trunc_err: result.trunc_err,
        bond_dim,
        lanczos_iters: result.iters,
        lanczos_converged: result.converged,
    })
}
