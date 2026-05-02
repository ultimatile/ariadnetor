//! 2-site DMRG sweep driver.
//!
//! Runs alternating L→R / R→L half-sweeps on top of [`super::heff::dmrg_2site_step`],
//! mutating the MPS site tensors and the [`DmrgEnvs`] in place. Caller
//! supplies a right-canonical (or `Mixed { center: 0 }`) MPS plus a
//! freshly-built `DmrgEnvs`; the driver does **not** auto-canonicalize
//! because doing so would silently invalidate the caller-supplied envs.
//!
//! # Convergence
//!
//! After each full L→R + R→L cycle we record the **normalized**
//! post-truncation expectation `<psi|H|psi>.re() / <psi|psi>`.
//! `trunc_svd` keeps unrenormalized singular values, so without
//! the `<psi|psi>` divisor the sweep energy drifts toward zero
//! whenever truncation happens — the divisor strips that
//! norm-artifact away. Convergence requires energy delta within
//! `energy_tol`, every step's Lanczos converged, and
//! `n_sweeps >= min_sweeps`.

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::{LinalgError, diagonal_scale};
use arnet_mps::{CanonicalForm, Mpo, Mps, TensorChain, braket, norm};
use arnet_tensor::Dense;
use num_traits::Zero;

use crate::krylov::LanczosParams;
use crate::numeric::try_real_from_f64;

use super::env::{DmrgEnvError, DmrgEnvs};
use super::heff::{DmrgHeffError, dmrg_2site_step};

/// Direction of a half-sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepDirection {
    /// Steps from `site = 0` to `site = n_sites - 2`. Each step's
    /// SVD splits the optimized 2-site block into a left-isometric
    /// site `i` and an `S·Vt`-carrying site `i + 1`.
    LeftToRight,
    /// Steps from `site = n_sites - 2` down to `site = 0`. Each
    /// step's SVD splits into a `U·S`-carrying site `i` and a
    /// right-isometric site `i + 1`.
    RightToLeft,
}

/// Caller-supplied parameters for both
/// [`dmrg_2site_sweep`] (Dense) and
/// [`super::sweep_block_sparse::dmrg_2site_sweep_block_sparse`]
/// (BlockSparse / U(1)).
///
/// Stored as plain `f64` for `energy_tol`; the entry point converts
/// it to `T::Real` via the same `NumCast::from(f64)` idiom as
/// [`LanczosParams::tol`]. This keeps the params type non-generic
/// across `T`.
#[derive(Debug, Clone)]
pub struct DmrgSweepParams {
    /// Maximum number of full L→R + R→L cycles. Must be `>= 1`;
    /// `0` is rejected with [`DmrgSweepError::InvalidParams`].
    pub max_sweeps: usize,
    /// Minimum number of sweeps before the energy-delta convergence
    /// test is honored. Pre-`min_sweeps` cycles always continue
    /// regardless of energy delta. Must be `<= max_sweeps`.
    pub min_sweeps: usize,
    /// Energy-delta tolerance. After cycle `n >= min_sweeps`,
    /// convergence requires `|E_n - E_{n-1}| <= energy_tol`. Must
    /// be finite and non-negative.
    pub energy_tol: f64,
    /// Local-solver parameters, forwarded to the per-step driver
    /// (Dense `dmrg_2site_step` or BlockSparse
    /// `dmrg_2site_step_block_sparse`).
    pub lanczos: LanczosParams,
    /// Truncated-SVD parameters, forwarded to the per-step driver
    /// (Dense `dmrg_2site_step` or BlockSparse
    /// `dmrg_2site_step_block_sparse`).
    pub trunc: arnet_linalg::TruncSvdParams,
}

/// Per-step diagnostics record.
#[derive(Debug, Clone)]
pub struct DmrgStepRecord<R> {
    pub sweep: usize,
    pub direction: SweepDirection,
    pub site: usize,
    /// Smallest eigenvalue of `H_eff` at this step (pre-truncation
    /// local-block variational minimum). May lie below the
    /// post-truncation sweep energy.
    pub eigenvalue: R,
    /// Lanczos true residual `‖H v − λ v‖₂`.
    pub residual: R,
    /// Frobenius norm of singular values discarded by this step's
    /// truncated SVD.
    pub trunc_err: R,
    /// New bond dimension between `site` and `site + 1` after the
    /// truncated split.
    pub bond_dim: usize,
    pub lanczos_iters: usize,
    pub lanczos_converged: bool,
}

/// Per-sweep diagnostics record (one full L→R + R→L cycle).
#[derive(Debug, Clone)]
pub struct DmrgSweepRecord<R> {
    pub sweep: usize,
    /// Normalized post-truncation `<psi|H|psi> / <psi|psi>` after
    /// this cycle. The convergence metric.
    pub sweep_energy: R,
    /// `min(step.eigenvalue)` across this cycle. Diagnostic only —
    /// reflects local-block variational minima, which can be lower
    /// than `sweep_energy`.
    pub min_step_eigenvalue: R,
    pub max_trunc_err: R,
    pub max_bond: usize,
    pub all_lanczos_converged: bool,
    pub steps: Vec<DmrgStepRecord<R>>,
}

/// Final result of either 2-site DMRG sweep driver
/// ([`dmrg_2site_sweep`] for Dense and
/// [`super::sweep_block_sparse::dmrg_2site_sweep_block_sparse`] for
/// BlockSparse / U(1)).
#[derive(Debug, Clone)]
pub struct DmrgResult<R> {
    /// Last cycle's `sweep_energy`.
    pub energy: R,
    /// `true` iff the final cycle satisfied:
    /// `n_sweeps >= min_sweeps`,
    /// `|delta_E| <= energy_tol`,
    /// and every step's Lanczos converged.
    pub converged: bool,
    pub n_sweeps: usize,
    pub sweeps: Vec<DmrgSweepRecord<R>>,
}

/// Errors raised by the 2-site DMRG sweep drivers
/// ([`dmrg_2site_sweep`] for Dense and
/// [`super::sweep_block_sparse::dmrg_2site_sweep_block_sparse`] for
/// BlockSparse / U(1)).
#[derive(Debug)]
#[non_exhaustive]
pub enum DmrgSweepError {
    /// MPS, MPO, and `DmrgEnvs` disagree on `n_sites`.
    LengthMismatch { mps: usize, mpo: usize, envs: usize },
    /// `n_sites < 2`. 2-site sweeps require at least 2 sites.
    TooFewSites { n_sites: usize },
    /// `DmrgSweepParams` failed entry-point validation. `detail`
    /// names the constraint that fired.
    InvalidParams { detail: &'static str },
    /// MPS canonical form was not `Right` or `Mixed { center: 0 }`.
    /// `Unknown` is also rejected — see the module-level docs for
    /// the rationale.
    MpsNotRightCanonical { found: CanonicalForm },
    /// The per-step driver (`dmrg_2site_step` or
    /// `dmrg_2site_step_block_sparse`) returned an error. Source
    /// preserved.
    Step {
        sweep: usize,
        direction: SweepDirection,
        site: usize,
        source: DmrgHeffError,
    },
    /// `DmrgEnvs::advance_left/right` returned an error during a
    /// post-step env update. Surfaced separately from `Step` so
    /// the caller can distinguish "local solve failed" from "env
    /// state became inconsistent". Defense-in-depth — under the
    /// driver's own advance ordering, this branch should never
    /// fire from the public API.
    Env {
        sweep: usize,
        direction: SweepDirection,
        site: usize,
        source: DmrgEnvError,
    },
    /// The post-step S-absorb (`arnet_linalg::diagonal_scale` for
    /// Dense or `diagonal_scale_block_sparse` for BlockSparse)
    /// failed. Carries the same `(sweep, direction, site)`
    /// breadcrumbs as `Step` / `Env` so the caller can pin down
    /// where the failure occurred without having to walk the
    /// `DmrgResult::sweeps` history manually.
    Scale {
        sweep: usize,
        direction: SweepDirection,
        site: usize,
        source: LinalgError,
    },
}

impl std::fmt::Display for DmrgSweepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DmrgSweepError::LengthMismatch { mps, mpo, envs } => write!(
                f,
                "chain length mismatch: mps = {mps}, mpo = {mpo}, envs = {envs}"
            ),
            DmrgSweepError::TooFewSites { n_sites } => {
                write!(f, "2-site sweep requires n_sites >= 2, got {n_sites}")
            }
            DmrgSweepError::InvalidParams { detail } => {
                write!(f, "invalid DmrgSweepParams: {detail}")
            }
            DmrgSweepError::MpsNotRightCanonical { found } => write!(
                f,
                "MPS must be in Right or Mixed {{ center: 0 }} form before sweep, got {found:?}"
            ),
            DmrgSweepError::Step {
                sweep,
                direction,
                site,
                ..
            } => write!(
                f,
                "2-site DMRG step failed at sweep {sweep}, {direction:?}, site {site}"
            ),
            DmrgSweepError::Env {
                sweep,
                direction,
                site,
                ..
            } => write!(
                f,
                "DmrgEnvs advance failed at sweep {sweep}, {direction:?}, site {site}"
            ),
            DmrgSweepError::Scale {
                sweep,
                direction,
                site,
                ..
            } => write!(
                f,
                "S-absorb (diagonal scale) failed during sweep {sweep}, {direction:?}, site {site}"
            ),
        }
    }
}

impl std::error::Error for DmrgSweepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DmrgSweepError::Step { source, .. } => Some(source),
            DmrgSweepError::Env { source, .. } => Some(source),
            DmrgSweepError::Scale { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Run alternating L→R / R→L sweeps until convergence or
/// `max_sweeps`. Mutates `mps` and `envs` in place; the final MPS
/// state is `CanonicalForm::Mixed { center: 0 }` (R→L runs last).
///
/// See the module-level rustdoc for the canonical-form contract on
/// the input MPS and for the convergence criterion.
pub fn dmrg_2site_sweep<T, B>(
    envs: &mut DmrgEnvs<Dense<T>, B>,
    mps: &mut Mps<Dense<T>, B>,
    mpo: &Mpo<Dense<T>, B>,
    params: &DmrgSweepParams,
) -> Result<DmrgResult<T::Real>, DmrgSweepError>
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
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
    // finite value outside f32 range (NumCast::from returns Some(inf),
    // which try_real_from_f64 then maps to None). Surface that as
    // `InvalidParams` so the public API stays fallible end-to-end
    // instead of panicking inside lanczos. lanczos.tol is gated here
    // too so a borderline f32 tol does not slip past sweep-level
    // validation only to abort the run from inside the local solve.
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
            let record = run_step(
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

            // Skip the trailing advance: site = n_sites - 2 is
            // the last L→R step; the next R→L step at the same
            // `site` consumes `left(site)` which is still valid.
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
            let record = run_step(
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

            // Always advance, including at the trailing `site == 0`
            // boundary. Skipping the boundary advance would leave
            // `right[1]` stale-but-`Some` (it would still hold the
            // pre-sweep `DmrgEnvs::build` value, computed against
            // the original MPS site 1) and `left[1]` stale-but-
            // `Some` (it would still hold the L→R-time value, even
            // though R→L has further mutated MPS site 0). Both are
            // contract violations against
            // `DmrgEnvs`'s "stale = None" convention even though
            // they do not affect the next sweep iteration's
            // numerics, which overwrites `left[1]` via
            // `advance_left(0)` before consumption.
            envs.advance_right(mps, mpo, site + 1)
                .map_err(|source| DmrgSweepError::Env {
                    sweep: sweep_idx,
                    direction: SweepDirection::RightToLeft,
                    site,
                    source,
                })?;
        }

        // R→L ends with the orthogonality center at site 0
        // (S absorbed leftward at every step). storage_mut reset
        // the form to Unknown along the way; re-pin it here so a
        // caller breaking out mid-loop sees a coherent state.
        mps.set_canonical_form(CanonicalForm::Mixed { center: 0 });

        // ---- Post-sweep diagnostics -----------------------------
        let bra_h_ket = braket(mps, mpo, mps);
        let nrm = norm(mps);
        let nrm_sq: T::Real = nrm * nrm;
        // T::Real / T::Real is always available since T::Real: Float.
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

/// Run a single 2-site step at `site`, then mutate the MPS site
/// tensors at `site` and `site + 1` according to `direction`.
///
/// Returns the step's diagnostics record. The caller is responsible
/// for advancing the env afterwards (or skipping the advance at the
/// trailing boundary of a half-sweep).
#[allow(clippy::too_many_arguments)]
fn run_step<T, B>(
    envs: &DmrgEnvs<Dense<T>, B>,
    mps: &mut Mps<Dense<T>, B>,
    mpo: &Mpo<Dense<T>, B>,
    site: usize,
    params: &DmrgSweepParams,
    sweep_idx: usize,
    direction: SweepDirection,
    backend: &Arc<B>,
) -> Result<DmrgStepRecord<T::Real>, DmrgSweepError>
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
    B: ComputeBackend,
{
    let result = dmrg_2site_step(envs, mps, mpo, site, &params.lanczos, &params.trunc).map_err(
        |source| DmrgSweepError::Step {
            sweep: sweep_idx,
            direction,
            site,
            source,
        },
    )?;

    let bond_dim = result.s.shape()[0];

    // Wrap any `diagonal_scale` failure with the same
    // (sweep, direction, site) breadcrumbs as `Step` / `Env`.
    let scale_err = |source: LinalgError| DmrgSweepError::Scale {
        sweep: sweep_idx,
        direction,
        site,
        source,
    };

    // Absorb S into the sweep direction and write back to MPS.
    match direction {
        SweepDirection::LeftToRight => {
            // site i ← U (left-isometric)
            // site i+1 ← S·Vt (axis 0 = new bond, carries S right)
            let s_vt =
                diagonal_scale(&**backend, &result.vt, result.s.data(), 0).map_err(scale_err)?;
            *mps.storage_mut(site) = result.u;
            *mps.storage_mut(site + 1) = s_vt;
        }
        SweepDirection::RightToLeft => {
            // site i   ← U·S  (axis 2 = new bond, carries S left)
            // site i+1 ← Vt   (right-isometric)
            let u_s =
                diagonal_scale(&**backend, &result.u, result.s.data(), 2).map_err(scale_err)?;
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

pub(super) fn validate_params(params: &DmrgSweepParams) -> Result<(), DmrgSweepError> {
    if params.max_sweeps == 0 {
        return Err(DmrgSweepError::InvalidParams {
            detail: "max_sweeps must be >= 1",
        });
    }
    if params.min_sweeps > params.max_sweeps {
        return Err(DmrgSweepError::InvalidParams {
            detail: "min_sweeps must be <= max_sweeps",
        });
    }
    if !params.energy_tol.is_finite() {
        return Err(DmrgSweepError::InvalidParams {
            detail: "energy_tol must be finite",
        });
    }
    if params.energy_tol < 0.0 {
        return Err(DmrgSweepError::InvalidParams {
            detail: "energy_tol must be non-negative",
        });
    }
    if params.lanczos.max_iter == 0 {
        return Err(DmrgSweepError::InvalidParams {
            detail: "lanczos.max_iter must be >= 1",
        });
    }
    if !params.lanczos.tol.is_finite() {
        return Err(DmrgSweepError::InvalidParams {
            detail: "lanczos.tol must be finite",
        });
    }
    if params.lanczos.tol < 0.0 {
        return Err(DmrgSweepError::InvalidParams {
            detail: "lanczos.tol must be non-negative",
        });
    }
    if let Some(chi) = params.trunc.chi_max
        && chi == 0
    {
        return Err(DmrgSweepError::InvalidParams {
            detail: "trunc.chi_max must be > 0 if Some",
        });
    }
    if let Some(te) = params.trunc.target_trunc_err {
        if !te.is_finite() {
            return Err(DmrgSweepError::InvalidParams {
                detail: "trunc.target_trunc_err must be finite",
            });
        }
        if te < 0.0 {
            return Err(DmrgSweepError::InvalidParams {
                detail: "trunc.target_trunc_err must be non-negative",
            });
        }
    }
    Ok(())
}
