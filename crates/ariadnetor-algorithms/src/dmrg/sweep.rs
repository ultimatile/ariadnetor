//! 2-site DMRG sweep driver.
//!
//! Runs alternating L→R / R→L half-sweeps over a [`super::DmrgOps`]
//! layout type, dispatching the layout-specific local solve and
//! S-absorb through the trait. Mutates the MPS site tensors and the
//! [`DmrgEnvs`] in place. Caller supplies a right-canonical (or
//! `Mixed { center: 0 }`) MPS plus a freshly-built `DmrgEnvs`; the
//! driver does **not** auto-canonicalize because doing so would
//! silently invalidate the caller-supplied envs.
//!
//! # Canonical-form precondition
//!
//! The local effective-Hamiltonian eigenvalue equation returns
//! physical energy directly only when, with active block `(i, i+1)`,
//! sites `(0..i)` are left-canonical and sites `(i+2..N-1)` are
//! right-canonical. The driver starts L→R from `i = 0`, so the
//! binding precondition is right-canonicality of `(2..N-1)`, met
//! exactly by `Right` and `Mixed { center: 0 }`. This argument is
//! storage-independent (the local-block requirement does not depend
//! on Dense vs BlockSparse representation), so the gate applies
//! uniformly through the trait.
//!
//! # Convergence
//!
//! After each full L→R + R→L cycle we record the **normalized**
//! post-truncation expectation `<psi|H|psi>.re() / <psi|psi>`.
//! The truncated SVD keeps unrenormalized singular values, so without
//! the `<psi|psi>` divisor the sweep energy drifts toward zero
//! whenever truncation happens — the divisor strips that
//! norm-artifact away. Convergence requires energy delta within
//! `energy_tol`, every step's local eigensolver converged, and
//! `n_sweeps >= min_sweeps`.

use std::sync::Arc;

use arnet_core::{ComputeBackend, Scalar};
use arnet_linalg::LinalgError;
use arnet_mps::{CanonicalForm, Mpo, Mps, MpsOps, TensorChain, braket, norm};

use crate::numeric::try_real_from_f64;

use super::dispatch::DmrgOps;
use super::env::{DmrgEnvError, DmrgEnvs};
use super::heff_error::DmrgHeffError;
use super::solver::{LocalEigensolverParams, eigensolver_tol, validate_eigensolver_params};

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

/// Caller-supplied parameters for [`sweep_2site`] (the layout-generic
/// 2-site DMRG sweep driver, dispatched over `L: super::DmrgOps<T, B>`
/// so the same params type covers both the Dense and BlockSparse /
/// U(1) paths).
///
/// Stored as plain `f64` for `energy_tol`; the entry point converts
/// it to `T::Real` via the same `NumCast::from(f64)` idiom as
/// [`crate::krylov::LanczosParams::tol`]. This keeps the params
/// type non-generic across the scalar type.
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
    /// Local-eigensolver selector + per-variant parameters, forwarded
    /// to the per-step driver (Dense `dmrg_2site_step` or BlockSparse
    /// `dmrg_2site_step_block_sparse`).
    pub eigensolver: LocalEigensolverParams,
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
    /// Local-eigensolver true residual `‖H v − λ v‖₂`.
    pub residual: R,
    /// Frobenius norm of singular values discarded by this step's
    /// truncated SVD.
    pub trunc_err: R,
    /// New bond dimension between `site` and `site + 1` after the
    /// truncated split.
    pub bond_dim: usize,
    /// Number of iterations the local eigensolver ran for this step.
    /// For Lanczos this is the inner loop count; for ARPACK it is
    /// the restart-iteration count returned in `iparam[2]`.
    pub eigensolver_iters: usize,
    /// `true` iff the local eigensolver succeeded — Lanczos by its
    /// absolute true-residual test against `LanczosParams::tol`,
    /// ARPACK by its relative-tol stopping criterion (i.e. `Ok`
    /// return from `arpack_smallest`). The two arms intentionally
    /// disagree on what they call "converged": Lanczos uses the
    /// absolute residual; ARPACK uses `residual <= tol * |lambda|`.
    /// See [`super::heff::TwoSiteStepResult::converged`] for the
    /// upstream contract this field forwards from.
    pub eigensolver_converged: bool,
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
    /// `true` iff every step in this cycle's local-eigensolver pass
    /// converged.
    pub all_eigensolver_converged: bool,
    pub steps: Vec<DmrgStepRecord<R>>,
}

/// Final result of the 2-site DMRG sweep driver [`sweep_2site`].
#[derive(Debug, Clone)]
pub struct DmrgResult<R> {
    /// Last cycle's `sweep_energy`.
    pub energy: R,
    /// `true` iff the final cycle satisfied:
    /// `n_sweeps >= min_sweeps`,
    /// `|delta_E| <= energy_tol`,
    /// and every step's local eigensolver converged.
    pub converged: bool,
    pub n_sweeps: usize,
    pub sweeps: Vec<DmrgSweepRecord<R>>,
}

/// Errors raised by the 2-site DMRG sweep driver [`sweep_2site`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DmrgSweepError {
    /// MPS, MPO, and `DmrgEnvs` disagree on `n_sites`.
    #[error("chain length mismatch: mps = {mps}, mpo = {mpo}, envs = {envs}")]
    LengthMismatch { mps: usize, mpo: usize, envs: usize },
    /// `n_sites < 2`. 2-site sweeps require at least 2 sites.
    #[error("2-site sweep requires n_sites >= 2, got {n_sites}")]
    TooFewSites { n_sites: usize },
    /// `DmrgSweepParams` failed entry-point validation. `detail`
    /// names the constraint that fired.
    #[error("invalid DmrgSweepParams: {detail}")]
    InvalidParams { detail: &'static str },
    /// MPS canonical form was not `Right` or `Mixed { center: 0 }`.
    /// `Unknown` is also rejected — see the module-level docs for
    /// the rationale.
    #[error("MPS must be in Right or Mixed {{ center: 0 }} form before sweep, got {found:?}")]
    MpsNotRightCanonical { found: CanonicalForm },
    /// The per-step driver (`dmrg_2site_step` or
    /// `dmrg_2site_step_block_sparse`) returned an error. Source
    /// preserved.
    #[error("2-site DMRG step failed at sweep {sweep}, {direction:?}, site {site}")]
    Step {
        sweep: usize,
        direction: SweepDirection,
        site: usize,
        #[source]
        source: DmrgHeffError,
    },
    /// `DmrgEnvs::advance_left/right` returned an error during a
    /// post-step env update. Surfaced separately from `Step` so
    /// the caller can distinguish "local solve failed" from "env
    /// state became inconsistent". Defense-in-depth — under the
    /// driver's own advance ordering, this branch should never
    /// fire from the public API.
    #[error("DmrgEnvs advance failed at sweep {sweep}, {direction:?}, site {site}")]
    Env {
        sweep: usize,
        direction: SweepDirection,
        site: usize,
        #[source]
        source: DmrgEnvError,
    },
    /// The post-step S-absorb (`arnet_linalg::diagonal_scale_with_backend`
    /// for Dense or `arnet_linalg::diagonal_scale_block_sparse_with_backend`
    /// for BlockSparse) failed. Carries the same `(sweep, direction, site)`
    /// breadcrumbs as `Step` / `Env` so the caller can pin down
    /// where the failure occurred without having to walk the
    /// `DmrgResult::sweeps` history manually.
    #[error("S-absorb (diagonal scale) failed during sweep {sweep}, {direction:?}, site {site}")]
    Scale {
        sweep: usize,
        direction: SweepDirection,
        site: usize,
        #[source]
        source: LinalgError,
    },
}

/// Run alternating L→R / R→L sweeps until convergence or
/// `max_sweeps` over a [`DmrgOps`] layout type. Mutates `mps` and
/// `envs` in place; the final MPS state is
/// `CanonicalForm::Mixed { center: 0 }` (R→L runs last).
///
/// Generic over `L: DmrgOps<T, B>`, so a single call site covers
/// both the Dense (`L = DenseLayout`) and BlockSparse / U(1)
/// (`L = BlockSparseLayout<S>`) paths. The trait dispatches the
/// local solve and S-absorb to the layout-specific implementations.
///
/// See the module-level rustdoc for the canonical-form contract on
/// the input MPS and for the convergence criterion.
pub fn sweep_2site<T, L, B>(
    envs: &mut DmrgEnvs<<L as MpsOps<T>>::Storage, L, B>,
    mps: &mut Mps<<L as MpsOps<T>>::Storage, L, B>,
    mpo: &Mpo<<L as MpsOps<T>>::Storage, L, B>,
    params: &DmrgSweepParams,
) -> Result<DmrgResult<T::Real>, DmrgSweepError>
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
    L: DmrgOps<T, B>,
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

    // ---- Tier 2: backend preferred_order across operands --------
    assert_eq!(
        mps.backend().preferred_order(),
        mpo.backend().preferred_order(),
        "sweep_2site: mps/mpo backend preferred_order mismatch",
    );
    let chain_backend_arc: Arc<B> = mps.backend_arc().clone();
    <L as crate::dmrg::env::DmrgEnvOps<T>>::assert_chain_order(
        &chain_backend_arc,
        mps.sites(),
        "sweep_2site.mps",
    );
    <L as crate::dmrg::env::DmrgEnvOps<T>>::assert_chain_order(
        &chain_backend_arc,
        mpo.sites(),
        "sweep_2site.mpo",
    );

    // ---- Param validation ---------------------------------------
    validate_params(params)?;
    // Casts may fail when the real scalar type (`T::Real`) is `f32`
    // and the user supplied a
    // finite value outside f32 range (NumCast::from returns Some(inf),
    // which try_real_from_f64 then maps to None). Surface that as
    // `InvalidParams` so the public API stays fallible end-to-end
    // instead of failing inside the local eigensolver (Lanczos's
    // internal `try_real_from_f64` would panic; ARPACK's `tol_real`
    // cast would also panic). The selected eigensolver's `tol` is
    // gated here too so a borderline f32 tol does not slip past
    // sweep-level validation only to abort the run from inside the
    // local solve.
    let energy_tol_real: T::Real =
        try_real_from_f64::<T>(params.energy_tol).ok_or(DmrgSweepError::InvalidParams {
            detail: "energy_tol is not representable in the storage's real scalar type",
        })?;
    if try_real_from_f64::<T>(eigensolver_tol(&params.eigensolver)).is_none() {
        return Err(DmrgSweepError::InvalidParams {
            detail: "eigensolver tol is not representable in the storage's real scalar type",
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

    // Reuse the entry-derived chain handle rather than deriving again.
    let backend: Arc<B> = chain_backend_arc;
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
                envs.advance_left::<T>(mps, mpo, site)
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
            envs.advance_right::<T>(mps, mpo, site + 1)
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
        let bra_h_ket: T = braket::<T, L, B>(mps, mpo, mps);
        let nrm: T::Real = norm::<T, L, B>(mps);
        let nrm_sq: T::Real = nrm * nrm;
        // The Float bound on the storage's real scalar type guarantees division.
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
            if !s.eigensolver_converged {
                all_ok = false;
            }
        }

        sweeps.push(DmrgSweepRecord {
            sweep: sweep_idx,
            sweep_energy,
            min_step_eigenvalue: min_eig,
            max_trunc_err: max_te,
            max_bond,
            all_eigensolver_converged: all_ok,
            steps,
        });

        completed_sweeps = sweep_idx + 1;

        // ---- Convergence check ----------------------------------
        if completed_sweeps >= params.min_sweeps
            && let Some(prev) = last_energy
        {
            let abs_delta = (sweep_energy - prev).abs();
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

/// Run a single 2-site step at `site` over a [`DmrgOps`] storage,
/// then mutate the MPS site tensors at `site` and `site + 1`
/// according to `direction`.
///
/// Returns the step's diagnostics record. The caller is responsible
/// for advancing the env afterwards (or skipping the advance at the
/// trailing boundary of a half-sweep).
#[allow(clippy::too_many_arguments)]
fn run_step<T, L, B>(
    envs: &DmrgEnvs<<L as MpsOps<T>>::Storage, L, B>,
    mps: &mut Mps<<L as MpsOps<T>>::Storage, L, B>,
    mpo: &Mpo<<L as MpsOps<T>>::Storage, L, B>,
    site: usize,
    params: &DmrgSweepParams,
    sweep_idx: usize,
    direction: SweepDirection,
    backend: &Arc<B>,
) -> Result<DmrgStepRecord<T::Real>, DmrgSweepError>
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
    L: DmrgOps<T, B>,
    B: ComputeBackend,
{
    let result =
        <L as DmrgOps<T, B>>::step(envs, mps, mpo, site, &params.eigensolver, &params.trunc)
            .map_err(|source| DmrgSweepError::Step {
                sweep: sweep_idx,
                direction,
                site,
                source,
            })?;

    // Project the typed scalars before `commit_step` consumes `result`.
    let (eigenvalue, residual, trunc_err, iters, converged) =
        <L as DmrgOps<T, B>>::step_scalars(&result);

    let scale_err = |source: LinalgError| DmrgSweepError::Scale {
        sweep: sweep_idx,
        direction,
        site,
        source,
    };

    let absorbed =
        <L as DmrgOps<T, B>>::commit_step(backend, result, direction).map_err(scale_err)?;
    *mps.site_mut(site) = absorbed.site_i;
    *mps.site_mut(site + 1) = absorbed.site_ip1;

    Ok(DmrgStepRecord {
        sweep: sweep_idx,
        direction,
        site,
        eigenvalue,
        residual,
        trunc_err,
        bond_dim: absorbed.bond_dim,
        eigensolver_iters: iters,
        eigensolver_converged: converged,
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
    validate_eigensolver_params(&params.eigensolver)
        .map_err(|detail| DmrgSweepError::InvalidParams { detail })?;
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
