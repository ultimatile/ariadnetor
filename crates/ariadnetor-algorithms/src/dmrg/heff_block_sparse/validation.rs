//! Entry-point validation for the BlockSparse 2-site DMRG step.
//!
//! Performs every input-side check up front so the matvec body's
//! `.expect` calls cannot fire on user input. Covers ranks, per-axis
//! `total_dim() >= 1`, cross-tensor dim agreement, contracted-axis
//! QNIndex compatibility (opposite directions + matching sector
//! lists), env free-output-leg equality with the psi template, MPO
//! well-formedness (bra-ket duality and bra-vs-MPS-phys direction),
//! identity-flux preconditions on env / MPO sites, and the selected
//! local eigensolver's parameter sanity (Lanczos by default, ARPACK
//! behind the `arpack` feature).

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_mps::{MpoRepr as Mpo, MpsRepr as Mps, TensorChainRepr as TensorChain};
use arnet_tensor::{BlockSparse, QNIndex, Sector};

use super::super::env::DmrgEnvs;
use super::super::heff_error::DmrgHeffError;
use super::super::solver::{LocalEigensolverParams, eigensolver_tol, validate_eigensolver_params};

/// Validated input handles + derived dims, returned to the caller
/// (the entry point in `mod.rs`) so it can build the Heff and drive
/// the local eigensolver without re-deriving anything.
pub(super) struct ValidatedInputs<'a, T: Scalar, S: Sector, B: ComputeBackend> {
    pub left: &'a BlockSparse<T, S>,
    pub right: &'a BlockSparse<T, S>,
    pub w_i: &'a BlockSparse<T, S>,
    pub w_ip1: &'a BlockSparse<T, S>,
    pub mps_i: &'a BlockSparse<T, S>,
    pub mps_ip1: &'a BlockSparse<T, S>,
    pub backend: Arc<B>,
}

pub(super) fn validate_inputs<'a, T, S, B>(
    envs: &'a DmrgEnvs<BlockSparse<T, S>, B>,
    mps: &'a Mps<BlockSparse<T, S>, B>,
    mpo: &'a Mpo<BlockSparse<T, S>, B>,
    site: usize,
    eigensolver: &LocalEigensolverParams,
) -> Result<ValidatedInputs<'a, T, S, B>, DmrgHeffError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let n_sites = envs.n_sites();
    if mps.len() != n_sites || mpo.len() != n_sites {
        return Err(DmrgHeffError::LengthMismatch {
            mps: mps.len(),
            mpo: mpo.len(),
            envs: n_sites,
        });
    }
    if site >= n_sites.saturating_sub(1) {
        return Err(DmrgHeffError::InvalidSite { site, n_sites });
    }

    validate_eigensolver_params(eigensolver)
        .map_err(|detail| DmrgHeffError::InvalidEigensolverParams { detail })?;
    if crate::numeric::try_real_from_f64::<T>(eigensolver_tol(eigensolver)).is_none() {
        return Err(DmrgHeffError::InvalidEigensolverParams {
            detail: "tol is not representable in T::Real",
        });
    }

    let left = envs.left(site).ok_or(DmrgHeffError::StaleEnv {
        side: "left",
        index: site,
    })?;
    let right = envs.right(site + 2).ok_or(DmrgHeffError::StaleEnv {
        side: "right",
        index: site + 2,
    })?;
    let w_i = mpo.storage(site);
    let w_ip1 = mpo.storage(site + 1);
    let mps_i = mps.storage(site);
    let mps_ip1 = mps.storage(site + 1);
    let backend: Arc<B> = mps.backend_arc().clone();

    let check_eq =
        |expected: usize, actual: usize, field: &'static str| -> Result<(), DmrgHeffError> {
            if expected == actual {
                Ok(())
            } else {
                Err(DmrgHeffError::ShapeMismatch {
                    site,
                    field,
                    expected,
                    actual,
                })
            }
        };
    let check_at_least =
        |min: usize, actual: usize, field: &'static str| -> Result<(), DmrgHeffError> {
            if actual >= min {
                Ok(())
            } else {
                Err(DmrgHeffError::ShapeMismatch {
                    site,
                    field,
                    expected: min,
                    actual,
                })
            }
        };

    check_eq(3, left.rank(), "left.rank")?;
    check_eq(3, right.rank(), "right.rank")?;
    check_eq(4, w_i.rank(), "W[i].rank")?;
    check_eq(4, w_ip1.rank(), "W[i+1].rank")?;
    check_eq(3, mps_i.rank(), "MPS[i].rank")?;
    check_eq(3, mps_ip1.rank(), "MPS[i+1].rank")?;

    check_at_least(1, left.shape()[0], "left.top_bra (chi_l) total_dim")?;
    check_at_least(1, right.shape()[0], "right.top_bra (chi_r) total_dim")?;
    check_at_least(1, mps_i.shape()[1], "MPS[i].physical total_dim")?;
    check_at_least(1, mps_ip1.shape()[1], "MPS[i+1].physical total_dim")?;
    check_at_least(1, w_i.shape()[0], "W[i].W_l total_dim")?;
    check_at_least(1, w_i.shape()[3], "W[i].W_r total_dim")?;
    check_at_least(1, w_ip1.shape()[3], "W[i+1].W_r total_dim")?;

    let chi_l = left.shape()[0];
    let chi_r = right.shape()[0];
    let d_i = mps_i.shape()[1];
    let d_ip1 = mps_ip1.shape()[1];
    check_eq(
        left.shape()[2],
        mps_i.shape()[0],
        "left.bot_ket vs MPS[i].left_bond total_dim",
    )?;
    check_eq(
        left.shape()[2],
        chi_l,
        "left.bot_ket vs left.top_bra total_dim",
    )?;
    check_eq(
        right.shape()[2],
        mps_ip1.shape()[2],
        "right.bot_ket vs MPS[i+1].right_bond total_dim",
    )?;
    check_eq(
        right.shape()[2],
        chi_r,
        "right.bot_ket vs right.top_bra total_dim",
    )?;
    check_eq(
        left.shape()[1],
        w_i.shape()[0],
        "left.W_bond vs W[i].W_l total_dim",
    )?;
    check_eq(
        right.shape()[1],
        w_ip1.shape()[3],
        "right.W_bond vs W[i+1].W_r total_dim",
    )?;
    check_eq(
        w_i.shape()[3],
        w_ip1.shape()[0],
        "W[i].W_r vs W[i+1].W_l total_dim",
    )?;
    check_eq(w_i.shape()[1], d_i, "W[i].d_ket vs MPS[i] phys total_dim")?;
    check_eq(w_i.shape()[2], d_i, "W[i].d_bra vs MPS[i] phys total_dim")?;
    check_eq(
        w_ip1.shape()[1],
        d_ip1,
        "W[i+1].d_ket vs MPS[i+1] phys total_dim",
    )?;
    check_eq(
        w_ip1.shape()[2],
        d_ip1,
        "W[i+1].d_bra vs MPS[i+1] phys total_dim",
    )?;

    // Contracted-axis pairs: opposite direction + same sector list.
    check_qn_pair(
        site,
        "left.bot_ket vs psi.axis 0 (MPS[i].left_bond)",
        &left.indices()[2],
        &mps_i.indices()[0],
        true,
    )?;
    check_qn_pair(
        site,
        "left.W_bond vs W[i].W_l",
        &left.indices()[1],
        &w_i.indices()[0],
        true,
    )?;
    check_qn_pair(
        site,
        "psi.axis 1 (MPS[i].phys) vs W[i].ket",
        &mps_i.indices()[1],
        &w_i.indices()[1],
        true,
    )?;
    check_qn_pair(
        site,
        "psi.axis 2 (MPS[i+1].phys) vs W[i+1].ket",
        &mps_ip1.indices()[1],
        &w_ip1.indices()[1],
        true,
    )?;
    check_qn_pair(
        site,
        "W[i].W_r vs W[i+1].W_l",
        &w_i.indices()[3],
        &w_ip1.indices()[0],
        true,
    )?;
    check_qn_pair(
        site,
        "psi.axis 3 (MPS[i+1].right_bond) vs right.bot_ket",
        &mps_ip1.indices()[2],
        &right.indices()[2],
        true,
    )?;
    check_qn_pair(
        site,
        "right.W_bond vs W[i+1].W_r",
        &right.indices()[1],
        &w_ip1.indices()[3],
        true,
    )?;

    // Env free output legs must equal the psi template's outer
    // axes (same direction + same sectors + same per-sector dims).
    check_qn_pair(
        site,
        "left.top_bra vs psi.axis 0 (MPS[i].left_bond)",
        &left.indices()[0],
        &mps_i.indices()[0],
        false,
    )?;
    check_qn_pair(
        site,
        "right.top_bra vs psi.axis 3 (MPS[i+1].right_bond)",
        &right.indices()[0],
        &mps_ip1.indices()[2],
        false,
    )?;

    // MPO well-formedness: bra leg = dual of ket leg in QNIndex
    // (opposite direction + same sectors), and bra direction
    // matches MPS physical direction so the matvec output's axis
    // 1 / 2 land in the psi template's axes 1 / 2 cleanly.
    check_qn_pair(
        site,
        "W[i].bra vs W[i].ket",
        &w_i.indices()[2],
        &w_i.indices()[1],
        true,
    )?;
    check_qn_pair(
        site,
        "W[i+1].bra vs W[i+1].ket",
        &w_ip1.indices()[2],
        &w_ip1.indices()[1],
        true,
    )?;
    if w_i.indices()[2].direction() != mps_i.indices()[1].direction() {
        return Err(DmrgHeffError::QnMismatch {
            site,
            field: "W[i].bra direction vs MPS[i] phys direction",
            detail: format!(
                "W[i].bra direction = {:?}, MPS[i].phys direction = {:?} (must be equal)",
                w_i.indices()[2].direction(),
                mps_i.indices()[1].direction()
            ),
        });
    }
    if w_ip1.indices()[2].direction() != mps_ip1.indices()[1].direction() {
        return Err(DmrgHeffError::QnMismatch {
            site,
            field: "W[i+1].bra direction vs MPS[i+1] phys direction",
            detail: format!(
                "W[i+1].bra direction = {:?}, MPS[i+1].phys direction = {:?} (must be equal)",
                w_ip1.indices()[2].direction(),
                mps_ip1.indices()[1].direction()
            ),
        });
    }

    // (The empty-psi-template guard — checking that
    // `BlockSparse::zeros(psi_indices, psi_flux)` has at least one
    // flux-allowed block — is handled in the entry point after
    // `EffectiveHamiltonian2SiteBlockSparse::new` builds the real
    // template, to avoid allocating it twice. See the
    // `heff.dim() == 0` branch in `mod.rs::dmrg_2site_step_block_sparse`.)

    // Identity-flux preconditions on env / MPO sites. Without
    // these the matvec output's flux drifts away from psi_flux and
    // the gather template flux check fails inside `apply`.
    if !is_identity_flux(left.flux()) {
        return Err(DmrgHeffError::QnMismatch {
            site,
            field: "left.flux",
            detail: format!("left.flux = {:?} (must be identity)", left.flux()),
        });
    }
    if !is_identity_flux(right.flux()) {
        return Err(DmrgHeffError::QnMismatch {
            site,
            field: "right.flux",
            detail: format!("right.flux = {:?} (must be identity)", right.flux()),
        });
    }
    if !is_identity_flux(w_i.flux()) {
        return Err(DmrgHeffError::QnMismatch {
            site,
            field: "W[i].flux",
            detail: format!("W[i].flux = {:?} (must be identity)", w_i.flux()),
        });
    }
    if !is_identity_flux(w_ip1.flux()) {
        return Err(DmrgHeffError::QnMismatch {
            site,
            field: "W[i+1].flux",
            detail: format!("W[i+1].flux = {:?} (must be identity)", w_ip1.flux()),
        });
    }

    Ok(ValidatedInputs {
        left,
        right,
        w_i,
        w_ip1,
        mps_i,
        mps_ip1,
        backend,
    })
}

/// Verify that two QNIndices are compatible for a contracted-axis
/// pair (`opposite_direction = true`) or a free-axis match
/// (`opposite_direction = false`). A contracted pair must have
/// opposite directions; a free pair must have equal directions.
/// Both cases require the same sector list (with matching
/// per-sector dims) modulo direction.
fn check_qn_pair<S: Sector>(
    site: usize,
    field: &'static str,
    lhs: &QNIndex<S>,
    rhs: &QNIndex<S>,
    opposite_direction: bool,
) -> Result<(), DmrgHeffError> {
    let dirs_ok = if opposite_direction {
        lhs.direction() != rhs.direction()
    } else {
        lhs.direction() == rhs.direction()
    };
    if !dirs_ok {
        return Err(DmrgHeffError::QnMismatch {
            site,
            field,
            detail: format!(
                "directions {:?} vs {:?} ({})",
                lhs.direction(),
                rhs.direction(),
                if opposite_direction {
                    "must be opposite"
                } else {
                    "must be equal"
                }
            ),
        });
    }
    if lhs.blocks() != rhs.blocks() {
        return Err(DmrgHeffError::QnMismatch {
            site,
            field,
            detail: format!(
                "sector lists {:?} vs {:?} (must match by sector + per-sector dim)",
                lhs.blocks(),
                rhs.blocks()
            ),
        });
    }
    Ok(())
}

fn is_identity_flux<S: Sector>(flux: &S) -> bool {
    flux == &S::identity()
}
