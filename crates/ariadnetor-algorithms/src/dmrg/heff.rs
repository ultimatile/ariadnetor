//! 2-site DMRG local update: build the effective Hamiltonian
//! operator from `(left(i), W[i], W[i+1], right(i+2))`, drive Lanczos
//! to the smallest eigenpair, and split the optimized two-site block
//! back into a left-canonical / right-canonical pair via truncated
//! SVD.
//!
//! Axis convention (consistent with [`super::env`] and
//! `arnet_mps::inner::braket_dense`):
//!
//! - Env tensor `(top-bra-bond, W-bond, bot-ket-bond)` with bra = ket
//!   = psi for ground-state DMRG.
//! - MPO site tensor `[W_l, d_ket, d_bra, W_r]`. Axis 1 is the ket
//!   physical leg (pairs with the input MPS); axis 2 is the bra
//!   physical leg (pairs with the conjugated MPS).
//! - 2-site block `psi[chi_l, d_i, d_{i+1}, chi_r]` with the two
//!   physical legs occupying the inner axes.
//!
//! [`EffectiveHamiltonian2Site`] borrows the env / MPO references and
//! implements [`LinearOp`] so the local matvec can drive the existing
//! Lanczos solver without materializing `H_eff` as a dense matrix.
//! [`dmrg_2site_step`] wires that operator through Lanczos and then
//! through `arnet_linalg::trunc_svd`, returning the U / S / Vt
//! factors separately so the sweep driver (Phase 4) can decide which
//! direction absorbs S.

use std::sync::Arc;

use arnet_core::backend::ComputeBackend;
use arnet_core::{MemoryOrder, Scalar};
use arnet_linalg::{LinalgError, TruncSvdParams, contract, trunc_svd};
use arnet_mps::{Mpo, Mps, TensorChain};
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, reorder};

use crate::krylov::{LanczosParams, LinearOp, lanczos_smallest};

use super::env::DmrgEnvs;

/// Errors raised by [`dmrg_2site_step`].
#[derive(Debug)]
#[non_exhaustive]
pub enum DmrgHeffError {
    /// `site + 1` was not a valid two-site index for the chain.
    InvalidSite { site: usize, n_sites: usize },
    /// The env slot required for the two-site step (`left(site)` for
    /// the left side, `right(site + 2)` for the right side) was
    /// `None`. Indicates the caller has not built / advanced the
    /// envs into a state where this `site` can be optimized.
    StaleEnv { side: &'static str, index: usize },
    /// MPS and MPO chain lengths disagree, or disagree with the
    /// envs the function was given.
    LengthMismatch { mps: usize, mpo: usize, envs: usize },
    /// A bond / physical dimension on one of the inputs to the
    /// 2-site step did not match the expectation derived from the
    /// surrounding tensors. Surfaced *before* the matvec runs so
    /// the operator's `.expect` calls can stay infallible. `field`
    /// names the constraint that failed (e.g.,
    /// `"left.bot_ket vs mps[i].left_bond"`).
    ShapeMismatch {
        site: usize,
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// An underlying `arnet_linalg` call (currently the truncated
    /// SVD) failed. The matvec body itself is shape-validated up
    /// front and never reaches this branch.
    Contract(LinalgError),
}

impl From<LinalgError> for DmrgHeffError {
    fn from(e: LinalgError) -> Self {
        DmrgHeffError::Contract(e)
    }
}

impl std::fmt::Display for DmrgHeffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DmrgHeffError::InvalidSite { site, n_sites } => write!(
                f,
                "two-site index {site} (with {site}+1) out of range for chain of length {n_sites}"
            ),
            DmrgHeffError::StaleEnv { side, index } => write!(
                f,
                "{side} env at index {index} is stale (None); build / advance envs into the right \
                 state before stepping"
            ),
            DmrgHeffError::LengthMismatch { mps, mpo, envs } => write!(
                f,
                "chain length mismatch: mps = {mps}, mpo = {mpo}, envs = {envs}"
            ),
            DmrgHeffError::ShapeMismatch {
                site,
                field,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch at site {site}, {field}: expected {expected}, got {actual}"
            ),
            DmrgHeffError::Contract(_) => {
                write!(f, "linalg failure during two-site DMRG step")
            }
        }
    }
}

impl std::error::Error for DmrgHeffError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DmrgHeffError::Contract(err) => Some(err),
            _ => None,
        }
    }
}

/// Effective Hamiltonian operator for the 2-site DMRG block at sites
/// `(i, i+1)`. Built once per local update and consumed by the
/// Lanczos solver via [`LinearOp`].
#[derive(Debug, Clone)]
pub struct EffectiveHamiltonian2Site<'a, T: Scalar, B: ComputeBackend = NativeBackend> {
    left: &'a Dense<T>,
    w_i: &'a Dense<T>,
    w_ip1: &'a Dense<T>,
    right: &'a Dense<T>,
    chi_l: usize,
    d_i: usize,
    d_ip1: usize,
    chi_r: usize,
    backend: Arc<B>,
}

impl<'a, T: Scalar, B: ComputeBackend> EffectiveHamiltonian2Site<'a, T, B> {
    /// Construct directly from env / MPO references plus the bond
    /// dimensions.
    ///
    /// Validates the shapes / bond dimensions in debug builds via
    /// `debug_assert!` so callers that bypass the standard
    /// [`dmrg_2site_step`] entry point still catch obvious mismatches
    /// in tests. Release builds skip the asserts and trust the
    /// caller; the matvec body uses `.expect` and would panic on
    /// genuinely inconsistent shapes — `dmrg_2site_step` performs
    /// the same validation up front via fallible `Result` returns.
    ///
    /// Exposed so callers that want to drive a different solver
    /// (e.g., ARPACK behind the feature gate) can build the operator
    /// manually instead of going through `dmrg_2site_step`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left: &'a Dense<T>,
        w_i: &'a Dense<T>,
        w_ip1: &'a Dense<T>,
        right: &'a Dense<T>,
        chi_l: usize,
        d_i: usize,
        d_ip1: usize,
        chi_r: usize,
        backend: Arc<B>,
    ) -> Self {
        debug_assert!(
            chi_l > 0 && d_i > 0 && d_ip1 > 0 && chi_r > 0,
            "heff: dims must be > 0"
        );
        debug_assert_eq!(
            left.shape(),
            &[chi_l, w_i.shape()[0], chi_l],
            "left env shape"
        );
        debug_assert_eq!(
            right.shape(),
            &[chi_r, w_ip1.shape()[3], chi_r],
            "right env shape"
        );
        debug_assert_eq!(w_i.shape()[1], d_i, "W[i] d_ket / d_i");
        debug_assert_eq!(w_i.shape()[2], d_i, "W[i] d_bra / d_i");
        debug_assert_eq!(w_ip1.shape()[1], d_ip1, "W[i+1] d_ket / d_ip1");
        debug_assert_eq!(w_ip1.shape()[2], d_ip1, "W[i+1] d_bra / d_ip1");
        debug_assert_eq!(w_i.shape()[3], w_ip1.shape()[0], "W bond W_mid agreement");
        Self {
            left,
            w_i,
            w_ip1,
            right,
            chi_l,
            d_i,
            d_ip1,
            chi_r,
            backend,
        }
    }

    /// Linear-operator vector dimension.
    ///
    /// Matches `chi_l * d_i * d_{i+1} * chi_r` — the size of the
    /// flattened 2-site block.
    pub fn dim(&self) -> usize {
        self.chi_l * self.d_i * self.d_ip1 * self.chi_r
    }

    /// Left bond dimension `chi_l` (the first axis of the 2-site
    /// block).
    pub fn chi_l(&self) -> usize {
        self.chi_l
    }

    /// Physical dimension at site `i`.
    pub fn d_i(&self) -> usize {
        self.d_i
    }

    /// Physical dimension at site `i + 1`.
    pub fn d_ip1(&self) -> usize {
        self.d_ip1
    }

    /// Right bond dimension `chi_r` (the last axis of the 2-site
    /// block).
    pub fn chi_r(&self) -> usize {
        self.chi_r
    }
}

impl<'a, T: Scalar, B: ComputeBackend> LinearOp<T> for EffectiveHamiltonian2Site<'a, T, B> {
    fn apply(&self, v: &Dense<T>) -> Dense<T> {
        // Shapes are pre-validated by `dmrg_2site_step` before this
        // operator is constructed, so the contractions cannot fail.
        let psi = v.reshape(vec![self.chi_l, self.d_i, self.d_ip1, self.chi_r]);
        let backend = &*self.backend;

        // left[a,b,c] · psi[c,i,j,f] -> tmp1[a,b,i,j,f]
        let tmp1 = contract(backend, self.left, &psi, "abc,cijf->abijf")
            .expect("heff matvec step 1: shape pre-validated");
        // tmp1[a,b,i,j,f] · W[i][b,i,s,m] -> tmp2[a,s,m,j,f]
        let tmp2 = contract(backend, &tmp1, self.w_i, "abijf,bism->asmjf")
            .expect("heff matvec step 2: shape pre-validated");
        // tmp2[a,s,m,j,f] · W[i+1][m,j,t,g] -> tmp3[a,s,t,g,f]
        let tmp3 = contract(backend, &tmp2, self.w_ip1, "asmjf,mjtg->astgf")
            .expect("heff matvec step 3: shape pre-validated");
        // tmp3[a,s,t,g,f] · right[h,g,f] -> out[a,s,t,h]
        let out = contract(backend, &tmp3, self.right, "astgf,hgf->asth")
            .expect("heff matvec step 4: shape pre-validated");

        out.reshape(vec![self.dim()])
    }
}

/// Result of a single 2-site DMRG step: the smallest local
/// eigenpair plus the truncated-SVD split of its eigenvector.
///
/// `u` and `vt` are returned **separately from `s`**. Phase 3 does
/// not pick a sweep direction; the sweep driver (Phase 4) will
/// absorb `s` into whichever side moves.
#[derive(Debug, Clone)]
pub struct TwoSiteStepResult<T: Scalar> {
    pub eigenvalue: T::Real,
    pub residual: T::Real,
    pub iters: usize,
    /// `true` iff the Lanczos true-residual test fell at or below
    /// `LanczosParams::tol`. On `false`, callers receive the
    /// best-effort eigenpair plus its split — they decide whether to
    /// accept it, retry with looser params, or abort.
    pub converged: bool,
    /// Left singular vectors, shape `[chi_l, d_i, chi_new]`.
    /// Left-canonical at axes `(chi_l, d_i)` — i.e. `U^† U =
    /// I_{chi_new}`.
    pub u: Dense<T>,
    /// Singular values, shape `[chi_new]`, descending.
    pub s: Dense<T::Real>,
    /// Right singular vectors, shape `[chi_new, d_{i+1}, chi_r]`.
    /// Right-canonical at axes `(d_{i+1}, chi_r)` — i.e. `Vt Vt^† =
    /// I_{chi_new}`.
    pub vt: Dense<T>,
    /// Frobenius norm of the discarded singular values.
    pub trunc_err: T::Real,
}

/// Run a single 2-site DMRG step at sites `(site, site+1)`.
///
/// Reads `envs.left(site)` and `envs.right(site + 2)`. Builds the
/// local effective Hamiltonian, drives Lanczos to the smallest
/// eigenpair, then splits the optimized two-site block via
/// `arnet_linalg::trunc_svd` with `nrow = 2` (grouping `(chi_l,
/// d_i)` as rows). The function does **not** mutate the envs or
/// the MPS — caller (sweep driver) advances the envs separately.
///
/// # Errors
///
/// - [`DmrgHeffError::InvalidSite`] if `site + 1 >= n_sites`.
/// - [`DmrgHeffError::LengthMismatch`] if MPS / MPO / envs disagree.
/// - [`DmrgHeffError::StaleEnv`] if `left(site)` or `right(site +
///   2)` is `None`.
/// - [`DmrgHeffError::Contract`] propagates an underlying linalg
///   failure from the truncated SVD step.
pub fn dmrg_2site_step<T, B>(
    envs: &DmrgEnvs<T, B>,
    mps: &Mps<Dense<T>, B>,
    mpo: &Mpo<Dense<T>, B>,
    site: usize,
    params: &LanczosParams,
    trunc: &TruncSvdParams,
) -> Result<TwoSiteStepResult<T>, DmrgHeffError>
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
    B: ComputeBackend,
{
    // ---- Length / index validation ------------------------------
    let n_sites = envs.n_sites();
    if mps.len() != n_sites || mpo.len() != n_sites {
        return Err(DmrgHeffError::LengthMismatch {
            mps: mps.len(),
            mpo: mpo.len(),
            envs: n_sites,
        });
    }
    if site + 1 >= n_sites {
        return Err(DmrgHeffError::InvalidSite { site, n_sites });
    }

    // ---- Env slot Some-check ------------------------------------
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

    // ---- Shape derivation + cross-check -------------------------
    // env shape (top-bra, W, bot-ket) with bra=ket → both bond axes share dim.
    let chi_l = left.shape()[0];
    let chi_r = right.shape()[0];
    let d_i = mps_i.shape()[1];
    let d_ip1 = mps_ip1.shape()[1];

    // Pin all the bond / physical dimensions reached into during the
    // matvec. These run unconditionally (release builds included) so
    // the matvec body's `.expect("shape pre-validated")` calls cannot
    // fire on dimension mismatches. The validation also covers the
    // bra-physical (axis 2) leg of each MPO site, which would
    // otherwise propagate silently through `apply` if `axis 1 ==
    // axis 2` happened to differ.
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
    if left.shape().len() != 3 {
        return Err(DmrgHeffError::ShapeMismatch {
            site,
            field: "left.rank",
            expected: 3,
            actual: left.shape().len(),
        });
    }
    if right.shape().len() != 3 {
        return Err(DmrgHeffError::ShapeMismatch {
            site,
            field: "right.rank",
            expected: 3,
            actual: right.shape().len(),
        });
    }
    if w_i.shape().len() != 4 {
        return Err(DmrgHeffError::ShapeMismatch {
            site,
            field: "W[i].rank",
            expected: 4,
            actual: w_i.shape().len(),
        });
    }
    if w_ip1.shape().len() != 4 {
        return Err(DmrgHeffError::ShapeMismatch {
            site,
            field: "W[i+1].rank",
            expected: 4,
            actual: w_ip1.shape().len(),
        });
    }
    if mps_i.shape().len() != 3 {
        return Err(DmrgHeffError::ShapeMismatch {
            site,
            field: "MPS[i].rank",
            expected: 3,
            actual: mps_i.shape().len(),
        });
    }
    if mps_ip1.shape().len() != 3 {
        return Err(DmrgHeffError::ShapeMismatch {
            site,
            field: "MPS[i+1].rank",
            expected: 3,
            actual: mps_ip1.shape().len(),
        });
    }
    check_eq(
        left.shape()[2],
        mps_i.shape()[0],
        "left.bot_ket vs MPS[i].left_bond",
    )?;
    check_eq(
        left.shape()[2],
        chi_l,
        "left.bot_ket vs left.top_bra (bra=ket)",
    )?;
    check_eq(
        right.shape()[2],
        mps_ip1.shape()[2],
        "right.bot_ket vs MPS[i+1].right_bond",
    )?;
    check_eq(
        right.shape()[2],
        chi_r,
        "right.bot_ket vs right.top_bra (bra=ket)",
    )?;
    check_eq(left.shape()[1], w_i.shape()[0], "left.W_bond vs W[i].W_l")?;
    check_eq(
        right.shape()[1],
        w_ip1.shape()[3],
        "right.W_bond vs W[i+1].W_r",
    )?;
    check_eq(
        w_i.shape()[3],
        w_ip1.shape()[0],
        "W[i].W_r vs W[i+1].W_l (W_mid)",
    )?;
    check_eq(w_i.shape()[1], d_i, "W[i].d_ket vs MPS[i].physical")?;
    check_eq(w_i.shape()[2], d_i, "W[i].d_bra vs MPS[i].physical")?;
    check_eq(w_ip1.shape()[1], d_ip1, "W[i+1].d_ket vs MPS[i+1].physical")?;
    check_eq(w_ip1.shape()[2], d_ip1, "W[i+1].d_bra vs MPS[i+1].physical")?;

    let heff = EffectiveHamiltonian2Site::new(
        left,
        w_i,
        w_ip1,
        right,
        chi_l,
        d_i,
        d_ip1,
        chi_r,
        Arc::clone(&backend),
    );

    // ---- Drive Lanczos ------------------------------------------
    let dim = heff.dim();
    let lan = lanczos_smallest::<T, _>(&heff, dim, params);

    // ---- Truncated SVD split ------------------------------------
    let psi_4d = lan.eigenvector.reshape(vec![chi_l, d_i, d_ip1, chi_r]);
    let (u_2d, s, vt_2d, trunc_err) = trunc_svd(&*backend, &psi_4d, 2, trunc)?;

    let chi_new = u_2d.shape()[1];
    debug_assert_eq!(vt_2d.shape()[0], chi_new, "U/Vt new bond dim agreement");

    let order = backend.preferred_order();
    let rm = MemoryOrder::RowMajor;
    let reshape_to_3d = |t_2d: Dense<T>, new_shape: Vec<usize>| -> Dense<T> {
        // 2D backend-order → RM → multi-dim split → backend-order.
        // Mirrors the pattern in `arnet_mps::truncate::truncate_dense`.
        let rm_view = reorder(&t_2d, order, rm);
        let multi = rm_view.reshape(new_shape);
        reorder(&multi, rm, order)
    };
    let u = reshape_to_3d(u_2d, vec![chi_l, d_i, chi_new]);
    let vt = reshape_to_3d(vt_2d, vec![chi_new, d_ip1, chi_r]);

    Ok(TwoSiteStepResult {
        eigenvalue: lan.eigenvalue,
        residual: lan.residual,
        iters: lan.iters,
        converged: lan.converged,
        u,
        s,
        vt,
        trunc_err,
    })
}
