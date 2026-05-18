//! 2-site DMRG local update: build the effective Hamiltonian
//! operator from `(left(i), W[i], W[i+1], right(i+2))`, drive the
//! selected local eigensolver (Lanczos by default, ARPACK behind
//! the `arpack` feature) to the smallest eigenpair, and split the
//! optimized two-site block back into a left-canonical /
//! right-canonical pair via truncated SVD.
//!
//! Axis convention (consistent with [`super::env`] and the
//! `arnet_mps::inner_repr` braket family — `braket_dense` for [`Dense<T>`],
//! `braket_bsp` for `BlockSparse<T, S>`):
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
//! implements [`LinearOp`] so the local matvec can drive either
//! Krylov solver in [`crate::krylov`] without materializing `H_eff`
//! as a dense matrix. [`dmrg_2site_step`] dispatches over
//! [`super::LocalEigensolverParams`] to that solver and then through
//! `arnet_linalg::trunc_svd`, returning the U / S / Vt factors
//! separately so callers (e.g. the sweep driver) can decide which
//! direction absorbs S.

use std::sync::Arc;

use arnet_core::backend::ComputeBackend;
use arnet_core::{MemoryOrder, Scalar};
use arnet_linalg::{TruncSvdParams, contract_dense as contract, trunc_svd_dense as trunc_svd};
use arnet_mps::{MpoRepr as Mpo, MpsRepr as Mps, TensorChainRepr as TensorChain};
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, reorder};

#[cfg(feature = "arpack")]
use crate::krylov::arpack_smallest;
use crate::krylov::{LinearOp, lanczos_smallest};

use super::env::DmrgEnvs;
use super::heff_error::DmrgHeffError;
use super::solver::{
    DmrgScalar, LocalEigensolverParams, eigensolver_tol, validate_eigensolver_params,
};

/// Effective Hamiltonian operator for the 2-site DMRG block at sites
/// `(i, i+1)`. Built once per local update and consumed by the
/// selected local eigensolver via [`LinearOp`] — Lanczos by default,
/// ARPACK behind the `arpack` feature.
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
        // Rank checks first so a malformed tensor surfaces a useful
        // assertion message rather than panicking on `shape()[N]`.
        debug_assert_eq!(left.shape().len(), 3, "left.rank == 3");
        debug_assert_eq!(right.shape().len(), 3, "right.rank == 3");
        debug_assert_eq!(w_i.shape().len(), 4, "W[i].rank == 4");
        debug_assert_eq!(w_ip1.shape().len(), 4, "W[i+1].rank == 4");
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
/// `u` and `vt` are returned **separately from `s`**. This function
/// does not pick a sweep direction; the caller absorbs `s` into
/// whichever side moves.
#[derive(Debug, Clone)]
pub struct TwoSiteStepResult<T: Scalar> {
    pub eigenvalue: T::Real,
    pub residual: T::Real,
    pub iters: usize,
    /// `true` iff the local eigensolver succeeded — Lanczos by its
    /// absolute true-residual test against `LanczosParams::tol`,
    /// ARPACK by its relative-tol stopping criterion (i.e. `Ok`
    /// return from `arpack_smallest`). The two arms intentionally
    /// disagree on what they call "converged": Lanczos uses the
    /// absolute residual; ARPACK uses `residual <= tol * |lambda|`.
    /// On `false`, callers receive the best-effort eigenpair plus
    /// its split — they decide whether to accept it, retry with
    /// looser params, or abort.
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
/// local effective Hamiltonian, drives the selected local
/// eigensolver (per [`LocalEigensolverParams`] — Lanczos by default,
/// ARPACK behind the `arpack` feature) to the smallest eigenpair,
/// then splits the optimized two-site block via
/// `arnet_linalg::trunc_svd` with `nrow = 2` (grouping `(chi_l,
/// d_i)` as rows). The function does **not** mutate the envs or
/// the MPS — caller (sweep driver) advances the envs separately.
///
/// # Errors
///
/// - [`DmrgHeffError::InvalidSite`] if `site + 1 >= n_sites` (the
///   `+1` is computed via `saturating_sub` so `site = usize::MAX`
///   does not overflow).
/// - [`DmrgHeffError::LengthMismatch`] if MPS / MPO / envs disagree.
/// - [`DmrgHeffError::StaleEnv`] if `left(site)` or `right(site +
///   2)` is `None`.
/// - [`DmrgHeffError::ShapeMismatch`] if any tensor rank, bond, or
///   physical dimension reached by the matvec does not match its
///   neighbours, or if a dimension is zero (the local-eigensolver
///   `dim` precondition and `contract` both reject zero-length
///   axes).
/// - [`DmrgHeffError::InvalidEigensolverParams`] if the selected
///   eigensolver variant's params violate their preconditions
///   (e.g. `max_iter == 0`, non-finite tol, Lanczos tol < 0,
///   ARPACK tol <= 0). These would otherwise trip the underlying
///   solver's internal asserts / upstream errors.
/// - [`DmrgHeffError::Contract`] propagates an underlying linalg
///   failure from the truncated SVD step.
///
/// Backend / allocation failures during the matvec body itself
/// (`LinalgError::Backend` from a downstream GEMM / transpose) still
/// propagate as a panic. Closing that path requires making the
/// Krylov [`crate::krylov::LinearOp`] trait fallible, which is a
/// cross-cutting change deferred to a later phase; the standard
/// preconditions for `contract` are validated at the entry, so this
/// branch is only reachable on genuine backend / allocation
/// failures rather than user input.
pub fn dmrg_2site_step<T, B>(
    envs: &DmrgEnvs<Dense<T>, B>,
    mps: &Mps<Dense<T>, B>,
    mpo: &Mpo<Dense<T>, B>,
    site: usize,
    eigensolver: &LocalEigensolverParams,
    trunc: &TruncSvdParams,
) -> Result<TwoSiteStepResult<T>, DmrgHeffError>
where
    T: DmrgScalar,
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
    // Use saturating_sub so `site = usize::MAX` does not overflow the
    // `site + 1` check. `n_sites < 2` (valid 2-site step requires at
    // least 2 sites) collapses to "every site is invalid" thanks to
    // saturating_sub returning 0.
    if site >= n_sites.saturating_sub(1) {
        return Err(DmrgHeffError::InvalidSite { site, n_sites });
    }

    // ---- Local-eigensolver params sanity ------------------------
    // Mirrors the underlying solver's internal asserts / upstream
    // errors so callers see `Err`, not a panic. The structural
    // checks (max_iter, tol-finite, tol-sign) are factored into
    // `validate_eigensolver_params`; the `T::Real` representability
    // check stays here because it depends on the storage element.
    validate_eigensolver_params(eigensolver)
        .map_err(|detail| DmrgHeffError::InvalidEigensolverParams { detail })?;
    if crate::numeric::try_real_from_f64::<T>(eigensolver_tol(eigensolver)).is_none() {
        return Err(DmrgHeffError::InvalidEigensolverParams {
            detail: "tol is not representable in T::Real",
        });
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

    // ---- Rank checks first (must precede any `shape()[N]` access)
    // so an unexpectedly-ranked tensor surfaces a `ShapeMismatch`
    // instead of an out-of-bounds panic.
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
    check_eq(3, left.shape().len(), "left.rank")?;
    check_eq(3, right.shape().len(), "right.rank")?;
    check_eq(4, w_i.shape().len(), "W[i].rank")?;
    check_eq(4, w_ip1.shape().len(), "W[i+1].rank")?;
    check_eq(3, mps_i.shape().len(), "MPS[i].rank")?;
    check_eq(3, mps_ip1.shape().len(), "MPS[i+1].rank")?;

    // ---- Shape derivation + cross-check -------------------------
    // env shape (top-bra, W, bot-ket) with bra=ket → both bond axes share dim.
    let chi_l = left.shape()[0];
    let chi_r = right.shape()[0];
    let d_i = mps_i.shape()[1];
    let d_ip1 = mps_ip1.shape()[1];

    // Reject zero-sized bond / physical dims explicitly. The local
    // eigensolver asserts `dim >= 1` (Lanczos panics, ARPACK rejects
    // with `InvalidParam`) and contract refuses zero-length axes, so
    // surface this here as `ShapeMismatch { expected: 1, .. }`
    // instead of letting it reach the operator.
    check_at_least(1, chi_l, "chi_l (left bond)")?;
    check_at_least(1, chi_r, "chi_r (right bond)")?;
    check_at_least(1, d_i, "d_i (MPS[i] physical)")?;
    check_at_least(1, d_ip1, "d_ip1 (MPS[i+1] physical)")?;
    check_at_least(1, w_i.shape()[0], "W[i].W_l")?;
    check_at_least(1, w_i.shape()[3], "W[i].W_r (= W_mid)")?;
    check_at_least(1, w_ip1.shape()[3], "W[i+1].W_r")?;

    // Pin all the bond / physical dimensions reached into during the
    // matvec. These run unconditionally (release builds included) so
    // the matvec body's `.expect("shape pre-validated")` calls cannot
    // fire on dimension mismatches. The validation also covers the
    // bra-physical (axis 2) leg of each MPO site, which would
    // otherwise propagate silently through `apply` if `axis 1 ==
    // axis 2` happened to differ.
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

    // ---- Drive the local eigensolver ----------------------------
    let dim = heff.dim();
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
            // For step-level convergence, treat any `Ok` return from
            // ARPACK as success: ARPACK has met its relative-tol
            // stopping criterion. `ArpackResult.converged` is a
            // stricter *absolute*-tol divergence indicator (true iff
            // `residual <= tol` with `tol` interpreted in absolute
            // units), and is structurally hard to satisfy at tight
            // `tol` even on successful runs because ARPACK's internal
            // criterion is `residual <= tol * |lambda|`. Propagating
            // it as the step-level flag would prevent
            // `DmrgResult.converged = true` for ARPACK-backed sweeps
            // even when the energy has stabilized; the absolute
            // residual is still surfaced via `residual` for callers
            // that want it.
            (
                res.eigenvalue,
                res.eigenvector,
                res.iters,
                true,
                res.residual,
            )
        }
    };

    // ---- Truncated SVD split ------------------------------------
    let psi_4d = eigenvector.reshape(vec![chi_l, d_i, d_ip1, chi_r]);
    let (u_2d, s, vt_2d, trunc_err) = trunc_svd(&*backend, &psi_4d, 2, trunc)?;

    let chi_new = u_2d.shape()[1];
    debug_assert_eq!(vt_2d.shape()[0], chi_new, "U/Vt new bond dim agreement");

    let order = backend.preferred_order();
    let rm = MemoryOrder::RowMajor;
    let reshape_to_3d = |t_2d: Dense<T>, new_shape: Vec<usize>| -> Dense<T> {
        // 2D backend-order → RM → multi-dim split → backend-order.
        // Mirrors the pattern in `arnet_mps::truncate_repr::truncate_dense`.
        let rm_view = reorder(&t_2d, order, rm);
        let multi = rm_view.reshape(new_shape);
        reorder(&multi, rm, order)
    };
    let u = reshape_to_3d(u_2d, vec![chi_l, d_i, chi_new]);
    let vt = reshape_to_3d(vt_2d, vec![chi_new, d_ip1, chi_r]);

    Ok(TwoSiteStepResult {
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
