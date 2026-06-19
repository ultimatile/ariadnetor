//! 2-site DMRG local update: build the effective Hamiltonian
//! operator from `(left(i), W[i], W[i+1], right(i+2))`, drive the
//! selected local eigensolver (Lanczos by default, ARPACK behind
//! the `arpack` feature) to the smallest eigenpair, and split the
//! optimized two-site block back into a left-canonical /
//! right-canonical pair via truncated SVD.
//!
//! Axis convention (consistent with [`super::env`] and the
//! `arnet_mps::inner` braket family):
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
//! implements [`LinearOp<T>`] so the local matvec can drive either
//! Krylov solver in [`crate::krylov`] without materializing `H_eff` as
//! a dense matrix. The matvec's contractions run on the [`Host`]
//! substrate (DMRG is host-resident in the CPU-only Stage B scope), so
//! the operator carries no backend and obtains one from [`Host::shared`]
//! per `apply`.

use arnet_core::Scalar;
use arnet_linalg::{TruncSvdParams, contract_with_backend, trunc_svd};
use arnet_mps::{Mpo, Mps, TensorChain};
use arnet_tensor::{DenseTensor, Host};

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
/// selected local eigensolver via [`LinearOp`].
#[derive(Debug, Clone)]
pub struct EffectiveHamiltonian2Site<'a, T: Scalar> {
    left: &'a DenseTensor<T>,
    w_i: &'a DenseTensor<T>,
    w_ip1: &'a DenseTensor<T>,
    right: &'a DenseTensor<T>,
    chi_l: usize,
    d_i: usize,
    d_ip1: usize,
    chi_r: usize,
}

impl<'a, T: Scalar> EffectiveHamiltonian2Site<'a, T> {
    /// Construct directly from env / MPO references plus the bond
    /// dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left: &'a DenseTensor<T>,
        w_i: &'a DenseTensor<T>,
        w_ip1: &'a DenseTensor<T>,
        right: &'a DenseTensor<T>,
        chi_l: usize,
        d_i: usize,
        d_ip1: usize,
        chi_r: usize,
    ) -> Self {
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
        }
    }

    /// Linear-operator vector dimension.
    pub fn dim(&self) -> usize {
        self.chi_l * self.d_i * self.d_ip1 * self.chi_r
    }

    pub fn chi_l(&self) -> usize {
        self.chi_l
    }

    pub fn d_i(&self) -> usize {
        self.d_i
    }

    pub fn d_ip1(&self) -> usize {
        self.d_ip1
    }

    pub fn chi_r(&self) -> usize {
        self.chi_r
    }
}

impl<'a, T: Scalar> LinearOp<T> for EffectiveHamiltonian2Site<'a, T> {
    fn apply(&self, v: &DenseTensor<T>) -> DenseTensor<T> {
        // The matvec runs on the host substrate; both the Krylov-side
        // input and the returned output are host-resident dense vectors,
        // so no backend relabel is needed — the contractions dispatch
        // directly through `Host`.
        let backend = Host::shared();
        let psi = v.reshape(vec![self.chi_l, self.d_i, self.d_ip1, self.chi_r]);

        let tmp1 = contract_with_backend(backend.as_ref(), self.left, &psi, "abc,cijf->abijf")
            .expect("heff matvec step 1: shape pre-validated");
        let tmp2 = contract_with_backend(backend.as_ref(), &tmp1, self.w_i, "abijf,bism->asmjf")
            .expect("heff matvec step 2: shape pre-validated");
        let tmp3 = contract_with_backend(backend.as_ref(), &tmp2, self.w_ip1, "asmjf,mjtg->astgf")
            .expect("heff matvec step 3: shape pre-validated");
        let out = contract_with_backend(backend.as_ref(), &tmp3, self.right, "astgf,hgf->asth")
            .expect("heff matvec step 4: shape pre-validated");

        out.reshape(vec![self.dim()])
    }
}

/// Result of a single 2-site DMRG step.
#[derive(Debug, Clone)]
pub struct TwoSiteStepResult<T: Scalar> {
    pub eigenvalue: T::Real,
    pub residual: T::Real,
    pub iters: usize,
    pub converged: bool,
    /// Left singular vectors, shape `[chi_l, d_i, chi_new]`.
    pub u: DenseTensor<T>,
    /// Singular values, shape `[chi_new]`, descending.
    pub s: DenseTensor<T::Real>,
    /// Right singular vectors, shape `[chi_new, d_{i+1}, chi_r]`.
    pub vt: DenseTensor<T>,
    /// Frobenius norm of the discarded singular values.
    pub trunc_err: T::Real,
}

/// Run a single 2-site DMRG step at sites `(site, site+1)`.
pub fn dmrg_2site_step<T>(
    envs: &DmrgEnvs<arnet_tensor::DenseStorage<T>, arnet_tensor::DenseLayout>,
    mps: &Mps<arnet_tensor::DenseStorage<T>, arnet_tensor::DenseLayout>,
    mpo: &Mpo<arnet_tensor::DenseStorage<T>, arnet_tensor::DenseLayout>,
    site: usize,
    eigensolver: &LocalEigensolverParams,
    trunc: &TruncSvdParams,
) -> Result<TwoSiteStepResult<T>, DmrgHeffError>
where
    T: DmrgScalar,
    T::Real: Scalar<Real = T::Real>,
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
    let w_i = mpo.site(site);
    let w_ip1 = mpo.site(site + 1);
    let mps_i = mps.site(site);
    let mps_ip1 = mps.site(site + 1);

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

    let chi_l = left.shape()[0];
    let chi_r = right.shape()[0];
    let d_i = mps_i.shape()[1];
    let d_ip1 = mps_ip1.shape()[1];

    check_at_least(1, chi_l, "chi_l (left bond)")?;
    check_at_least(1, chi_r, "chi_r (right bond)")?;
    check_at_least(1, d_i, "d_i (MPS[i] physical)")?;
    check_at_least(1, d_ip1, "d_ip1 (MPS[i+1] physical)")?;
    check_at_least(1, w_i.shape()[0], "W[i].W_l")?;
    check_at_least(1, w_i.shape()[3], "W[i].W_r (= W_mid)")?;
    check_at_least(1, w_ip1.shape()[3], "W[i+1].W_r")?;

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

    let heff = EffectiveHamiltonian2Site::new(left, w_i, w_ip1, right, chi_l, d_i, d_ip1, chi_r);

    let dim = heff.dim();
    // Lanczos/ARPACK return a host-resident `DenseTensor<T>`; it feeds
    // straight into the truncated SVD with no backend relabel.
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
            (
                res.eigenvalue,
                res.eigenvector,
                res.iters,
                true,
                res.residual,
            )
        }
    };

    let psi_4d = eigenvector.reshape(vec![chi_l, d_i, d_ip1, chi_r]);
    let (u_2d, s, vt_2d, trunc_err) = trunc_svd(Host::shared().as_ref(), &psi_4d, 2, trunc)?;

    let chi_new = u_2d.shape()[1];
    debug_assert_eq!(vt_2d.shape()[0], chi_new, "U/Vt new bond dim agreement");

    // Split the SVD factors' fused legs back into site shape:
    // U (chi_l*d_i, chi_new) -> (chi_l, d_i, chi_new),
    // Vt (chi_new, d_ip1*chi_r) -> (chi_new, d_ip1, chi_r).
    let u = u_2d.split_leg(0, &[chi_l, d_i]);
    let vt = vt_2d.split_leg(1, &[d_ip1, chi_r]);

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
