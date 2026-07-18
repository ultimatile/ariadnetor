//! Variational (fit) MPO-MPS application (Dense and BlockSparse).
//!
//! Fits `φ ≈ Wψ` at a fixed bond by single-site DMRG-style sweeps. The local
//! update replaces the orthogonality-center tensor with the `⟨φ|W|ψ⟩`
//! environment projection `P_j = L(j)·W_j·ψ_j·R(j+1)`; because the off-center
//! sites are isometric, `P_j` is the exact per-site minimizer of `‖φ − Wψ‖²`
//! (the normal equations reduce to `φ_j = P_j`). The bond is fixed by the
//! initial guess (zip-up or density-matrix, truncated to `chi_max`); the sweeps
//! refine the tensors without changing it.
//!
//! Host-pinned: the three-layer `⟨φ|W|ψ⟩` environment is the host-resident
//! [`BraketEnvs`](crate::env::BraketEnvs) primitive, so — like DMRG — the whole
//! computation runs on the [`Host`] substrate and the caller's backend is not
//! consulted. The single-site center moves reuse the state-preserving
//! [`left_qr_step`](crate::canonicalize::left_qr_step) /
//! [`right_lq_step`](crate::canonicalize::right_lq_step) gauge steps, which keep
//! the swept state invariant so the per-cycle objective is monotone.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{TruncSvdParams, contract, tensordot};
use ariadnetor_tensor::{
    BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, DenseLayout, DenseStorage,
    DenseTensor, Host, OpsFor, Sector,
};
use num_traits::{Float, Zero};

use crate::canonicalize::{
    canonicalize_bsp, canonicalize_dense, left_qr_step, left_qr_step_bsp, right_lq_step,
    right_lq_step_bsp,
};
use crate::chain::TensorChain;
use crate::env::BraketEnvs;
use crate::types::{CanonicalForm, Mpo, Mps, TruncateParams, VariationalInit};

/// Truncation parameters for the initial-guess seed: `chi_max` only.
///
/// The variational method fixes the bond at the seed's `chi_max` and defers
/// `target_trunc_err` (a state-domain cutoff) as a bond criterion, so it is
/// stripped here before seeding — otherwise the zip-up seed would consult it
/// (the density-matrix seed already drops it), making the bond depend on a
/// parameter the method documents as not consulted. `absorb` / `center` are
/// left at their defaults; the seed generators ignore them.
fn seed_params(params: Option<&TruncateParams>) -> Option<TruncateParams> {
    params.map(|p| {
        TruncateParams::from(TruncSvdParams {
            chi_max: p.svd.chi_max,
            target_trunc_err: None,
        })
    })
}

/// Relative-change convergence test on the center overlap `‖P_center‖²`.
///
/// Returns `true` when `|current − prev| ≤ tol · current` (falling back to
/// `≤ tol` when `current ≤ 0`, guarding the degenerate zero-overlap case) — a
/// relative tolerance, because `⟨Wψ|Wψ⟩` (the value the overlap converges to)
/// is unnormalized and scale-varying, so an absolute tolerance would be
/// scale-dependent.
fn converged<R: Float>(prev: R, current: R, tol: R) -> bool {
    let denom = if current > R::zero() {
        current
    } else {
        R::one()
    };
    ((current - prev).abs() / denom) <= tol
}

// ===========================================================================
// Dense
// ===========================================================================

/// Single-site `⟨φ|W|ψ⟩` projection for the Dense path: contract the left env
/// `L(j)` `(φ_l, w_l, ψ_l)`, the ket site `ψ_j` `(ψ_l, d_ket, ψ_r)`, the MPO
/// site `W_j` `(w_l, d_ket, d_bra, w_r)`, and the right env `R(j+1)`
/// `(φ_r, w_r, ψ_r)` into the new center tensor `(φ_l, d_bra, φ_r)`.
fn project_dense<T, B>(
    envs: &BraketEnvs<DenseStorage<T>, DenseLayout>,
    op: &Mpo<DenseStorage<T>, DenseLayout>,
    psi: &Mps<DenseStorage<T>, DenseLayout>,
    j: usize,
    backend: &B,
) -> DenseTensor<T>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    let l = envs
        .left(j)
        .expect("left env present: sweep advance ordering guarantees it");
    let r = envs
        .right(j + 1)
        .expect("right env present: sweep advance ordering guarantees it");
    // L(abc)·ψ(cde) → (abde) = (φ_l, w_l, d_ket, ψ_r)
    let t1 = contract(backend, l, psi.site(j), "abc,cde->abde")
        .expect("projection L·ψ: validated by entry point");
    // (abde)·W(bdfg) → (aefg) = (φ_l, ψ_r, d_bra, w_r)
    let t2 = contract(backend, &t1, op.site(j), "abde,bdfg->aefg")
        .expect("projection ·W: validated by entry point");
    // (aefg)·R(hge) → (afh) = (φ_l, d_bra, φ_r)
    contract(backend, &t2, r, "aefg,hge->afh").expect("projection ·R: validated by entry point")
}

/// Apply a Dense MPO to a Dense MPS via single-site variational fitting.
///
/// See the module docs for the algorithm. Host-pinned: `backend` is not
/// consulted (the computation runs on [`Host`]). The result is left in
/// `CanonicalForm::Mixed { center: 0 }`.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths, or if either is empty.
pub(crate) fn apply_variational_dense<T>(
    op: &Mpo<DenseStorage<T>, DenseLayout>,
    psi: &Mps<DenseStorage<T>, DenseLayout>,
    params: Option<&TruncateParams>,
    init: VariationalInit,
    max_sweeps: usize,
    tol: f64,
) -> Mps<DenseStorage<T>, DenseLayout>
where
    T: Scalar,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let backend = Host::shared();
    let h = backend.as_ref();

    // Initial guess; its chi_max truncation fixes the bond the sweeps refine.
    let sp = seed_params(params);
    let mut phi = match init {
        VariationalInit::ZipUp => super::apply_zipup_dense(h, op, psi, sp.as_ref()),
        VariationalInit::DensityMatrix => {
            super::apply_density_matrix_dense(h, op, psi, sp.as_ref())
        }
    };

    // A 1-site guess is already the exact product Wψ; nothing to sweep.
    if n == 1 {
        return phi;
    }

    // Validate the tolerance before the canonicalize + environment build, so an
    // unrepresentable tol fails fast rather than after that setup work; the cast
    // is independent of the environments.
    let tol_real: T::Real = ariadnetor_core::try_real_from_f64::<T>(tol)
        .expect("tol must be representable as a finite value in the scalar's real type");

    // Right-canonicalize so the L→R sweep starts against valid right envs.
    canonicalize_dense(h, &mut phi, 0);
    let mut envs = BraketEnvs::<DenseStorage<T>, DenseLayout>::build::<T>(&phi, op, psi)
        .expect("braket env build: validated by entry point");

    let mut last: Option<T::Real> = None;

    for _ in 0..max_sweeps {
        // L→R half-sweep: project each site, then move the center right.
        for j in 0..n {
            *phi.site_mut(j) = project_dense(&envs, op, psi, j, h);
            if j < n - 1 {
                left_qr_step(&mut phi, j, h);
                envs.advance_left::<T>(&phi, op, psi, j)
                    .expect("advance_left: validated by entry point");
            }
        }
        // R→L half-sweep. The turning site (n-1) was just projected at the end
        // of the L→R sweep against the same environments, so re-projecting it
        // here would reproduce the identical tensor — skip it and only move its
        // gauge left. The final projection (site 0) is the convergence probe,
        // so its norm is the only one needed.
        let mut center_norm = T::Real::zero();
        for j in (0..n).rev() {
            if j < n - 1 {
                let p = project_dense(&envs, op, psi, j, h);
                if j == 0 {
                    center_norm = p.norm();
                }
                *phi.site_mut(j) = p;
            }
            if j > 0 {
                right_lq_step(&mut phi, j, h);
                envs.advance_right::<T>(&phi, op, psi, j)
                    .expect("advance_right: validated by entry point");
            }
        }
        let overlap = center_norm * center_norm;
        if let Some(prev) = last
            && converged(prev, overlap, tol_real)
        {
            break;
        }
        last = Some(overlap);
    }

    phi.set_canonical_form(CanonicalForm::Mixed { center: 0 });
    phi
}

// ===========================================================================
// BlockSparse
// ===========================================================================

/// Single-site `⟨φ|W|ψ⟩` projection for the BlockSparse path. Mirrors
/// [`project_dense`] via `tensordot`: the block-sparse env legs carry
/// direction-flipped W / ket bonds, so the contracted axis pairs are opposite
/// in direction and the flux of `W_j` + `ψ_j` propagates onto the new center
/// tensor `[φ_l, d_bra, φ_r]`.
fn project_bsp<T, S, B>(
    envs: &BraketEnvs<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    j: usize,
    backend: &B,
) -> BlockSparseTensor<T, S>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    let l = envs
        .left(j)
        .expect("left env present: sweep advance ordering guarantees it");
    let r = envs
        .right(j + 1)
        .expect("right env present: sweep advance ordering guarantees it");
    // L[φ_l, w_l~, ψ_l~] · ψ_j[ψ_l, d_ket, ψ_r] over ψ_l → [φ_l, w_l~, d_ket, ψ_r]
    let s1 = tensordot(backend, l, psi.site(j), &[2], &[0])
        .expect("projection L·ψ: validated by entry point");
    // · W_j[w_l, d_ket, d_bra, w_r] over (w_l~, d_ket) → [φ_l, ψ_r, d_bra, w_r]
    let s2 = tensordot(backend, &s1, op.site(j), &[1, 2], &[0, 1])
        .expect("projection ·W: validated by entry point");
    // · R[φ_r, w_r~, ψ_r~] over (w_r, ψ_r) → [φ_l, d_bra, φ_r]
    tensordot(backend, &s2, r, &[3, 1], &[1, 2]).expect("projection ·R: validated by entry point")
}

/// Apply a BlockSparse MPO to a BlockSparse MPS via single-site variational
/// fitting.
///
/// See [`apply_variational_dense`] for the algorithm; the block-sparse variant
/// mirrors it via [`project_bsp`], `canonicalize_bsp`, and the block-sparse
/// gauge steps. Host-pinned.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths, or if either is empty.
pub(crate) fn apply_variational_bsp<T, S>(
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    params: Option<&TruncateParams>,
    init: VariationalInit,
    max_sweeps: usize,
    tol: f64,
) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    T: Scalar,
    S: Sector,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let backend = Host::shared();
    let h = backend.as_ref();

    let sp = seed_params(params);
    let mut phi = match init {
        VariationalInit::ZipUp => super::apply_zipup_bsp(h, op, psi, sp.as_ref()),
        VariationalInit::DensityMatrix => super::apply_density_matrix_bsp(h, op, psi, sp.as_ref()),
    };

    if n == 1 {
        return phi;
    }

    // Validate the tolerance before the canonicalize + environment build, so an
    // unrepresentable tol fails fast rather than after that setup work; the cast
    // is independent of the environments.
    let tol_real: T::Real = ariadnetor_core::try_real_from_f64::<T>(tol)
        .expect("tol must be representable as a finite value in the scalar's real type");

    canonicalize_bsp(h, &mut phi, 0);
    let mut envs =
        BraketEnvs::<BlockSparseStorage<T>, BlockSparseLayout<S>>::build::<T>(&phi, op, psi)
            .expect("braket env build: validated by entry point");

    let mut last: Option<T::Real> = None;

    for _ in 0..max_sweeps {
        for j in 0..n {
            *phi.site_mut(j) = project_bsp(&envs, op, psi, j, h);
            if j < n - 1 {
                left_qr_step_bsp(&mut phi, j, h);
                envs.advance_left::<T>(&phi, op, psi, j)
                    .expect("advance_left: validated by entry point");
            }
        }
        // See the Dense path: skip re-projecting the just-optimized turning
        // site; site 0's projection norm is the convergence probe.
        let mut center_norm = T::Real::zero();
        for j in (0..n).rev() {
            if j < n - 1 {
                let p = project_bsp(&envs, op, psi, j, h);
                if j == 0 {
                    center_norm = p.norm();
                }
                *phi.site_mut(j) = p;
            }
            if j > 0 {
                right_lq_step_bsp(&mut phi, j, h);
                envs.advance_right::<T>(&phi, op, psi, j)
                    .expect("advance_right: validated by entry point");
            }
        }
        let overlap = center_norm * center_norm;
        if let Some(prev) = last
            && converged(prev, overlap, tol_real)
        {
            break;
        }
        last = Some(overlap);
    }

    phi.set_canonical_form(CanonicalForm::Mixed { center: 0 });
    phi
}
