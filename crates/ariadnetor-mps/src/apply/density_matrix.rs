//! Density-matrix MPO-MPS application (Dense and BlockSparse).
//!
//! Materializes the untruncated product `φ = Wψ` via the parent module's
//! `local_product_*` helpers, accumulates the `⟨φ|φ⟩` right environments, then
//! a single left-to-right sweep that forms the reduced density matrix
//! `ρ = θ · R · θ†` at each bond and keeps its dominant `chi_max` eigenvectors.
//! Because `ρ` is Hermitian positive-semidefinite, its dominant eigenvectors
//! are its dominant left singular vectors, so the truncation reuses
//! [`trunc_svd`] (for a PSD matrix the SVD coincides with the
//! eigendecomposition).

use crate::chain::TensorChain;
use crate::types::{CanonicalForm, Mpo, Mps, TruncateParams};
use ariadnetor_core::Scalar;
use ariadnetor_linalg::{TruncSvdParams, contract, tensordot, trunc_svd};
use ariadnetor_tensor::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, DenseLayout,
    DenseStorage, DenseTensor, Direction, OpsFor, QNIndex, Sector,
};

/// Truncated-SVD parameters for a density-matrix sweep. Only `chi_max` is
/// consulted. `target_trunc_err` is deliberately dropped: `trunc_svd` applied to
/// `ρ` truncates in `ρ`'s eigenvalue domain — the *squared* Schmidt values, and
/// `trunc_svd`'s Frobenius cutoff squares them again — whereas a caller's
/// `target_trunc_err` means an error in the state's Schmidt (singular-value)
/// domain. Threading it through would truncate in the wrong metric; a
/// state-domain cutoff needs a dedicated truncated eigensolver (deferred).
fn density_matrix_svd_params(params: Option<&TruncateParams>) -> TruncSvdParams {
    TruncSvdParams {
        chi_max: params.and_then(|p| p.svd.chi_max),
        target_trunc_err: None,
    }
}

// ===========================================================================
// Dense
// ===========================================================================

/// Absorb Dense site `p` `(a, σ, b)` into the right `⟨φ|φ⟩` overlap
/// environment `r_next` `(b, b')`, returning the environment `(a, a')` that
/// now covers `p` and everything to its right.
fn absorb_right_env_dense<T, B>(
    p: &DenseTensor<T>,
    r_next: &DenseTensor<T>,
    backend: &B,
) -> DenseTensor<T>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    // tmp(a,σ,b') = Σ_b P(a,σ,b) R(b,b')
    let tmp = contract(backend, p, r_next, "asb,bd->asd")
        .expect("right-env carry: validated by entry point");
    // R'(a,a') = Σ_{σ,b'} tmp(a,σ,b') conj(P)(a',σ,b')
    let p_conj = p.conj();
    contract(backend, &tmp, &p_conj, "asd,esd->ae")
        .expect("right-env overlap: validated by entry point")
}

/// Apply a Dense MPO to a Dense MPS via the density-matrix algorithm.
///
/// Materializes the untruncated product `φ = Wψ` (per-site
/// [`local_product_dense`](super::local_product_dense), bond `w·χ`),
/// accumulates the `⟨φ|φ⟩` right environments, then a single left-to-right
/// sweep. At each non-final site it forms the reduced density matrix
/// `ρ = θ · R · θ†` — with `θ` the carry-absorbed site viewed as
/// `(χ_trunc·d, w_R·χ_R)` and `R` the right environment of the untruncated
/// remainder — and keeps its largest `chi_max` eigenvectors via [`trunc_svd`]
/// on the PSD `ρ`; the discarded `S·Vt` factor is dropped and the carry
/// `U† · θ` passes to the next site.
///
/// `params = None`, or `chi_max = None`, keeps full rank at every bond, making
/// the sweep lossless (matches the exact MPO-MPS product up to gauge and
/// roundoff). Only `chi_max` is consulted; `params.absorb`, `params.center`, and
/// `params.target_trunc_err` are not (see [`density_matrix_svd_params`] for why
/// `target_trunc_err` is dropped). The sweep intrinsically carries the
/// orthogonality center rightward and ends in `Mixed { center: n - 1 }`.
///
/// Forming `ρ = θ · R · θ†` squares the Schmidt spectrum, so Schmidt values
/// below roughly `√ε` relative to the largest are lost to `ρ`'s null space —
/// the standard density-matrix accuracy floor. Prefer
/// [`ZipUp`](crate::ApplyMethod::ZipUp), which decomposes the site directly,
/// when a bond carries a very wide Schmidt spectrum.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths, or if either is empty.
pub(crate) fn apply_density_matrix_dense<T, B>(
    backend: &B,
    op: &Mpo<DenseStorage<T>, DenseLayout>,
    psi: &Mps<DenseStorage<T>, DenseLayout>,
    params: Option<&TruncateParams>,
) -> Mps<DenseStorage<T>, DenseLayout>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let svd_params = density_matrix_svd_params(params);

    // Materialize φ = Wψ (untruncated, bond w·χ).
    let phi: Vec<DenseTensor<T>> = (0..n)
        .map(|j| super::local_product_dense(op.site(j), psi.site(j), backend))
        .collect();

    // Right ⟨φ|φ⟩ environments: envs[j] covers sites j..n-1; envs[n] is the
    // 1×1 boundary. Site j's sweep weights ρ by envs[j+1] (its right remainder).
    let mut envs: Vec<DenseTensor<T>> = vec![DenseTensor::ones(vec![1, 1]); n + 1];
    for j in (1..n).rev() {
        envs[j] = absorb_right_env_dense(&phi[j], &envs[j + 1], backend);
    }

    let mut tensors: Vec<DenseTensor<T>> = Vec::with_capacity(n);
    let mut carry: Option<DenseTensor<T>> = None;

    for j in 0..n {
        // θ = C · P_j, shape (χ_trunc, d, w_R·χ_R); C absent ⇒ θ = P_j.
        let theta = match carry.as_ref() {
            Some(c) => contract(backend, c, &phi[j], "ab,bde->ade")
                .expect("carry absorption: validated by entry point"),
            None => phi[j].clone(),
        };

        if j < n - 1 {
            let s = theta.shape();
            let (chi_t, d, right) = (s[0], s[1], s[2]);
            // View θ as the matrix (χ_trunc·d, w_R·χ_R).
            let theta2 = theta.reshape_logical(vec![chi_t * d, right]);
            let r_next = &envs[j + 1];

            // ρ = θ · R · θ†, shape (χ_trunc·d, χ_trunc·d), Hermitian PSD.
            let tr = contract(backend, &theta2, r_next, "xr,rs->xs")
                .expect("θ·R: validated by entry point");
            let theta2_conj = theta2.conj();
            let rho = contract(backend, &tr, &theta2_conj, "xs,ys->xy")
                .expect("ρ = θRθ†: validated by entry point");

            // trunc_svd of the PSD ρ: U's columns are ρ's dominant eigenvectors.
            let (u, _s, _vt, _err) = trunc_svd(backend, &rho, 1, &svd_params)
                .expect("trunc_svd: validated by entry point");
            // Split U's fused row leg back into (χ_trunc, d, k): the new site.
            tensors.push(u.split_leg(0, &[chi_t, d]));

            // Carry C' = U† · θ, shape (k, w_R·χ_R).
            let u_conj = u.conj();
            carry = Some(
                contract(backend, &u_conj, &theta2, "xk,xr->kr")
                    .expect("carry projection: validated by entry point"),
            );
        } else {
            // Final site is the orthogonality center; no truncation.
            tensors.push(theta);
        }
    }

    let mut result_mps: Mps<DenseStorage<T>, DenseLayout> = Mps::from_sites(tensors);
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });
    result_mps
}

// ===========================================================================
// BlockSparse
// ===========================================================================

/// Flip a block-sparse leg direction (Out ↔ In).
fn flip_dir(d: Direction) -> Direction {
    match d {
        Direction::Out => Direction::In,
        Direction::In => Direction::Out,
    }
}

/// Right `⟨φ|φ⟩` boundary environment for the block-sparse sweep: a `1×1`
/// identity-flux tensor whose legs mirror the last site's right bond and its
/// dual, so its first leg (`Out`) contracts the ket's right bond (`In`) and its
/// second leg (`In`) contracts the bra's.
fn right_env_boundary_bsp<T, S>(last: &BlockSparseTensor<T, S>) -> BlockSparseTensor<T, S>
where
    T: Scalar,
    S: Sector,
{
    let ri = &last.data().layout().indices()[2];
    let leg0 = QNIndex::new(ri.blocks().to_vec(), flip_dir(ri.direction()));
    let leg1 = QNIndex::new(ri.blocks().to_vec(), ri.direction());
    let mut e = BlockSparseTensor::<T, S>::zeros(vec![leg0, leg1], S::identity());
    if let Some(d) = e.data_mut().block_data_mut(&BlockCoord(vec![0, 0])) {
        d[0] = T::one();
    }
    e
}

/// Absorb block-sparse site `p` `[left, phys, right]` into the right `⟨φ|φ⟩`
/// environment `r_next` `[right(Out), right_dual(In)]`, returning
/// `[left(Out), left_dual(In)]`.
fn absorb_right_env_bsp<T, S, B>(
    p: &BlockSparseTensor<T, S>,
    r_next: &BlockSparseTensor<T, S>,
    backend: &B,
) -> BlockSparseTensor<T, S>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    // step1[left, phys, right'] = Σ_right P[left,phys,right] R[right, right']
    let step1 = tensordot(backend, p, r_next, &[2], &[0])
        .expect("right-env carry: validated by entry point");
    // R'[left, left'] = Σ_{phys,right'} step1[left,phys,right'] dagger(P)[left', phys, right']
    let p_dag = p.dagger();
    tensordot(backend, &step1, &p_dag, &[1, 2], &[1, 2])
        .expect("right-env overlap: validated by entry point")
}

/// Apply a BlockSparse MPO to a BlockSparse MPS via the density-matrix
/// algorithm.
///
/// See [`apply_density_matrix_dense`] for the algorithm. The block-sparse
/// reduced density matrix is built rank-4
/// `[left, phys, left_dual, phys_dual]` via `tensordot` + `dagger`, keeping
/// `[left, phys]` as separate row legs so [`trunc_svd`] with `nrow = 2` returns
/// the rank-3 site `[left, phys, bond]` directly, with no manual leg unfuse.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths, or if either is empty.
pub(crate) fn apply_density_matrix_bsp<T, S, B>(
    backend: &B,
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>,
    params: Option<&TruncateParams>,
) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    T: Scalar,
    S: Sector,
    B: OpsFor<BlockSparseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let svd_params = density_matrix_svd_params(params);

    // Materialize φ = Wψ (untruncated, bond w·χ).
    let phi: Vec<BlockSparseTensor<T, S>> = (0..n)
        .map(|j| super::local_product_bsp(op.site(j), psi.site(j), backend))
        .collect();

    // Right ⟨φ|φ⟩ environments, built right-to-left: after the reverse,
    // `right_env[j]` is the overlap of sites j+1..n-1 (the remainder to the
    // right of site j); `right_env[n-1]` is the unused 1×1 boundary.
    let mut right_env: Vec<BlockSparseTensor<T, S>> = Vec::with_capacity(n);
    let mut acc = right_env_boundary_bsp(&phi[n - 1]);
    for j in (1..n).rev() {
        let next = absorb_right_env_bsp(&phi[j], &acc, backend);
        right_env.push(acc);
        acc = next;
    }
    right_env.push(acc);
    right_env.reverse();

    let mut tensors: Vec<BlockSparseTensor<T, S>> = Vec::with_capacity(n);
    let mut carry: Option<BlockSparseTensor<T, S>> = None;

    for j in 0..n {
        // θ = C ∘ P_j, legs [left, phys, right]; C absent ⇒ θ = P_j.
        let theta = match carry.as_ref() {
            Some(c) => tensordot(backend, c, &phi[j], &[1], &[0])
                .expect("carry absorption: validated by entry point"),
            None => phi[j].clone(),
        };

        if j < n - 1 {
            let r_next = &right_env[j];

            // ρ = θ · R · θ†, legs [left, phys, left_dual, phys_dual], identity flux.
            let step1 = tensordot(backend, &theta, r_next, &[2], &[0])
                .expect("θ·R: validated by entry point");
            let theta_dag = theta.dagger();
            let rho = tensordot(backend, &step1, &theta_dag, &[2], &[2])
                .expect("ρ = θRθ†: validated by entry point");

            // trunc_svd (nrow = 2) of the PSD ρ returns U = [left, phys, bond]:
            // the new left-isometric site.
            let (u, _s, _vt, _err) = trunc_svd(backend, &rho, 2, &svd_params)
                .expect("trunc_svd: validated by entry point");

            // Carry C' = U† · θ, legs [bond, right].
            let u_dag = u.dagger();
            let c_new = tensordot(backend, &u_dag, &theta, &[0, 1], &[0, 1])
                .expect("carry projection: validated by entry point");

            tensors.push(u);
            carry = Some(c_new);
        } else {
            // Final site is the orthogonality center; no truncation.
            tensors.push(theta);
        }
    }

    let mut result_mps: Mps<BlockSparseStorage<T>, BlockSparseLayout<S>> = Mps::from_sites(tensors);
    result_mps.set_canonical_form(CanonicalForm::Mixed { center: n - 1 });
    result_mps
}
