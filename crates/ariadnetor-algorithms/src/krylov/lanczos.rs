//! Lanczos iteration for the smallest eigenvalue / eigenvector
//! of a Hermitian linear operator, with full reorthogonalization.

use arnet_core::Scalar;
use arnet_tensor::{ComputeBackendTensorExt, DenseTensor, Host, linear_combine};
use num_traits::{Float, One, ToPrimitive, Zero};
use rand::SeedableRng;
use rand::rngs::{StdRng, SysRng};

use super::lanczos_kernels::{
    inner, random_unit_vector, solve_tridiagonal_smallest, sub_complex_axpy, sub_real_axpy,
    sub_two_real_axpy,
};

/// A Hermitian linear operator on a flat vector of length `dim`.
///
/// The Lanczos solver only ever needs to apply the operator to a
/// vector — it never inspects matrix elements directly. Closures of
/// type `Fn(&DenseTensor<T>) -> DenseTensor<T>` automatically
/// implement this trait via the blanket impl.
pub trait LinearOp<T: Scalar> {
    /// Apply the operator to `v`, returning `H · v`.
    fn apply(&self, v: &DenseTensor<T>) -> DenseTensor<T>;
}

impl<T, F> LinearOp<T> for F
where
    T: Scalar,
    F: Fn(&DenseTensor<T>) -> DenseTensor<T>,
{
    fn apply(&self, v: &DenseTensor<T>) -> DenseTensor<T> {
        self(v)
    }
}

/// Parameters controlling the Lanczos iteration.
#[derive(Debug, Clone)]
pub struct LanczosParams {
    /// Maximum number of Lanczos iterations. Capped internally at `dim`.
    pub max_iter: usize,
    /// Convergence tolerance, interpreted as the corresponding `T::Real`.
    ///
    /// Used in two places: (1) the iteration loop exits as soon as the
    /// cheap Lanczos residual estimate `beta_j * |z[m-1]|` falls at or
    /// below `tol`, and (2) the returned [`LanczosResult::converged`]
    /// flag is set from the *true* residual `||H psi - lambda psi||_2`
    /// against the same `tol`, so the flag is consistent with the
    /// residual the caller sees.
    pub tol: f64,
    /// Optional seed for the initial vector. `None` draws from the OS RNG.
    pub seed: Option<u64>,
}

impl Default for LanczosParams {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-10,
            seed: None,
        }
    }
}

/// Output of [`lanczos_smallest`].
#[derive(Debug, Clone)]
pub struct LanczosResult<T: Scalar> {
    /// Smallest eigenvalue.
    pub eigenvalue: T::Real,
    /// Corresponding (unit-norm) eigenvector of length `dim`.
    pub eigenvector: DenseTensor<T>,
    /// Number of Lanczos iterations actually run.
    pub iters: usize,
    /// True residual `|| H v - lambda v ||_2` of the returned pair.
    pub residual: T::Real,
    /// `true` if the returned pair satisfies the true-residual test
    /// `|| H v - lambda v ||_2 ≤ tol`, `false` otherwise. The cheap
    /// Lanczos residual estimate `beta * |z[m-1]|` and the `beta == 0`
    /// invariant-subspace check are used as early-exit heuristics
    /// inside the iteration loop, but neither sets this flag on its
    /// own — the flag comes from comparing the residual the caller
    /// sees against `tol`.
    pub converged: bool,
}

/// Errors from the native Lanczos solver.
///
/// This carries only genuine runtime failures the caller can recover
/// from. Caller-side invariant violations (`dim == 0`, `max_iter == 0`,
/// a non-finite or negative `tol`) and operator-contract violations (an
/// [`LinearOp::apply`] returning a tensor of shape other than `[dim]`)
/// remain panics — they are programmer errors, not recoverable
/// conditions. Failing to reach the requested `tol` is likewise not an
/// error: it is reported as [`LanczosResult::converged`] being `false`
/// so the caller keeps the best Ritz pair.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LanczosError {
    /// The converged eigenpair is not finite (NaN/Inf), typically
    /// because the operator emitted a non-finite vector or the 2-norm
    /// overflowed (see the overflow note on [`lanczos_smallest`]). A
    /// non-finite result must not flow on as a normal `LanczosResult`,
    /// where it would silently poison every downstream computation. The
    /// diagnostics are carried as `f64` so the error type stays
    /// non-generic.
    #[error(
        "Lanczos produced a non-finite result after {iters} iterations: \
         eigenvalue = {eigenvalue}, residual = {residual}"
    )]
    NonFinite {
        /// Lanczos iterations run before the non-finite result was detected.
        iters: usize,
        /// Computed eigenvalue (lossily cast to `f64`).
        eigenvalue: f64,
        /// True residual (lossily cast to `f64`).
        residual: f64,
    },
}

/// Compute the smallest eigenvalue and corresponding eigenvector of a
/// Hermitian operator using Lanczos with full reorthogonalization.
///
/// `dim` is the length of the flat vector the operator acts on. The
/// initial Lanczos vector is drawn at random and normalized; pass
/// [`LanczosParams::seed`] for reproducibility.
///
/// # Errors
///
/// Returns [`LanczosError::NonFinite`] when the computed eigenvalue or
/// residual is not finite (NaN/Inf) — see the overflow note below and
/// the NaN-emitting-operator case. A finite but imprecise result is
/// **not** an error: it returns `Ok` with [`LanczosResult::converged`]
/// set to `false`, leaving the best Ritz pair for the caller to judge.
///
/// # Panics
///
/// Panics on caller-side invariant violations: `dim == 0`,
/// `params.max_iter == 0`, or a non-finite / negative `params.tol`.
/// Panics if [`LinearOp::apply`] returns a tensor whose shape is not
/// `[dim]`. These are programmer errors, kept as panics so the boundary
/// between a programmer bug and a recoverable runtime condition stays
/// principled.
///
/// # Numerical preconditions
///
/// - `params.tol` is honored down to roughly `T::Real::epsilon()`. Asking
///   for a tolerance below working precision (e.g. `1e-10` for `f32`,
///   whose epsilon is `~1.2e-7`) cannot be satisfied; `converged` will
///   reflect the achievable precision rather than the requested one.
/// - The 2-norm used to compute β is the straightforward
///   `sum |x|^2 -> sqrt`. Operator outputs whose elements approach
///   `sqrt(T::Real::MAX)` (roughly `1e19` for `f32`, `1e154` for `f64`)
///   may overflow during squaring, producing a non-finite result that is
///   surfaced as [`LanczosError::NonFinite`]. DMRG-scale Hermitians stay
///   far below this in practice.
pub fn lanczos_smallest<T, Op>(
    op: &Op,
    dim: usize,
    params: &LanczosParams,
) -> Result<LanczosResult<T>, LanczosError>
where
    T: Scalar,
    // The tridiagonal eigenproblem is real symmetric, so we run
    // `eigh_with_backend::<T::Real, _>` and need the inner real type to
    // coincide with T::Real itself. This holds for all valid `Scalar`
    // impls (f32, f64, Complex<f32>, Complex<f64>).
    T::Real: Scalar<Real = T::Real>,
    Op: LinearOp<T>,
{
    assert!(dim >= 1, "lanczos: dim must be >= 1");
    assert!(params.max_iter >= 1, "lanczos: max_iter must be >= 1");
    assert!(
        params.tol.is_finite() && params.tol >= 0.0,
        "lanczos: tol must be a finite non-negative f64, got {}",
        params.tol,
    );
    let max_iter = params.max_iter.min(dim);

    let tol_real: T::Real =
        crate::numeric::try_real_from_f64::<T>(params.tol).unwrap_or_else(|| {
            panic!(
                "try_real_from_f64: tol = {} is not representable in T::Real",
                params.tol
            )
        });
    // `beta_floor` is only a guard against dividing by an unrepresentably
    // small β when normalizing v_{j+1} — it must NOT override the user's
    // tolerance. Convergence is decided exclusively by `residual_estimate`
    // (which is itself bounded by β, so any `tol`-meeting β is caught
    // there first). We use the smallest normal value of T::Real so the
    // floor is a real number in the actual scalar precision (an
    // `f64::MIN_POSITIVE` cast underflows to zero when T::Real = f32).
    let beta_floor: T::Real = T::Real::min_positive_value();

    let mut rng = match params.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::try_from_rng(&mut SysRng).expect("OS RNG must be available"),
    };
    let v0 = random_unit_vector::<T>(dim, &mut rng);

    let mut basis: Vec<DenseTensor<T>> = vec![v0];
    let mut alphas: Vec<T::Real> = Vec::new();
    let mut betas: Vec<T::Real> = Vec::new();

    let mut iters = 0usize;
    let mut converged_lambda: T::Real = T::Real::zero();
    let mut converged_z: DenseTensor<T::Real> = Host::shared().dense(vec![T::Real::one()], vec![1]);

    for j in 0..max_iter {
        iters = j + 1;
        let v_j = basis[j].clone();
        let w_raw = op.apply(&v_j);
        assert_eq!(
            w_raw.shape(),
            &[dim],
            "LinearOp::apply must return a rank-1 tensor of shape [dim]",
        );
        // Operators are not required to declare an output `order()`
        // matching the Lanczos basis. Normalize against `v_j.order()`
        // so the recurrence and the eventual `linear_combine(&basis_refs, ...)`
        // see a uniform-order vector set; for 1D data this is metadata-only.
        let mut w = if w_raw.order() == v_j.order() {
            w_raw
        } else {
            w_raw.reordered(v_j.order())
        };

        // alpha_j = Re<v_j, H v_j>; the imaginary part is zero up to
        // numerical noise for a Hermitian operator.
        let alpha = inner(&v_j, &w).re();
        alphas.push(alpha);

        // Three-term recurrence: w <- w - alpha v_j - beta_{j-1} v_{j-1}.
        if j == 0 {
            w = sub_real_axpy(&w, alpha, &v_j);
        } else {
            let beta_prev = betas[j - 1];
            let v_prev = &basis[j - 1];
            w = sub_two_real_axpy(&w, alpha, &v_j, beta_prev, v_prev);
        }

        // Full reorthogonalization. Two passes of classical Gram-Schmidt
        // ("twice is enough" — Kahan / Parlett) restores orthogonality
        // to working precision even after substantial cancellation.
        for _ in 0..2 {
            for v_k in basis.iter().take(j + 1) {
                let gamma = inner(v_k, &w);
                w = sub_complex_axpy(&w, gamma, v_k);
            }
        }

        let beta = w.norm();

        // Solve current tridiagonal eigenproblem of size m = j + 1.
        let m = j + 1;
        let (lambda, z) = solve_tridiagonal_smallest::<T>(&alphas, &betas, m);
        converged_lambda = lambda;
        converged_z = z;

        // Convergence: residual estimate = beta_j * |z[m-1]|.
        let z_last = Float::abs(converged_z.data_slice()[m - 1]);
        let residual_estimate = beta * z_last;

        if residual_estimate <= tol_real {
            // The Ritz residual ||(H - λ I) ψ|| in the Lanczos basis is at most
            // beta * |z[m-1]|; with full reorthogonalization this also bounds
            // the true residual to working precision. Eigenvalue convergence
            // is quadratic in the residual, so an "eigenvalue stagnation"
            // criterion (prev λ ≈ λ) would exit ~half the precision early —
            // we deliberately do not use it.
            //
            // Note: `converged` is decided AFTER computing the true residual
            // below. Asking for `tol` below working precision cannot be
            // satisfied (Ritz says yes, true residual says no); honesty over
            // optimism.
            break;
        }

        if j + 1 == max_iter {
            break;
        }

        if beta <= beta_floor {
            // β has collapsed to the bottom of the FP range; we cannot safely
            // form v_{j+1} = w / β. The Krylov subspace is effectively
            // exhausted at this point — the current Ritz pair is exact in
            // the spanned subspace.
            break;
        }
        let inv = T::Real::one() / beta;
        let v_next_data: Vec<T> = w.data_slice().iter().map(|&x| x.scale_real(inv)).collect();
        basis.push(Host::shared().dense(v_next_data, vec![dim]));
        betas.push(beta);
    }

    // Reconstruct the Ritz vector psi = sum_k z[k] v_k.
    let m = converged_z.len();
    let basis_refs: Vec<&DenseTensor<T>> = basis.iter().take(m).collect();
    let coefs: Vec<T> = converged_z
        .data_slice()
        .iter()
        .map(|&zk| T::from_real_imag(zk, T::Real::zero()))
        .collect();
    let mut psi = linear_combine(&basis_refs, &coefs).expect("linear_combine on Lanczos basis");
    psi.normalize();

    // True residual: ||H psi - lambda psi||.
    let h_psi_raw = op.apply(&psi);
    assert_eq!(
        h_psi_raw.shape(),
        &[dim],
        "LinearOp::apply must return a rank-1 tensor of shape [dim]",
    );
    // Same rationale as the recurrence loop above: align `op.apply`'s
    // declared order with `psi.order()` so `linear_combine` does not
    // reject mixed-order inputs.
    let h_psi = if h_psi_raw.order() == psi.order() {
        h_psi_raw
    } else {
        h_psi_raw.reordered(psi.order())
    };
    let lambda_t = T::from_real_imag(converged_lambda, T::Real::zero());
    let neg_lambda = lambda_t.scale_real(-T::Real::one());
    let residual_vec =
        linear_combine(&[&h_psi, &psi], &[T::one(), neg_lambda]).expect("residual lc");
    let residual = residual_vec.norm();

    // A non-finite eigenvalue or residual means the operator emitted NaN/Inf
    // or the 2-norm overflowed; the eigenpair is meaningless. Surface it as an
    // error instead of returning an `Ok` result whose `converged == false`
    // would let the NaN flow on silently (e.g. into a DMRG sweep, poisoning the
    // energy). `residual` is the norm of `H psi - lambda psi`, so it is itself
    // non-finite whenever the eigenvector is — checking eigenvalue + residual
    // covers every non-finite source.
    if !converged_lambda.is_finite() || !residual.is_finite() {
        return Err(LanczosError::NonFinite {
            iters,
            eigenvalue: converged_lambda.to_f64().unwrap_or(f64::NAN),
            residual: residual.to_f64().unwrap_or(f64::NAN),
        });
    }

    // Set `converged` from the true residual rather than the Lanczos estimate,
    // so the flag is consistent with the residual the caller sees: the Ritz
    // estimate can claim convergence while the true residual still exceeds
    // `tol` (e.g. when the requested tolerance is below working precision).
    let converged = residual <= tol_real;

    Ok(LanczosResult {
        eigenvalue: converged_lambda,
        eigenvector: psi,
        iters,
        residual,
        converged,
    })
}
