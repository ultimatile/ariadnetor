//! Lanczos iteration for the smallest eigenvalue / eigenvector
//! of a Hermitian linear operator, with full reorthogonalization.

use arnet_core::Scalar;
use arnet_core::backend::MemoryOrder;
use arnet_linalg::{eigh, linear_combine, norm, normalize};
use arnet_native::NativeBackend;
use arnet_tensor::Dense;
use num_traits::{Float, One, Zero};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A Hermitian linear operator on a flat vector of length `dim`.
///
/// The Lanczos solver only ever needs to apply the operator to a
/// vector — it never inspects matrix elements directly. Closures of
/// type `Fn(&Dense<T>) -> Dense<T>` automatically implement this
/// trait via the blanket impl.
pub trait LinearOp<T: Scalar> {
    fn apply(&self, v: &Dense<T>) -> Dense<T>;
}

impl<T, F> LinearOp<T> for F
where
    T: Scalar,
    F: Fn(&Dense<T>) -> Dense<T>,
{
    fn apply(&self, v: &Dense<T>) -> Dense<T> {
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
    pub eigenvector: Dense<T>,
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

/// Compute the smallest eigenvalue and corresponding eigenvector of a
/// Hermitian operator using Lanczos with full reorthogonalization.
///
/// `dim` is the length of the flat vector the operator acts on. The
/// initial Lanczos vector is drawn at random and normalized; pass
/// [`LanczosParams::seed`] for reproducibility.
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
///   may overflow during squaring. DMRG-scale Hermitians stay far below
///   this in practice.
pub fn lanczos_smallest<T, Op>(op: &Op, dim: usize, params: &LanczosParams) -> LanczosResult<T>
where
    T: Scalar,
    // The tridiagonal eigenproblem is real symmetric, so we run `eigh::<T::Real>`
    // and need the inner real type to coincide with T::Real itself. This holds
    // for all valid `Scalar` impls (f32, f64, Complex<f32>, Complex<f64>).
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
    let backend = NativeBackend::shared();

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
        None => StdRng::from_os_rng(),
    };
    let v0 = random_unit_vector::<T>(dim, &mut rng);

    let mut basis: Vec<Dense<T>> = vec![v0];
    let mut alphas: Vec<T::Real> = Vec::new();
    let mut betas: Vec<T::Real> = Vec::new();

    let mut iters = 0usize;
    let mut converged_lambda: T::Real = T::Real::zero();
    let mut converged_z: Dense<T::Real> =
        Dense::new(vec![T::Real::one()], vec![1], MemoryOrder::ColumnMajor);

    for j in 0..max_iter {
        iters = j + 1;
        let v_j = basis[j].clone();
        let mut w = op.apply(&v_j);
        assert_eq!(
            w.shape(),
            &[dim],
            "LinearOp::apply must return a rank-1 tensor of shape [dim]",
        );

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

        let beta = norm(&w);

        // Solve current tridiagonal eigenproblem of size m = j + 1.
        let m = j + 1;
        let (lambda, z) = solve_tridiagonal_smallest::<T>(&backend, &alphas, &betas, m);
        converged_lambda = lambda;
        converged_z = z;

        // Convergence: residual estimate = beta_j * |z[m-1]|.
        let z_last = Float::abs(converged_z.data()[m - 1]);
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
        let v_next_data: Vec<T> = w.data().iter().map(|&x| x.scale_real(inv)).collect();
        basis.push(Dense::new(v_next_data, vec![dim], w.order()));
        betas.push(beta);
    }

    // Reconstruct the Ritz vector psi = sum_k z[k] v_k.
    let m = converged_z.len();
    let basis_refs: Vec<&Dense<T>> = basis.iter().take(m).collect();
    let coefs: Vec<T> = converged_z
        .data()
        .iter()
        .map(|&zk| T::from_real_imag(zk, T::Real::zero()))
        .collect();
    let psi = linear_combine(&basis_refs, &coefs).expect("linear_combine on Lanczos basis");
    let (psi, _) = normalize(&psi);

    // True residual: ||H psi - lambda psi||.
    let h_psi = op.apply(&psi);
    assert_eq!(
        h_psi.shape(),
        &[dim],
        "LinearOp::apply must return a rank-1 tensor of shape [dim]",
    );
    let lambda_t = T::from_real_imag(converged_lambda, T::Real::zero());
    let neg_lambda = lambda_t.scale_real(-T::Real::one());
    let residual_vec =
        linear_combine(&[&h_psi, &psi], &[T::one(), neg_lambda]).expect("residual lc");
    let residual = norm(&residual_vec);

    // Set `converged` from the true residual rather than the Lanczos estimate,
    // so the flag is consistent with the residual the caller sees: the Ritz
    // estimate can claim convergence while the true residual still exceeds
    // `tol` (e.g. when the requested tolerance is below working precision).
    let converged = residual <= tol_real;

    LanczosResult {
        eigenvalue: converged_lambda,
        eigenvector: psi,
        iters,
        residual,
        converged,
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Hermitian inner product `<a, b> = sum_i conj(a_i) * b_i`.
fn inner<T: Scalar>(a: &Dense<T>, b: &Dense<T>) -> T {
    a.data()
        .iter()
        .zip(b.data().iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x.conj() * y)
}

/// Compute `w - alpha * v` where alpha is real.
fn sub_real_axpy<T: Scalar>(w: &Dense<T>, alpha: T::Real, v: &Dense<T>) -> Dense<T> {
    let neg_alpha = -alpha;
    let data: Vec<T> = w
        .data()
        .iter()
        .zip(v.data().iter())
        .map(|(&wi, &vi)| wi + vi.scale_real(neg_alpha))
        .collect();
    Dense::new(data, w.shape().to_vec(), w.order())
}

/// Compute `w - alpha * v - beta * u` where alpha, beta are real.
fn sub_two_real_axpy<T: Scalar>(
    w: &Dense<T>,
    alpha: T::Real,
    v: &Dense<T>,
    beta: T::Real,
    u: &Dense<T>,
) -> Dense<T> {
    let neg_alpha = -alpha;
    let neg_beta = -beta;
    let data: Vec<T> = w
        .data()
        .iter()
        .zip(v.data().iter())
        .zip(u.data().iter())
        .map(|((&wi, &vi), &ui)| wi + vi.scale_real(neg_alpha) + ui.scale_real(neg_beta))
        .collect();
    Dense::new(data, w.shape().to_vec(), w.order())
}

/// Compute `w - gamma * v` where gamma is the (possibly complex) scalar T.
fn sub_complex_axpy<T: Scalar>(w: &Dense<T>, gamma: T, v: &Dense<T>) -> Dense<T> {
    let neg_gamma = gamma.scale_real(-T::Real::one());
    let data: Vec<T> = w
        .data()
        .iter()
        .zip(v.data().iter())
        .map(|(&wi, &vi)| wi + neg_gamma * vi)
        .collect();
    Dense::new(data, w.shape().to_vec(), w.order())
}

/// Draw a unit-norm random vector by sampling each component independently
/// from the uniform distribution on (−0.5, 0.5) and normalizing. The same
/// helper covers real and complex `T`: the imaginary part is dropped for
/// real scalars (see [`Scalar::from_real_imag`]).
fn random_unit_vector<T: Scalar>(dim: usize, rng: &mut StdRng) -> Dense<T> {
    let mut data: Vec<T> = (0..dim)
        .map(|_| {
            let re_f64: f64 = rng.random_range(-0.5..0.5);
            let im_f64: f64 = rng.random_range(-0.5..0.5);
            // Random samples drawn from `(-0.5, 0.5)` are always
            // representable in any `Scalar::Real` (`f32`/`f64`), so
            // `try_real_from_f64` will not return `None` here.
            let re = crate::numeric::try_real_from_f64::<T>(re_f64)
                .expect("uniform [-0.5, 0.5) sample fits in Scalar::Real");
            let im = crate::numeric::try_real_from_f64::<T>(im_f64)
                .expect("uniform [-0.5, 0.5) sample fits in Scalar::Real");
            T::from_real_imag(re, im)
        })
        .collect();
    // Probability of all components sampling to exactly zero is on the order
    // of `2^(-53*dim)` for f64 — astronomical for any reasonable `dim`, but
    // not strictly impossible. Substitute a deterministic non-zero vector so
    // the subsequent `normalize` cannot panic.
    if data.iter().all(|x| x.abs() == T::Real::zero()) {
        data[0] = T::one();
    }
    let v = Dense::new(data, vec![dim], MemoryOrder::ColumnMajor);
    let (normalized, _) = normalize(&v);
    normalized
}

/// Smallest eigenpair of the symmetric tridiagonal matrix of dimension `m`
/// formed from `alphas[0..m]` (diagonal) and `betas[0..m-1]` (off-diagonal).
///
/// The eigenvector is returned as a length-`m` vector in the Lanczos basis.
fn solve_tridiagonal_smallest<T>(
    backend: &NativeBackend,
    alphas: &[T::Real],
    betas: &[T::Real],
    m: usize,
) -> (T::Real, Dense<T::Real>)
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
{
    if m == 1 {
        return (
            alphas[0],
            Dense::new(vec![T::Real::one()], vec![1], MemoryOrder::ColumnMajor),
        );
    }
    // Build the m×m matrix in column-major order to match
    // `NativeBackend::preferred_order()`. For column-major, the (i, j) entry
    // lives at index `i + m * j`.
    let mut data = vec![T::Real::zero(); m * m];
    for i in 0..m {
        data[i + m * i] = alphas[i];
        if i + 1 < m {
            data[(i + 1) + m * i] = betas[i];
            data[i + m * (i + 1)] = betas[i];
        }
    }
    let matrix = Dense::new(data, vec![m, m], MemoryOrder::ColumnMajor);
    let (eigvals, eigvecs) = eigh(backend, &matrix, 1).expect("tridiagonal eigh");
    let lambda = eigvals.data()[0];
    // First column of eigvecs (column-major) holds the eigenvector for the
    // smallest eigenvalue: indices 0..m.
    let z_data = eigvecs.data()[0..m].to_vec();
    (
        lambda,
        Dense::new(z_data, vec![m], MemoryOrder::ColumnMajor),
    )
}

#[cfg(test)]
mod tests {
    //! Direct unit tests for the private helpers `sub_real_axpy`,
    //! `sub_two_real_axpy`, and `random_unit_vector`. These helpers feed the
    //! Lanczos recurrence and the initial-vector draw; sign or branch
    //! mutations inside them are absorbed downstream (the recurrence by full
    //! reorthogonalization, the random helper by normalization), so they
    //! cannot be killed end-to-end. Pinning each helper against an exact
    //! hand-computed or RNG-replicated reference closes that gap.
    use super::*;
    use num_complex::Complex;

    fn real_from_f64<T: Scalar>(x: f64) -> T::Real {
        crate::numeric::try_real_from_f64::<T>(x)
            .expect("test value must be representable in T::Real")
    }

    fn dense_from_real<T: Scalar>(values: &[f64]) -> Dense<T> {
        let data: Vec<T> = values
            .iter()
            .map(|&x| T::from_real_imag(real_from_f64::<T>(x), T::Real::zero()))
            .collect();
        Dense::new(data, vec![values.len()], MemoryOrder::ColumnMajor)
    }

    fn assert_dense_close<T>(got: &Dense<T>, expected: &Dense<T>, tol: T::Real)
    where
        T: Scalar + std::fmt::Debug,
        T::Real: std::fmt::Debug,
    {
        assert_eq!(got.shape(), expected.shape());
        let neg_one = T::Real::zero() - T::Real::one();
        for (i, (&g, &e)) in got.data().iter().zip(expected.data().iter()).enumerate() {
            let diff = Scalar::abs(g + e.scale_real(neg_one));
            assert!(
                diff <= tol,
                "mismatch at index {i}: got = {g:?}, expected = {e:?}, diff = {diff:?}",
            );
        }
    }

    fn check_sub_real_axpy<T>()
    where
        T: Scalar + std::fmt::Debug,
        T::Real: std::fmt::Debug,
    {
        let w = dense_from_real::<T>(&[10.0, 20.0, 30.0]);
        let v = dense_from_real::<T>(&[1.0, 2.0, 3.0]);
        let alpha = real_from_f64::<T>(2.0);
        // w - alpha v = [10-2, 20-4, 30-6] = [8, 16, 24]
        let expected = dense_from_real::<T>(&[8.0, 16.0, 24.0]);
        let result = sub_real_axpy(&w, alpha, &v);
        assert_dense_close::<T>(&result, &expected, real_from_f64::<T>(1e-12));
    }

    #[test]
    fn sub_real_axpy_subtracts_alpha_v_for_real_and_complex() {
        check_sub_real_axpy::<f64>();
        check_sub_real_axpy::<Complex<f64>>();
    }

    fn check_sub_two_real_axpy<T>()
    where
        T: Scalar + std::fmt::Debug,
        T::Real: std::fmt::Debug,
    {
        let w = dense_from_real::<T>(&[10.0, 20.0]);
        let v = dense_from_real::<T>(&[1.0, 2.0]);
        let u = dense_from_real::<T>(&[4.0, 5.0]);
        let alpha = real_from_f64::<T>(2.0);
        let beta = real_from_f64::<T>(3.0);
        // w - 2 v - 3 u = [10-2-12, 20-4-15] = [-4, 1]
        let expected = dense_from_real::<T>(&[-4.0, 1.0]);
        let result = sub_two_real_axpy(&w, alpha, &v, beta, &u);
        assert_dense_close::<T>(&result, &expected, real_from_f64::<T>(1e-12));
    }

    #[test]
    fn sub_two_real_axpy_subtracts_alpha_v_and_beta_u_for_real_and_complex() {
        check_sub_two_real_axpy::<f64>();
        check_sub_two_real_axpy::<Complex<f64>>();
    }

    fn check_random_unit_vector_matches_unaltered_path<T>()
    where
        T: Scalar + std::fmt::Debug,
        T::Real: std::fmt::Debug,
    {
        // Production helper consumes one re-sample and one im-sample per
        // element regardless of T. Reconstruct the un-overwritten path with
        // the same RNG consumption order so we compare against a vector that
        // could only be produced when the all-zero retry branch is NOT taken.
        // The mutation `== → !=` flips that branch into "always take", which
        // would overwrite data[0] = T::one() before normalization and produce
        // a different vector here.
        let dim = 4;
        let seed = 42_u64;

        let mut rng = StdRng::seed_from_u64(seed);
        let observed = random_unit_vector::<T>(dim, &mut rng);

        let mut rng = StdRng::seed_from_u64(seed);
        let raw: Vec<T> = (0..dim)
            .map(|_| {
                let re_f64: f64 = rng.random_range(-0.5..0.5);
                let im_f64: f64 = rng.random_range(-0.5..0.5);
                T::from_real_imag(real_from_f64::<T>(re_f64), real_from_f64::<T>(im_f64))
            })
            .collect();
        let raw_norm = raw
            .iter()
            .map(|&x| {
                let a = Scalar::abs(x);
                a * a
            })
            .fold(T::Real::zero(), |acc, x| acc + x)
            .sqrt();
        // The seed is chosen so the random draw is non-zero, exercising the
        // un-overwritten code path. If raw_norm collapses (RNG semantics change
        // or seed swap), surface that explicitly rather than NaN-mismatching.
        assert!(
            raw_norm > T::Real::zero(),
            "test seed must produce a non-zero sample so the un-overwritten path is exercised",
        );
        let inv_norm = T::Real::one() / raw_norm;
        let expected_data: Vec<T> = raw.iter().map(|&x| x.scale_real(inv_norm)).collect();
        let expected = Dense::new(expected_data, vec![dim], MemoryOrder::ColumnMajor);

        assert_dense_close::<T>(&observed, &expected, real_from_f64::<T>(1e-12));
    }

    #[test]
    fn random_unit_vector_matches_unaltered_path_for_real_and_complex() {
        check_random_unit_vector_matches_unaltered_path::<f64>();
        check_random_unit_vector_matches_unaltered_path::<Complex<f64>>();
    }
}
