//! Internal arithmetic / RNG / tridiagonal helpers for
//! [`super::lanczos::lanczos_smallest`].
//!
//! Split from `lanczos.rs` to keep the public entry-point file under
//! the per-file line cap. The helpers themselves are
//! crate-private — only the lanczos kernel needs them.
//!
//! The three `sub_*_axpy` helpers all subtract a linear combination from
//! their accumulator `w` in place. Two shared properties make this sound:
//! each output element depends only on its own index, so reading `w_i`
//! before overwriting it is exact — bit-identical to materializing the
//! result into a fresh buffer; and `data_slice_mut()` detaches a shared
//! buffer via copy-on-write before the first write, so an operator that
//! returns a `w` aliasing one of the input vectors cannot have that read
//! corrupted by the in-place write.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::tridiag_eigh_with_backend;
use ariadnetor_tensor::{ComputeBackendTensorExt, DenseTensor, Host};
use num_traits::{One, Zero};
use rand::RngExt;
use rand::rngs::StdRng;

/// Hermitian inner product `<a, b> = sum_i conj(a_i) * b_i`.
pub(super) fn inner<T: Scalar>(a: &DenseTensor<T>, b: &DenseTensor<T>) -> T {
    a.data_slice()
        .iter()
        .zip(b.data_slice().iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x.conj() * y)
}

/// Subtract `alpha * v` from `w` in place, where alpha is real.
pub(super) fn sub_real_axpy<T: Scalar>(w: &mut DenseTensor<T>, alpha: T::Real, v: &DenseTensor<T>) {
    // The zip below truncates to the shorter slice; a length mismatch is a
    // solver bug (all Lanczos vectors are length `dim`), so fail fast in debug
    // instead of silently updating only a prefix of `w`.
    debug_assert_eq!(
        w.data_slice().len(),
        v.data_slice().len(),
        "sub_real_axpy: w and v length mismatch",
    );
    let neg_alpha = -alpha;
    for (wi, &vi) in w.data_slice_mut().iter_mut().zip(v.data_slice().iter()) {
        *wi = *wi + vi.scale_real(neg_alpha);
    }
}

/// Subtract `alpha * v + beta * u` from `w` in place, where alpha, beta
/// are real.
pub(super) fn sub_two_real_axpy<T: Scalar>(
    w: &mut DenseTensor<T>,
    alpha: T::Real,
    v: &DenseTensor<T>,
    beta: T::Real,
    u: &DenseTensor<T>,
) {
    debug_assert_eq!(
        w.data_slice().len(),
        v.data_slice().len(),
        "sub_two_real_axpy: w and v length mismatch",
    );
    debug_assert_eq!(
        w.data_slice().len(),
        u.data_slice().len(),
        "sub_two_real_axpy: w and u length mismatch",
    );
    let neg_alpha = -alpha;
    let neg_beta = -beta;
    for ((wi, &vi), &ui) in w
        .data_slice_mut()
        .iter_mut()
        .zip(v.data_slice().iter())
        .zip(u.data_slice().iter())
    {
        *wi = *wi + vi.scale_real(neg_alpha) + ui.scale_real(neg_beta);
    }
}

/// Subtract `gamma * v` from `w` in place, where gamma is the (possibly
/// complex) scalar T.
pub(super) fn sub_complex_axpy<T: Scalar>(w: &mut DenseTensor<T>, gamma: T, v: &DenseTensor<T>) {
    debug_assert_eq!(
        w.data_slice().len(),
        v.data_slice().len(),
        "sub_complex_axpy: w and v length mismatch",
    );
    let neg_gamma = gamma.scale_real(-T::Real::one());
    for (wi, &vi) in w.data_slice_mut().iter_mut().zip(v.data_slice().iter()) {
        *wi = *wi + neg_gamma * vi;
    }
}

/// Draw a unit-norm random vector by sampling each component
/// independently from the uniform distribution on (−0.5, 0.5) and
/// normalizing.
pub(super) fn random_unit_vector<T: Scalar>(dim: usize, rng: &mut StdRng) -> DenseTensor<T> {
    let mut data: Vec<T> = (0..dim)
        .map(|_| {
            let re_f64: f64 = rng.random_range(-0.5..0.5);
            let im_f64: f64 = rng.random_range(-0.5..0.5);
            let re = ariadnetor_core::try_real_from_f64::<T>(re_f64)
                .expect("uniform [-0.5, 0.5) sample fits in Scalar::Real");
            let im = ariadnetor_core::try_real_from_f64::<T>(im_f64)
                .expect("uniform [-0.5, 0.5) sample fits in Scalar::Real");
            T::from_real_imag(re, im)
        })
        .collect();
    // Probability of all components sampling to exactly zero is on
    // the order of `2^(-53*dim)` for f64 — astronomical for any
    // reasonable `dim`, but not strictly impossible. Substitute a
    // deterministic non-zero vector so the subsequent `normalize`
    // cannot panic.
    if data.iter().all(|x| x.abs() == T::Real::zero()) {
        data[0] = T::one();
    }
    let mut v = Host::shared().dense(data, vec![dim]);
    v.normalize();
    v
}

/// Smallest eigenpair of the symmetric tridiagonal matrix of
/// dimension `m` formed from `alphas[0..m]` (diagonal) and
/// `betas[0..m-1]` (off-diagonal).
///
/// The eigenvector is returned as a length-`m` vector in the Lanczos
/// basis.
pub(super) fn solve_tridiagonal_smallest<T>(
    alphas: &[T::Real],
    betas: &[T::Real],
    m: usize,
) -> (T::Real, DenseTensor<T::Real>)
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
{
    // The specialized tridiagonal path takes the diagonal / subdiagonal
    // directly — no m×m dense build, no O(m^3) tridiagonalization.
    // Eigenvalues come back ascending, so index 0 is the smallest pair.
    // Column 0 is extracted through the order-aware `get` so the code
    // does not assume which memory order the backend produced `V` in.
    let (eigvals, eigvecs) =
        tridiag_eigh_with_backend(Host::shared().as_ref(), &alphas[..m], &betas[..m - 1])
            .expect("tridiagonal eigh");
    let lambda = eigvals.data_slice()[0];
    let z_data: Vec<T::Real> = (0..m).map(|i| eigvecs.get([i, 0])).collect();
    (lambda, Host::shared().dense(z_data, vec![m]))
}

#[cfg(test)]
mod tests {
    //! Direct unit tests for the private helpers `sub_real_axpy`,
    //! `sub_two_real_axpy`, `sub_complex_axpy`, and `random_unit_vector`.
    use super::*;
    use num_complex::Complex;
    use num_traits::Float;
    use rand::SeedableRng;

    fn real_from_f64<T: Scalar>(x: f64) -> T::Real {
        ariadnetor_core::try_real_from_f64::<T>(x)
            .expect("test value must be representable in T::Real")
    }

    fn dense_from_real<T: Scalar>(values: &[f64]) -> DenseTensor<T> {
        let data: Vec<T> = values
            .iter()
            .map(|&x| T::from_real_imag(real_from_f64::<T>(x), T::Real::zero()))
            .collect();
        Host::shared().dense(data, vec![values.len()])
    }

    fn assert_dense_close<T>(got: &DenseTensor<T>, expected: &DenseTensor<T>, tol: T::Real)
    where
        T: Scalar + std::fmt::Debug,
        T::Real: std::fmt::Debug,
    {
        assert_eq!(got.shape(), expected.shape());
        let neg_one = T::Real::zero() - T::Real::one();
        for (i, (&g, &e)) in got
            .data_slice()
            .iter()
            .zip(expected.data_slice().iter())
            .enumerate()
        {
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
        let expected = dense_from_real::<T>(&[8.0, 16.0, 24.0]);
        let mut w = w;
        sub_real_axpy(&mut w, alpha, &v);
        assert_dense_close::<T>(&w, &expected, real_from_f64::<T>(1e-12));

        // Aliasing case: `w` shares its storage buffer with `v` via clone.
        // The copy-on-write detach in `data_slice_mut` must give `w` a private
        // buffer before the first write, so `w` becomes v - 2*v = [-1, -2, -3]
        // and `v` itself stays unmodified.
        let v_alias = dense_from_real::<T>(&[1.0, 2.0, 3.0]);
        let mut w_alias = v_alias.clone();
        sub_real_axpy(&mut w_alias, alpha, &v_alias);
        let expected_alias = dense_from_real::<T>(&[-1.0, -2.0, -3.0]);
        assert_dense_close::<T>(&w_alias, &expected_alias, real_from_f64::<T>(1e-12));
        let v_untouched = dense_from_real::<T>(&[1.0, 2.0, 3.0]);
        assert_dense_close::<T>(&v_alias, &v_untouched, real_from_f64::<T>(1e-12));
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
        let expected = dense_from_real::<T>(&[-4.0, 1.0]);
        let mut w = w;
        sub_two_real_axpy(&mut w, alpha, &v, beta, &u);
        assert_dense_close::<T>(&w, &expected, real_from_f64::<T>(1e-12));

        // Aliasing case: `w` shares its storage buffer with `v` via clone. The
        // copy-on-write detach must keep the `v` read intact, so `w` becomes
        // v - 2*v - 3*u = -v - 3*u = [-13, -17] and `v` stays unmodified.
        let v_alias = dense_from_real::<T>(&[1.0, 2.0]);
        let u2 = dense_from_real::<T>(&[4.0, 5.0]);
        let mut w_alias = v_alias.clone();
        sub_two_real_axpy(&mut w_alias, alpha, &v_alias, beta, &u2);
        let expected_alias = dense_from_real::<T>(&[-13.0, -17.0]);
        assert_dense_close::<T>(&w_alias, &expected_alias, real_from_f64::<T>(1e-12));
        let v_untouched = dense_from_real::<T>(&[1.0, 2.0]);
        assert_dense_close::<T>(&v_alias, &v_untouched, real_from_f64::<T>(1e-12));
    }

    #[test]
    fn sub_two_real_axpy_subtracts_alpha_v_and_beta_u_for_real_and_complex() {
        check_sub_two_real_axpy::<f64>();
        check_sub_two_real_axpy::<Complex<f64>>();
    }

    fn check_sub_complex_axpy<T>()
    where
        T: Scalar + std::fmt::Debug,
        T::Real: std::fmt::Debug,
    {
        let w = dense_from_real::<T>(&[10.0, 20.0, 30.0]);
        let v = dense_from_real::<T>(&[1.0, 2.0, 3.0]);
        let gamma = T::from_real_imag(real_from_f64::<T>(2.0), T::Real::zero());
        let expected = dense_from_real::<T>(&[8.0, 16.0, 24.0]);
        let mut w = w;
        sub_complex_axpy(&mut w, gamma, &v);
        assert_dense_close::<T>(&w, &expected, real_from_f64::<T>(1e-12));

        // Aliasing case: `w` shares its storage buffer with `v` via clone.
        // The copy-on-write detach in `data_slice_mut` must give `w` a
        // private buffer before the first write, so the result is exact
        // (v - gamma * v = [-1, -2, -3]) and `v` itself stays unmodified.
        let v_alias = dense_from_real::<T>(&[1.0, 2.0, 3.0]);
        let mut w_alias = v_alias.clone();
        sub_complex_axpy(&mut w_alias, gamma, &v_alias);
        let expected_alias = dense_from_real::<T>(&[-1.0, -2.0, -3.0]);
        assert_dense_close::<T>(&w_alias, &expected_alias, real_from_f64::<T>(1e-12));
        let v_untouched = dense_from_real::<T>(&[1.0, 2.0, 3.0]);
        assert_dense_close::<T>(&v_alias, &v_untouched, real_from_f64::<T>(1e-12));
    }

    #[test]
    fn sub_complex_axpy_subtracts_gamma_v_and_is_alias_safe_for_real_and_complex() {
        check_sub_complex_axpy::<f64>();
        check_sub_complex_axpy::<Complex<f64>>();
    }

    fn check_random_unit_vector_matches_unaltered_path<T>()
    where
        T: Scalar + std::fmt::Debug,
        T::Real: std::fmt::Debug,
    {
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
        assert!(
            raw_norm > T::Real::zero(),
            "test seed must produce a non-zero sample so the un-overwritten path is exercised",
        );
        let inv_norm = T::Real::one() / raw_norm;
        let expected_data: Vec<T> = raw.iter().map(|&x| x.scale_real(inv_norm)).collect();
        let expected = Host::shared().dense(expected_data, vec![dim]);

        assert_dense_close::<T>(&observed, &expected, real_from_f64::<T>(1e-12));
    }

    #[test]
    fn random_unit_vector_matches_unaltered_path_for_real_and_complex() {
        check_random_unit_vector_matches_unaltered_path::<f64>();
        check_random_unit_vector_matches_unaltered_path::<Complex<f64>>();
    }
}
