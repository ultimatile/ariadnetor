//! Internal arithmetic / RNG / tridiagonal helpers for
//! [`super::lanczos::lanczos_smallest`].
//!
//! Split from `lanczos.rs` to keep the public entry-point file under
//! the per-file line cap. The helpers themselves are
//! crate-private — only the lanczos kernel needs them.

use arnet_core::Scalar;
use arnet_linalg::eigh_with_backend;
use arnet_tensor::{ComputeBackendTensorExt, DenseTensor, Host};
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

/// Compute `w - alpha * v` where alpha is real.
pub(super) fn sub_real_axpy<T: Scalar>(
    w: &DenseTensor<T>,
    alpha: T::Real,
    v: &DenseTensor<T>,
) -> DenseTensor<T> {
    let neg_alpha = -alpha;
    let data: Vec<T> = w
        .data_slice()
        .iter()
        .zip(v.data_slice().iter())
        .map(|(&wi, &vi)| wi + vi.scale_real(neg_alpha))
        .collect();
    DenseTensor::from_data(Host::shared().make_tensor(data, w.shape().to_vec()))
}

/// Compute `w - alpha * v - beta * u` where alpha, beta are real.
pub(super) fn sub_two_real_axpy<T: Scalar>(
    w: &DenseTensor<T>,
    alpha: T::Real,
    v: &DenseTensor<T>,
    beta: T::Real,
    u: &DenseTensor<T>,
) -> DenseTensor<T> {
    let neg_alpha = -alpha;
    let neg_beta = -beta;
    let data: Vec<T> = w
        .data_slice()
        .iter()
        .zip(v.data_slice().iter())
        .zip(u.data_slice().iter())
        .map(|((&wi, &vi), &ui)| wi + vi.scale_real(neg_alpha) + ui.scale_real(neg_beta))
        .collect();
    DenseTensor::from_data(Host::shared().make_tensor(data, w.shape().to_vec()))
}

/// Compute `w - gamma * v` where gamma is the (possibly complex) scalar T.
pub(super) fn sub_complex_axpy<T: Scalar>(
    w: &DenseTensor<T>,
    gamma: T,
    v: &DenseTensor<T>,
) -> DenseTensor<T> {
    let neg_gamma = gamma.scale_real(-T::Real::one());
    let data: Vec<T> = w
        .data_slice()
        .iter()
        .zip(v.data_slice().iter())
        .map(|(&wi, &vi)| wi + neg_gamma * vi)
        .collect();
    DenseTensor::from_data(Host::shared().make_tensor(data, w.shape().to_vec()))
}

/// Draw a unit-norm random vector by sampling each component
/// independently from the uniform distribution on (−0.5, 0.5) and
/// normalizing.
pub(super) fn random_unit_vector<T: Scalar>(dim: usize, rng: &mut StdRng) -> DenseTensor<T> {
    let mut data: Vec<T> = (0..dim)
        .map(|_| {
            let re_f64: f64 = rng.random_range(-0.5..0.5);
            let im_f64: f64 = rng.random_range(-0.5..0.5);
            let re = crate::numeric::try_real_from_f64::<T>(re_f64)
                .expect("uniform [-0.5, 0.5) sample fits in Scalar::Real");
            let im = crate::numeric::try_real_from_f64::<T>(im_f64)
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
    let mut v = DenseTensor::from_data(Host::shared().make_tensor(data, vec![dim]));
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
    if m == 1 {
        return (
            alphas[0],
            DenseTensor::from_data(Host::shared().make_tensor(vec![T::Real::one()], vec![1])),
        );
    }
    // Build the m×m matrix in column-major order to match the host
    // substrate's `preferred_order()`. For column-major, the (i, j)
    // entry lives at index `i + m * j`.
    let mut data = vec![T::Real::zero(); m * m];
    for i in 0..m {
        data[i + m * i] = alphas[i];
        if i + 1 < m {
            data[(i + 1) + m * i] = betas[i];
            data[i + m * (i + 1)] = betas[i];
        }
    }
    let matrix = DenseTensor::from_data(Host::shared().make_tensor(data, vec![m, m]));
    let (eigvals, eigvecs) =
        eigh_with_backend(Host::shared().as_ref(), &matrix, 1).expect("tridiagonal eigh");
    let lambda = eigvals.data_slice()[0];
    let z_data = eigvecs.data_slice()[0..m].to_vec();
    (
        lambda,
        DenseTensor::from_data(Host::shared().make_tensor(z_data, vec![m])),
    )
}

#[cfg(test)]
mod tests {
    //! Direct unit tests for the private helpers `sub_real_axpy`,
    //! `sub_two_real_axpy`, and `random_unit_vector`.
    use super::*;
    use num_complex::Complex;
    use num_traits::Float;
    use rand::SeedableRng;

    fn real_from_f64<T: Scalar>(x: f64) -> T::Real {
        crate::numeric::try_real_from_f64::<T>(x)
            .expect("test value must be representable in T::Real")
    }

    fn dense_from_real<T: Scalar>(values: &[f64]) -> DenseTensor<T> {
        let data: Vec<T> = values
            .iter()
            .map(|&x| T::from_real_imag(real_from_f64::<T>(x), T::Real::zero()))
            .collect();
        DenseTensor::from_data(Host::shared().make_tensor(data, vec![values.len()]))
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
        let expected = DenseTensor::from_data(Host::shared().make_tensor(expected_data, vec![dim]));

        assert_dense_close::<T>(&observed, &expected, real_from_f64::<T>(1e-12));
    }

    #[test]
    fn random_unit_vector_matches_unaltered_path_for_real_and_complex() {
        check_random_unit_vector_matches_unaltered_path::<f64>();
        check_random_unit_vector_matches_unaltered_path::<Complex<f64>>();
    }
}
