//! Cross-validation tests for the ARPACK-backed solver.
//!
//! Strategy: build a small dense Hermitian matrix, drive
//! `arpack_smallest` via a closure, compare the eigenvalue against
//! `arnet_linalg::eigh` ground truth and the eigenvector against the
//! eigenpair contract `||H psi - lambda psi|| ≈ 0`.

#![cfg(feature = "arpack")]

use approx::assert_abs_diff_eq;
use arnet_algorithms::krylov::{ArpackError, ArpackParams, ArpackScalar, arpack_smallest};
use arnet_core::Scalar;
use arnet_linalg::eigh;
use arnet_native::NativeBackend;
use arnet_tensor::Dense;
use num_complex::Complex;
use num_traits::{Float, NumCast, Zero};
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Hermitian matrix-vector product `H v`. Column-major storage.
fn matvec_cm<T: Scalar>(h: &Dense<T>, n: usize, v: &Dense<T>) -> Dense<T> {
    let h_data = h.data();
    let v_data = v.data();
    let mut out = vec![T::zero(); n];
    for j in 0..n {
        let vj = v_data[j];
        for (i, out_i) in out.iter_mut().enumerate().take(n) {
            *out_i = *out_i + h_data[i + n * j] * vj;
        }
    }
    Dense::new(out, vec![n])
}

fn random_hermitian_f64(n: usize, seed: u64) -> Dense<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let a = Dense::<f64>::random(vec![n, n], &mut rng);
    let mut data = vec![0.0_f64; n * n];
    let a_data = a.data();
    for i in 0..n {
        for j in 0..n {
            let aij = a_data[i + n * j];
            let aji = a_data[j + n * i];
            data[i + n * j] = 0.5 * (aij + aji);
        }
    }
    Dense::new(data, vec![n, n])
}

fn random_hermitian_complex_f64(n: usize, seed: u64) -> Dense<Complex<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let real = Dense::<f64>::random(vec![n, n], &mut rng);
    let imag = Dense::<f64>::random(vec![n, n], &mut rng);
    let r = real.data();
    let im = imag.data();
    let mut data = vec![Complex::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let aij = Complex::new(r[i + n * j], im[i + n * j]);
            let aji = Complex::new(r[j + n * i], im[j + n * i]);
            data[i + n * j] = (aij + aji.conj()) * 0.5;
        }
    }
    Dense::new(data, vec![n, n])
}

fn eigh_smallest<T: Scalar>(h: &Dense<T>) -> T::Real {
    let backend = NativeBackend::shared();
    let (eigvals, _) = eigh(&*backend, h, 1).expect("eigh");
    eigvals.data()[0]
}

/// Diagonal-spectrum smallest-eigenpair check, generic over the scalar type.
/// Builds `H = diag(diag_re)` (in the T-typed storage), drives
/// `arpack_smallest`, and verifies the returned eigenvalue matches the
/// minimum of `diag_re` to within `abs_eps` (in `T::Real`).
fn assert_diagonal_smallest<T>(diag_re: &[f64], expected_smallest: f64, abs_eps: f64)
where
    T: ArpackScalar + std::fmt::Debug,
    T::Real: Scalar<Real = T::Real> + std::fmt::Debug,
{
    let n = diag_re.len();
    let real_zero = T::Real::zero();
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        let v: T::Real = NumCast::from(diag_re[i]).unwrap();
        data[i + n * i] = T::from_real_imag(v, real_zero);
    }
    let h = Dense::new(data, vec![n, n]);

    let result = arpack_smallest::<T, _>(
        &|v: &Dense<T>| matvec_cm(&h, n, v),
        n,
        &ArpackParams {
            tol: 1e-6,
            max_iter: 200,
            ncv: None,
        },
    )
    .expect("arpack should converge on diagonal spectrum");

    let expected: T::Real = NumCast::from(expected_smallest).unwrap();
    let abs_eps: T::Real = NumCast::from(abs_eps).unwrap();
    let err = Float::abs(result.eigenvalue - expected);
    assert!(
        err < abs_eps,
        "scalar type {}: lambda = {:?}, expected = {:?}, err = {:?}",
        std::any::type_name::<T>(),
        result.eigenvalue,
        expected,
        err,
    );
    assert!(result.n_matvec >= 1);
}

// ---------------------------------------------------------------------------
// Diagonal-spectrum coverage across all four scalar types
// ---------------------------------------------------------------------------

#[test]
fn arpack_diagonal_f32() {
    assert_diagonal_smallest::<f32>(&[-3.0, -1.0, 0.0, 2.0, 5.0], -3.0, 1e-4);
}

#[test]
fn arpack_diagonal_complex_f32() {
    assert_diagonal_smallest::<Complex<f32>>(&[-3.0, -1.0, 0.0, 2.0, 5.0], -3.0, 1e-4);
}

#[test]
fn arpack_diagonal_complex_f64() {
    assert_diagonal_smallest::<Complex<f64>>(&[-3.0, -1.0, 0.0, 2.0, 5.0], -3.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Real-symmetric (f64)
// ---------------------------------------------------------------------------

#[test]
fn arpack_diagonal_f64_returns_smallest() {
    let n = 5;
    let diag = [-3.0_f64, -1.0, 0.0, 2.0, 5.0];
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        data[i + n * i] = diag[i];
    }
    let h = Dense::new(data, vec![n, n]);

    let result = arpack_smallest::<f64, _>(
        &|v: &Dense<f64>| matvec_cm(&h, n, v),
        n,
        &ArpackParams {
            tol: 1e-10,
            max_iter: 200,
            ncv: None,
        },
    )
    .expect("arpack should converge");

    assert_abs_diff_eq!(result.eigenvalue, -3.0, epsilon = 1e-9);
    // ARPACK uses a relative stopping criterion, so the absolute
    // residual scales with `|lambda|`. For `|lambda| = 3` and
    // `tol = 1e-10` the wrapper-recomputed residual lands well below
    // 1e-8.
    assert!(result.residual < 1e-8, "residual = {}", result.residual);
    assert!(
        result.n_matvec >= 1,
        "matvec count should be positive: {}",
        result.n_matvec
    );
}

#[test]
fn arpack_random_symmetric_f64_matches_eigh() {
    for &n in &[16usize, 64] {
        let h = random_hermitian_f64(n, 0xC0FFEE + n as u64);
        let lambda_ref = eigh_smallest(&h);

        let result = arpack_smallest::<f64, _>(
            &|v: &Dense<f64>| matvec_cm(&h, n, v),
            n,
            &ArpackParams {
                tol: 1e-10,
                max_iter: 4 * n,
                ncv: None,
            },
        )
        .expect("arpack should converge");

        let rel_err = (result.eigenvalue - lambda_ref).abs() / lambda_ref.abs().max(1.0);
        assert!(
            rel_err < 1e-8,
            "n = {n}: lambda = {}, ref = {lambda_ref}, rel_err = {rel_err}",
            result.eigenvalue
        );
        // ARPACK's relative stopping criterion: residual scales with
        // |lambda| and ||A||. For random Hermitians of these sizes a
        // residual of 1e-7 is comfortably above the floor.
        let residual_bound = result.eigenvalue.abs().max(1.0) * 1e-7;
        assert!(
            result.residual < residual_bound,
            "n = {n}: residual = {} exceeds bound {residual_bound}",
            result.residual
        );
    }
}

// ---------------------------------------------------------------------------
// Complex Hermitian (Complex<f64>)
// ---------------------------------------------------------------------------

#[test]
fn arpack_complex_hermitian_matches_eigh() {
    let n = 32;
    let h = random_hermitian_complex_f64(n, 0xDEADBEEF);
    let lambda_ref = eigh_smallest(&h);

    let result = arpack_smallest::<Complex<f64>, _>(
        &|v: &Dense<Complex<f64>>| matvec_cm(&h, n, v),
        n,
        &ArpackParams {
            tol: 1e-10,
            max_iter: 4 * n,
            ncv: None,
        },
    )
    .expect("arpack should converge");

    let rel_err = (result.eigenvalue - lambda_ref).abs() / lambda_ref.abs().max(1.0);
    assert!(
        rel_err < 1e-8,
        "lambda = {}, ref = {lambda_ref}, rel_err = {rel_err}",
        result.eigenvalue
    );
    let residual_bound = result.eigenvalue.abs().max(1.0) * 1e-7;
    assert!(
        result.residual < residual_bound,
        "residual = {} exceeds bound {residual_bound}",
        result.residual
    );
}

// ---------------------------------------------------------------------------
// `converged` flag — divergence between ARPACK's relative stopping
// criterion and the wrapper's absolute interpretation of params.tol
// ---------------------------------------------------------------------------

#[test]
fn arpack_converged_flag_reflects_absolute_tol() {
    // Random Hermitian on n = 32 with `tol = 1e-15`. ARPACK uses a
    // relative criterion (`residual <= tol * |lambda|`), which it can
    // satisfy in f64. The wrapper-recomputed absolute residual must
    // still exceed `1e-15` for any realistic spectrum, so `converged`
    // is `false` even though `Ok` is returned. This is the divergence
    // signal the field exists to provide.
    let n = 32;
    let h = random_hermitian_f64(n, 0xBAD_C0DE);

    let result = arpack_smallest::<f64, _>(
        &|v: &Dense<f64>| matvec_cm(&h, n, v),
        n,
        &ArpackParams {
            tol: 1e-15,
            max_iter: 4 * n,
            ncv: None,
        },
    )
    .expect("arpack should still return Ok under its relative criterion");

    assert!(
        !result.converged,
        "absolute residual = {} should exceed tol = 1e-15",
        result.residual
    );
    assert!(
        result.residual > 1e-15,
        "if this fails, the divergence test is moot — pick a tighter tol",
    );
}

// ---------------------------------------------------------------------------
// `tol` validation — tol = 0 (and non-finite / negative) must be rejected
// up-front since the wrapper's absolute `converged` check can't honor
// ARPACK's "tol = 0 means machine-epsilon default" sentinel.
// ---------------------------------------------------------------------------

#[test]
fn arpack_rejects_tol_zero() {
    let n = 4;
    let result = arpack_smallest::<f64, _>(
        &|_v: &Dense<f64>| unreachable!("matvec should not run when params are rejected"),
        n,
        &ArpackParams {
            tol: 0.0,
            max_iter: 100,
            ncv: None,
        },
    );
    assert!(matches!(result, Err(ArpackError::InvalidParam(_))));
}

// ---------------------------------------------------------------------------
// MaxIterReached propagation
// ---------------------------------------------------------------------------

#[test]
fn arpack_max_iter_too_small_surfaces_max_iter_reached() {
    // 1D Laplacian-style tridiagonal, max_iter = 1 → forced partial.
    let n = 64usize;
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        data[i + n * i] = 2.0;
        if i + 1 < n {
            data[(i + 1) + n * i] = -1.0;
            data[i + n * (i + 1)] = -1.0;
        }
    }
    let h = Dense::new(data, vec![n, n]);

    let result = arpack_smallest::<f64, _>(
        &|v: &Dense<f64>| matvec_cm(&h, n, v),
        n,
        &ArpackParams {
            tol: 1e-15,
            max_iter: 1,
            ncv: None,
        },
    );

    match result {
        Err(ArpackError::MaxIterReached {
            iters,
            nconv,
            n_matvec,
        }) => {
            assert_eq!(nconv, 0, "nev = 1 forces nconv = 0 on max_iter exit");
            assert!(iters >= 1, "iters should be at least 1: {iters}");
            assert!(n_matvec >= 1, "matvec count should be positive: {n_matvec}");
        }
        other => panic!("expected MaxIterReached, got {other:?}"),
    }
}
