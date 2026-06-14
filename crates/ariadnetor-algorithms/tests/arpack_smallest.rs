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
use arnet_linalg::eigh_with_backend;
use arnet_native::NativeBackend;
use arnet_tensor::DenseTensor;
use num_complex::Complex;
use num_traits::{Float, NumCast, One, Zero};
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Hermitian matrix-vector product `H v`. Column-major storage.
fn matvec_cm<T: Scalar>(h: &DenseTensor<T>, n: usize, v: &DenseTensor<T>) -> DenseTensor<T> {
    let h_data = h.data_slice();
    let v_data = v.data_slice();
    let mut out = vec![T::zero(); n];
    for j in 0..n {
        let vj = v_data[j];
        for (i, out_i) in out.iter_mut().enumerate().take(n) {
            *out_i = *out_i + h_data[i + n * j] * vj;
        }
    }
    DenseTensor::from_raw_parts(out, vec![n])
}

fn random_hermitian_f64(n: usize, seed: u64) -> DenseTensor<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let a = DenseTensor::<f64>::random(vec![n, n], &mut rng);
    let mut data = vec![0.0_f64; n * n];
    let a_data = a.data_slice();
    for i in 0..n {
        for j in 0..n {
            let aij = a_data[i + n * j];
            let aji = a_data[j + n * i];
            data[i + n * j] = 0.5 * (aij + aji);
        }
    }
    DenseTensor::from_raw_parts(data, vec![n, n])
}

fn random_hermitian_complex_f64(n: usize, seed: u64) -> DenseTensor<Complex<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let real = DenseTensor::<f64>::random(vec![n, n], &mut rng);
    let imag = DenseTensor::<f64>::random(vec![n, n], &mut rng);
    let r = real.data_slice();
    let im = imag.data_slice();
    let mut data = vec![Complex::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let aij = Complex::new(r[i + n * j], im[i + n * j]);
            let aji = Complex::new(r[j + n * i], im[j + n * i]);
            data[i + n * j] = (aij + aji.conj()) * 0.5;
        }
    }
    DenseTensor::from_raw_parts(data, vec![n, n])
}

fn eigh_smallest<T: Scalar>(h: &DenseTensor<T>) -> T::Real {
    let (eigvals, _) = eigh_with_backend(&NativeBackend::new(), h, 1).expect("eigh");
    eigvals.data_slice()[0]
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
    let h = DenseTensor::from_raw_parts(data, vec![n, n]);

    let result = arpack_smallest::<T, _>(
        &|v: &DenseTensor<T>| matvec_cm(&h, n, v),
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

    // Eigenvector dimension contract
    assert_eq!(
        result.eigenvector.shape(),
        &[n],
        "scalar type {}: eigenvector shape mismatch",
        std::any::type_name::<T>(),
    );

    // Unit normalization. The wrapper applies `normalize` to its
    // output, so ||v|| equals 1 modulo IEEE rounding. 1e-5 is
    // generous against the f32 floor (~few * ULP * sqrt(n)) and
    // trivial against the f64 floor.
    let v_norm = result.eigenvector.norm();
    let one_real = T::Real::one();
    let norm_eps_real: T::Real = NumCast::from(1e-5_f64).unwrap();
    assert!(
        Float::abs(v_norm - one_real) < norm_eps_real,
        "scalar type {}: ||v|| = {:?}, expected ~1",
        std::any::type_name::<T>(),
        v_norm,
    );

    // True residual ||H psi - lambda psi||. ARPACK's stopping
    // criterion is relative (residual <= tol * |lambda|); for the
    // hardcoded tol = 1e-6 here the wrapper-recomputed residual is
    // at most a few * 1e-6 across all four scalar types. Bound
    // 1e-4 * (|lambda| + 1) keeps two orders of headroom.
    let lambda_abs = Float::abs(result.eigenvalue);
    let res_mult: T::Real = NumCast::from(1e-4_f64).unwrap();
    let residual_bound = res_mult * (lambda_abs + one_real);
    assert!(
        result.residual < residual_bound,
        "scalar type {}: residual = {:?} exceeds bound {:?}",
        std::any::type_name::<T>(),
        result.residual,
        residual_bound,
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
    let h = DenseTensor::from_raw_parts(data, vec![n, n]);

    let result = arpack_smallest::<f64, _>(
        &|v: &DenseTensor<f64>| matvec_cm(&h, n, v),
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
    assert_eq!(
        result.eigenvector.shape(),
        &[n],
        "eigenvector shape mismatch",
    );
    assert_abs_diff_eq!(result.eigenvector.norm(), 1.0_f64, epsilon = 1e-12);
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
            &|v: &DenseTensor<f64>| matvec_cm(&h, n, v),
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
        &|v: &DenseTensor<Complex<f64>>| matvec_cm(&h, n, v),
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
        &|v: &DenseTensor<f64>| matvec_cm(&h, n, v),
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
// `ArpackError` contract — trait bounds, From-arpack conversion preserves
// each variant, and Display carries payload-derived content.
// ---------------------------------------------------------------------------

/// Compile-time contract: `ArpackError` must implement the standard
/// error traits so callers can use `?` in `anyhow::Result` /
/// `Box<dyn Error>` contexts. Asserted via a generic stub that only
/// type-checks when the bounds are satisfied.
#[test]
fn arpack_error_implements_standard_traits() {
    fn assert_standard_error<E>()
    where
        E: std::error::Error + std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    {
    }
    assert_standard_error::<ArpackError>();
}

/// Each `arpack::Error` variant must round-trip into the matching
/// `ArpackError` variant (no collapsing into the catch-all
/// `InvalidParam("unrecognized arpack error variant")`), and the
/// resulting Display output must carry a payload-derived substring.
fn assert_round_trip<F>(input: arpack::Error, expected_substr: &str, predicate: F)
where
    F: FnOnce(&ArpackError) -> bool,
{
    let converted: ArpackError = input.into();
    assert!(
        predicate(&converted),
        "variant lost on conversion: got {converted:?}",
    );
    let display = format!("{converted}");
    assert!(
        display.contains(expected_substr),
        "Display missing {expected_substr:?}: {display:?}",
    );
}

#[test]
fn arpack_error_from_arpack_preserves_variant_and_display() {
    use arpack::Error;

    assert_round_trip(Error::InvalidParam("dummy"), "dummy", |e| {
        matches!(e, ArpackError::InvalidParam("dummy"))
    });
    assert_round_trip(
        Error::AupdFailed {
            info: 7,
            iters: 0,
            nconv: 0,
            n_matvec: 0,
        },
        "7",
        |e| matches!(e, ArpackError::AupdFailed(7)),
    );
    assert_round_trip(
        Error::EupdFailed {
            info: -3,
            iters: 0,
            nconv: 0,
            n_matvec: 0,
        },
        "-3",
        |e| matches!(e, ArpackError::EupdFailed(-3)),
    );
    assert_round_trip(Error::UnexpectedIdo(99), "99", |e| {
        matches!(e, ArpackError::UnexpectedIdo(99))
    });
    assert_round_trip(
        Error::MaxIterReached {
            iters: 10,
            nconv: 0,
            n_matvec: 50,
        },
        "iters = 10",
        |e| {
            matches!(
                e,
                ArpackError::MaxIterReached {
                    iters: 10,
                    nconv: 0,
                    n_matvec: 50,
                }
            )
        },
    );
}

// ---------------------------------------------------------------------------
// `tol` validation — non-finite and non-positive `tol` must be rejected
// up-front since the wrapper's absolute `converged` check can't honor
// ARPACK's "tol = 0 means machine-epsilon default" sentinel.
// ---------------------------------------------------------------------------

#[test]
fn arpack_rejects_invalid_tol() {
    let n = 4;
    let bad_tols = [
        ("zero", 0.0_f64),
        ("negative", -1.0),
        ("nan", f64::NAN),
        ("infinity", f64::INFINITY),
        ("negative infinity", f64::NEG_INFINITY),
    ];
    for (label, tol) in bad_tols {
        let result = arpack_smallest::<f64, _>(
            &|_v: &DenseTensor<f64>| {
                unreachable!("matvec must not run when tol = {label} is rejected up-front")
            },
            n,
            &ArpackParams {
                tol,
                max_iter: 100,
                ncv: None,
            },
        );
        assert!(
            matches!(result, Err(ArpackError::InvalidParam(_))),
            "tol = {label} should be rejected, got {result:?}",
        );
    }
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
    let h = DenseTensor::from_raw_parts(data, vec![n, n]);

    let result = arpack_smallest::<f64, _>(
        &|v: &DenseTensor<f64>| matvec_cm(&h, n, v),
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
