//! Tests for the Lanczos smallest-eigenvalue solver.
//!
//! Validation strategy: build a small dense Hermitian matrix, drive
//! the solver via a closure that does a matrix-vector product, and
//! compare against `arnet_linalg::eigh` ground truth.

use approx::assert_abs_diff_eq;
use arnet_algorithms::krylov::{LanczosError, LanczosParams, lanczos_smallest};
use arnet_core::Scalar;
use arnet_linalg::eigh_with_backend;
use arnet_native::NativeBackend;
use arnet_tensor::{ComputeBackendTensorExt, DenseTensor, Host};
use num_complex::Complex;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Hermitian matrix-vector product `H v`. The matrix is stored in
/// column-major order (matching `NativeBackend::preferred_order()`):
/// element `(i, j)` lives at flat index `i + n * j`.
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
    Host::shared().dense(out, vec![n])
}

/// Build a random Hermitian matrix `H = (A + A^H) / 2` of size `n×n`,
/// stored in column-major.
fn random_hermitian_f64(n: usize, seed: u64) -> DenseTensor<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let a = DenseTensor::<f64>::random(vec![n, n], &mut rng);
    let a_data = a.data_slice();
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let aij = a_data[i + n * j];
            let aji = a_data[j + n * i];
            data[i + n * j] = 0.5 * (aij + aji);
        }
    }
    Host::shared().dense(data, vec![n, n])
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
    Host::shared().dense(data, vec![n, n])
}

/// Smallest eigenvalue of a Hermitian matrix via dense `eigh`.
fn eigh_smallest<T: Scalar>(h: &DenseTensor<T>) -> T::Real {
    let (eigvals, _) = eigh_with_backend(&NativeBackend::new(), h, 1).expect("eigh");
    eigvals.data_slice()[0]
}

// ---------------------------------------------------------------------------
// Real symmetric tests
// ---------------------------------------------------------------------------

#[test]
fn lanczos_diagonal_returns_min_eigenvalue() {
    let n = 5;
    let diag = [-3.0_f64, -1.0, 0.0, 2.0, 5.0];
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        data[i + n * i] = diag[i];
    }
    let h = Host::shared().dense(data, vec![n, n]);

    let result = lanczos_smallest::<f64, _>(
        &|v: &DenseTensor<f64>| matvec_cm(&h, n, v),
        n,
        &LanczosParams {
            max_iter: 50,
            tol: 1e-12,
            seed: Some(42),
        },
    )
    .unwrap();

    assert_abs_diff_eq!(result.eigenvalue, -3.0, epsilon = 1e-10);
    assert!(result.residual < 1e-9, "residual = {}", result.residual);
    assert!(result.converged, "expected converged = true");
}

#[test]
fn lanczos_random_symmetric_matches_eigh() {
    for &n in &[16usize, 64, 256] {
        let h = random_hermitian_f64(n, 0xC0FFEE + n as u64);
        let lambda_ref = eigh_smallest(&h);

        let result = lanczos_smallest::<f64, _>(
            &|v: &DenseTensor<f64>| matvec_cm(&h, n, v),
            n,
            &LanczosParams {
                max_iter: 4 * n,
                tol: 1e-11,
                seed: Some(7),
            },
        )
        .unwrap();

        let rel_err = (result.eigenvalue - lambda_ref).abs() / lambda_ref.abs().max(1.0);
        assert!(
            rel_err < 1e-9,
            "n = {n}: lambda = {}, ref = {lambda_ref}, rel_err = {rel_err}",
            result.eigenvalue
        );
        assert!(
            result.residual < 1e-7,
            "n = {n}: residual = {}",
            result.residual
        );
        assert!(result.converged, "n = {n}: expected converged = true");
    }
}

#[test]
fn lanczos_near_degenerate_cluster() {
    let n = 4;
    let diag = [0.0_f64, 1e-6, 1e-6, 1.0];
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        data[i + n * i] = diag[i];
    }
    let h = Host::shared().dense(data, vec![n, n]);

    let result = lanczos_smallest::<f64, _>(
        &|v: &DenseTensor<f64>| matvec_cm(&h, n, v),
        n,
        &LanczosParams {
            max_iter: 60,
            tol: 1e-12,
            seed: Some(11),
        },
    )
    .unwrap();

    assert!(
        result.eigenvalue.abs() < 1e-9,
        "lambda = {}",
        result.eigenvalue
    );
    let v = result.eigenvector.data_slice();
    assert!(v[0].abs() > 0.99, "v[0] = {} (expected near ±1)", v[0]);
    assert!(result.residual < 1e-7);
    assert!(result.converged, "expected converged = true");
}

// ---------------------------------------------------------------------------
// Complex Hermitian tests
// ---------------------------------------------------------------------------

#[test]
fn lanczos_complex_hermitian_matches_eigh() {
    let n = 32;
    let h = random_hermitian_complex_f64(n, 0xDEADBEEF);
    let lambda_ref = eigh_smallest(&h);

    let result = lanczos_smallest::<Complex<f64>, _>(
        &|v: &DenseTensor<Complex<f64>>| matvec_cm(&h, n, v),
        n,
        &LanczosParams {
            max_iter: 4 * n,
            tol: 1e-11,
            seed: Some(99),
        },
    )
    .unwrap();

    let rel_err = (result.eigenvalue - lambda_ref).abs() / lambda_ref.abs().max(1.0);
    assert!(
        rel_err < 1e-9,
        "lambda = {}, ref = {lambda_ref}, rel_err = {rel_err}",
        result.eigenvalue
    );
    assert!(result.residual < 1e-7, "residual = {}", result.residual);
    assert!(result.converged, "expected converged = true");
}

// ---------------------------------------------------------------------------
// Iteration-count contract
// ---------------------------------------------------------------------------

#[test]
fn lanczos_n1_returns_iters_one() {
    let h = Host::shared().dense(vec![5.0_f64], vec![1, 1]);
    let result = lanczos_smallest::<f64, _>(
        &|v: &DenseTensor<f64>| matvec_cm(&h, 1, v),
        1,
        &LanczosParams {
            max_iter: 10,
            tol: 1e-12,
            seed: Some(0),
        },
    )
    .unwrap();

    assert_eq!(
        result.iters, 1,
        "n=1 must converge in exactly one iteration"
    );
    assert_abs_diff_eq!(result.eigenvalue, 5.0, epsilon = 1e-12);
    assert!(result.converged, "expected converged = true");
}

// ---------------------------------------------------------------------------
// Fallible-API contract: non-finite -> Err, finite-imprecise -> Ok
// ---------------------------------------------------------------------------

#[test]
fn lanczos_nonfinite_operator_returns_error() {
    // An operator that emits a NaN vector of the correct shape drives the
    // whole iteration non-finite. The solver must surface this as
    // `LanczosError::NonFinite` rather than returning an `Ok` result whose
    // NaN eigenpair would silently propagate downstream.
    let n = 4;
    let op = |v: &DenseTensor<f64>| -> DenseTensor<f64> {
        let data: Vec<f64> = v.data_slice().iter().map(|_| f64::NAN).collect();
        Host::shared().dense(data, vec![n])
    };

    let result = lanczos_smallest::<f64, _>(
        &op,
        n,
        &LanczosParams {
            max_iter: 50,
            tol: 1e-10,
            seed: Some(1),
        },
    );

    // Bind the diagnostic fields, not just the variant: the solver populates
    // them from its own computed values (`iters`, `eigenvalue`/`residual` via
    // `to_f64`), so reading them here pins that population end to end. The
    // NaN-driven run yields non-finite eigenvalue and residual after at least
    // one iteration.
    match result {
        Err(LanczosError::NonFinite {
            iters,
            eigenvalue,
            residual,
        }) => {
            assert!(iters >= 1, "expected at least one iteration, got {iters}");
            assert!(
                eigenvalue.is_nan(),
                "expected NaN eigenvalue, got {eigenvalue}"
            );
            assert!(residual.is_nan(), "expected NaN residual, got {residual}");
        }
        other => panic!("expected Err(NonFinite), got {other:?}"),
    }
}

#[test]
fn lanczos_finite_unconverged_is_ok_not_error() {
    // A non-trivial spectrum capped at a single iteration cannot reach `tol`,
    // but the result is finite. The finiteness gate must NOT be mistaken for a
    // convergence gate: this is a valid best-Ritz-pair result, reported as
    // `Ok` with `converged == false`, not an error.
    let n = 5;
    let diag = [-3.0_f64, -1.0, 0.0, 2.0, 5.0];
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        data[i + n * i] = diag[i];
    }
    let h = Host::shared().dense(data, vec![n, n]);

    let result = lanczos_smallest::<f64, _>(
        &|v: &DenseTensor<f64>| matvec_cm(&h, n, v),
        n,
        &LanczosParams {
            max_iter: 1,
            tol: 1e-12,
            seed: Some(3),
        },
    )
    .expect("a finite-but-imprecise run must return Ok, not Err");

    assert!(
        !result.converged,
        "expected converged = false at max_iter = 1"
    );
    assert!(
        result.eigenvalue.is_finite() && result.residual.is_finite(),
        "eigenvalue and residual must be finite: lambda = {}, residual = {}",
        result.eigenvalue,
        result.residual
    );
}
