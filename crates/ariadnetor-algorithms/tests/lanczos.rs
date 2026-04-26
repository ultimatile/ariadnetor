//! Tests for the Lanczos smallest-eigenvalue solver.
//!
//! Validation strategy: build a small dense Hermitian matrix, drive
//! the solver via a closure that does a matrix-vector product, and
//! compare against `arnet_linalg::eigh` ground truth.

use approx::assert_abs_diff_eq;
use arnet_algorithms::krylov::{LanczosParams, lanczos_smallest};
use arnet_core::Scalar;
use arnet_linalg::eigh;
use arnet_native::NativeBackend;
use arnet_tensor::Dense;
use num_complex::Complex;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Hermitian matrix-vector product `H v`. The matrix is stored in
/// column-major order (matching `NativeBackend::preferred_order()`):
/// element `(i, j)` lives at flat index `i + n * j`.
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

/// Build a random Hermitian matrix `H = (A + A^H) / 2` of size `n×n`,
/// stored in column-major.
fn random_hermitian_f64(n: usize, seed: u64) -> Dense<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let a = Dense::<f64>::random(vec![n, n], &mut rng);
    // Symmetrize: H[i,j] = (A[i,j] + A[j,i]) / 2.
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
            // H[i,j] = (A[i,j] + conj(A[j,i])) / 2 makes H Hermitian.
            let aij = Complex::new(r[i + n * j], im[i + n * j]);
            let aji = Complex::new(r[j + n * i], im[j + n * i]);
            data[i + n * j] = (aij + aji.conj()) * 0.5;
        }
    }
    Dense::new(data, vec![n, n])
}

/// Smallest eigenvalue of a Hermitian matrix via dense `eigh` (ground truth).
fn eigh_smallest<T: Scalar>(h: &Dense<T>) -> T::Real {
    let backend = NativeBackend::shared();
    let (eigvals, _) = eigh(&*backend, h, 1).expect("eigh");
    eigvals.data()[0]
}

// ---------------------------------------------------------------------------
// Real symmetric tests
// ---------------------------------------------------------------------------

#[test]
fn lanczos_diagonal_returns_min_eigenvalue() {
    // Diagonal matrix with eigenvalues {-3, -1, 0, 2, 5}; smallest is -3.
    let n = 5;
    let diag = [-3.0_f64, -1.0, 0.0, 2.0, 5.0];
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        data[i + n * i] = diag[i];
    }
    let h = Dense::new(data, vec![n, n]);

    let result = lanczos_smallest::<f64, _>(
        &|v: &Dense<f64>| matvec_cm(&h, n, v),
        n,
        &LanczosParams {
            max_iter: 50,
            tol: 1e-12,
            seed: Some(42),
        },
    );

    assert_abs_diff_eq!(result.eigenvalue, -3.0, epsilon = 1e-10);
    assert!(result.residual < 1e-9, "residual = {}", result.residual);
}

#[test]
fn lanczos_random_symmetric_matches_eigh() {
    // Compare smallest eigenvalue against dense eigh for several sizes.
    for &n in &[16usize, 64, 256] {
        let h = random_hermitian_f64(n, 0xC0FFEE + n as u64);
        let lambda_ref = eigh_smallest(&h);

        let result = lanczos_smallest::<f64, _>(
            &|v: &Dense<f64>| matvec_cm(&h, n, v),
            n,
            &LanczosParams {
                max_iter: 4 * n,
                tol: 1e-11,
                seed: Some(7),
            },
        );

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
    }
}

#[test]
fn lanczos_near_degenerate_cluster() {
    // Spectrum {0, 1e-6, 1e-6, 1.0}; the smallest is 0, with a near-degenerate
    // cluster at 1e-6. A correct solver returns lambda ≈ 0 within tol; the
    // returned eigenvector lies in the {0}-eigenspace (i.e. essentially
    // orthogonal to everything else).
    let n = 4;
    let diag = [0.0_f64, 1e-6, 1e-6, 1.0];
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        data[i + n * i] = diag[i];
    }
    let h = Dense::new(data, vec![n, n]);

    let result = lanczos_smallest::<f64, _>(
        &|v: &Dense<f64>| matvec_cm(&h, n, v),
        n,
        &LanczosParams {
            max_iter: 60,
            tol: 1e-12,
            seed: Some(11),
        },
    );

    assert!(
        result.eigenvalue.abs() < 1e-9,
        "lambda = {}",
        result.eigenvalue
    );
    // Eigenvector should be aligned with e_0 (up to a sign / phase).
    let v = result.eigenvector.data();
    assert!(v[0].abs() > 0.99, "v[0] = {} (expected near ±1)", v[0]);
    assert!(result.residual < 1e-7);
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
        &|v: &Dense<Complex<f64>>| matvec_cm(&h, n, v),
        n,
        &LanczosParams {
            max_iter: 4 * n,
            tol: 1e-11,
            seed: Some(99),
        },
    );

    let rel_err = (result.eigenvalue - lambda_ref).abs() / lambda_ref.abs().max(1.0);
    assert!(
        rel_err < 1e-9,
        "lambda = {}, ref = {lambda_ref}, rel_err = {rel_err}",
        result.eigenvalue
    );
    assert!(result.residual < 1e-7, "residual = {}", result.residual);
}
