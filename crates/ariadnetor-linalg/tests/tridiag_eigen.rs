//! Contract tests for `tridiag_eigh_with_backend`.
//!
//! Eigenvector columns of a symmetric matrix are only defined up to
//! sign (and, under a degenerate eigenvalue, up to a rotation of the
//! eigenspace), so no test here compares eigenvector entries against a
//! reference decomposition. The sign-independent contract is asserted
//! instead: ascending eigenvalues matching the dense `eigh` oracle,
//! per-pair residuals `||T u_j - lambda_j u_j||`, and orthonormality
//! of the eigenvector matrix.

mod tridiag_fixtures;

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{eigh_with_backend, tridiag_eigh_with_backend};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::DenseTensor;
use num_traits::Float;
use tridiag_fixtures::{assemble_dense, fixture};

/// `y = T x` for the tridiagonal matrix defined by `d` / `e`, used to
/// compute residuals without trusting the assembled dense oracle.
fn tridiag_matvec<T: Float>(d: &[T], e: &[T], x: &[T]) -> Vec<T> {
    let n = d.len();
    (0..n)
        .map(|i| {
            let mut y = d[i] * x[i];
            if i > 0 {
                y = y + e[i - 1] * x[i - 1];
            }
            if i + 1 < n {
                y = y + e[i] * x[i + 1];
            }
            y
        })
        .collect()
}

/// Column `j` of the `[n, n]` eigenvector tensor via order-aware `get`.
fn column<T: Scalar>(v: &DenseTensor<T>, n: usize, j: usize) -> Vec<T> {
    (0..n).map(|i| v.get([i, j])).collect()
}

/// Full sign-independent contract check for one fixture: ascending
/// eigenvalues matching dense `eigh`, per-pair residuals, and
/// orthonormal eigenvectors.
fn check_contract<T>(d: &[T], e: &[T], tol: T)
where
    T: Scalar<Real = T> + Float + std::fmt::Display,
{
    let backend = NativeBackend::new();
    let n = d.len();

    let (w, v) = tridiag_eigh_with_backend(&backend, d, e).expect("tridiag_eigh");
    assert_eq!(w.shape(), &[n]);
    assert_eq!(v.shape(), &[n, n]);
    let w = w.data_slice();

    // Ascending order.
    for (j, pair) in w.windows(2).enumerate() {
        assert!(pair[0] <= pair[1], "eigenvalues not ascending at {j}");
    }

    // Eigenvalues match the dense eigh oracle on the assembled matrix.
    let (w_ref, _) = eigh_with_backend(&backend, &assemble_dense(d, e), 1).expect("dense eigh");
    for (j, (&got, &want)) in w.iter().zip(w_ref.data_slice().iter()).enumerate() {
        assert!(
            Float::abs(got - want) <= tol,
            "eigenvalue {j}: {got} vs dense {want}",
        );
    }

    // Per-pair residual ||T u_j - lambda_j u_j||_inf against the
    // tridiagonal matvec (independent of the assembled oracle).
    for (j, &wj) in w.iter().enumerate() {
        let u = column(&v, n, j);
        let tu = tridiag_matvec(d, e, &u);
        for (i, (&tui, &ui)) in tu.iter().zip(u.iter()).enumerate() {
            let r = Float::abs(tui - wj * ui);
            assert!(r <= tol, "residual [{i}] of pair {j}: {r}");
        }
    }

    // Orthonormality: V^T V = I.
    for j in 0..n {
        for k in 0..n {
            let dot = (0..n).fold(T::zero(), |acc, i| acc + v.get([i, j]) * v.get([i, k]));
            let want = if j == k { T::one() } else { T::zero() };
            assert!(
                Float::abs(dot - want) <= tol,
                "V^T V [{j}, {k}] = {dot}, want {want}",
            );
        }
    }
}

#[test]
fn tridiag_eigh_f64_contract_on_generic_tridiagonal() {
    let (d, e) = fixture::<f64>(12);
    check_contract(&d, &e, 1e-10);
}

#[test]
fn tridiag_eigh_f32_contract_on_generic_tridiagonal() {
    let (d, e) = fixture::<f32>(12);
    check_contract(&d, &e, 1e-4);
}

#[test]
fn tridiag_eigh_f64_contract_above_recursion_threshold() {
    // faer switches from the QR algorithm to divide-and-conquer at its
    // recursion threshold (128 by default); the small-n tests never
    // cross it, so this fixture pins the divide-and-conquer path — the
    // one whose scratch demand the wrapper's scratch-superset choice
    // must cover.
    let (d, e) = fixture::<f64>(160);
    check_contract(&d, &e, 1e-9);
}

#[test]
fn tridiag_eigh_n1_returns_the_diagonal() {
    let backend = NativeBackend::new();
    let (w, v) = tridiag_eigh_with_backend(&backend, &[3.5_f64], &[]).expect("n=1");
    assert_eq!(w.shape(), &[1]);
    assert_eq!(v.shape(), &[1, 1]);
    assert!((w.data_slice()[0] - 3.5).abs() < 1e-15);
    // The single eigenvector is unit length; its sign is unspecified.
    assert!((v.data_slice()[0].abs() - 1.0).abs() < 1e-15);
}

#[test]
fn tridiag_eigh_n2_matches_analytic_eigenvalues() {
    // T = [[1, 0.5], [0.5, 2]]: trace 3, det 1.75, so the
    // characteristic polynomial is l^2 - 3 l + 1.75 with roots
    // (3 -+ sqrt(2)) / 2.
    let backend = NativeBackend::new();
    let (w, _v) = tridiag_eigh_with_backend(&backend, &[1.0_f64, 2.0], &[0.5]).expect("n=2");
    let s = 2.0_f64.sqrt();
    assert!((w.data_slice()[0] - (3.0 - s) / 2.0).abs() < 1e-12);
    assert!((w.data_slice()[1] - (3.0 + s) / 2.0).abs() < 1e-12);
}

#[test]
fn tridiag_eigh_degenerate_spectrum_stays_orthonormal() {
    // Constant diagonal with zero subdiagonal: every eigenvalue is 2,
    // and any orthonormal basis of R^4 is a valid eigenvector set — so
    // only the sign-independent contract can be asserted.
    let d = vec![2.0_f64; 4];
    let e = vec![0.0_f64; 3];
    check_contract(&d, &e, 1e-12);
}

#[test]
fn tridiag_eigh_rejects_empty_diagonal() {
    let backend = NativeBackend::new();
    let err = tridiag_eigh_with_backend::<f64, _>(&backend, &[], &[]).unwrap_err();
    assert!(
        err.to_string().contains("non-empty diagonal"),
        "unexpected error: {err}",
    );
}

#[test]
fn tridiag_eigh_rejects_subdiagonal_length_mismatch() {
    let backend = NativeBackend::new();
    let err = tridiag_eigh_with_backend(&backend, &[1.0_f64, 2.0], &[0.5, 0.5]).unwrap_err();
    assert!(
        err.to_string().contains("subdiag length"),
        "unexpected error: {err}",
    );
}

#[test]
fn tridiag_eigh_rejects_complex_scalars_at_dispatch() {
    use ariadnetor_core::Complex;
    use ariadnetor_core::backend::{
        ComputeBackend, ExecPolicy, MemoryOrder, TridiagEighDescriptor,
    };

    // The public linalg entry points bound `T` to real scalars at
    // compile time, so a complex descriptor can only reach a backend
    // through `ComputeBackend::tridiag_eigh` directly. A general
    // complex symmetric tridiagonal matrix is not Hermitian, so the
    // dispatch layer rejects it instead of reinterpreting the data.
    let backend = NativeBackend::new();
    let d = vec![Complex::new(1.0_f64, 0.0), Complex::new(2.0, 0.0)];
    let e = vec![Complex::new(0.5_f64, 0.0)];
    let mut w = vec![Complex::new(0.0_f64, 0.0); 2];
    let mut v = vec![Complex::new(0.0_f64, 0.0); 4];
    let err = backend
        .tridiag_eigh(TridiagEighDescriptor {
            n: 2,
            d: &d,
            e: &e,
            w: &mut w,
            v: &mut v,
            order: MemoryOrder::ColumnMajor,
            policy: ExecPolicy::Sequential,
        })
        .unwrap_err();
    assert!(
        err.to_string().contains("real scalar"),
        "unexpected error: {err}",
    );
}
