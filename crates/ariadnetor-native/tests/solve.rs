//! Linear solve tests for all scalar types

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, SolveDescriptor};
use arnet_native::NativeBackend;
use num_complex::Complex;

/// Verify solve for any Scalar: A * X ≈ B (column-major layout)
fn assert_solve_laws<T: Scalar>(
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    tol: f64,
    to_c64: impl Fn(T) -> Complex<f64>,
) {
    let backend = NativeBackend::new();
    let mut x = vec![T::zero(); n * nrhs];

    backend
        .solve(SolveDescriptor {
            n,
            nrhs,
            a,
            b,
            x: &mut x,
        })
        .unwrap();

    let a64: Vec<Complex<f64>> = a.iter().map(|&v| to_c64(v)).collect();
    let x64: Vec<Complex<f64>> = x.iter().map(|&v| to_c64(v)).collect();
    let b64: Vec<Complex<f64>> = b.iter().map(|&v| to_c64(v)).collect();

    for j in 0..nrhs {
        for i in 0..n {
            let mut ax = Complex::new(0.0, 0.0);
            for k in 0..n {
                ax = ax + a64[k * n + i] * x64[j * n + k];
            }
            assert!(
                (ax.re - b64[j * n + i].re).abs() < tol && (ax.im - b64[j * n + i].im).abs() < tol,
                "A*X != B at i={i}, j={j}: ax={ax:?}, b={b:?}",
                b = b64[j * n + i],
            );
        }
    }
}

// A = [[2,1,0],[1,3,1],[0,1,2]] (column-major), B = [[1],[0],[1]]

#[test]
fn test_solve_f64() {
    let a = [2.0f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    // nrhs=2, B column-major: col0=[1,0,1], col1=[0,1,0]
    let b = [1.0f64, 0.0, 1.0, 0.0, 1.0, 0.0];
    assert_solve_laws(&a, &b, 3, 2, 1e-10, |x| Complex::new(x, 0.0));
}

#[test]
fn test_solve_f32() {
    let a = [2.0f32, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let b = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
    assert_solve_laws(&a, &b, 3, 2, 1e-4, |x| Complex::new(x as f64, 0.0));
}

#[test]
fn test_solve_c64() {
    let a: Vec<Complex<f64>> = vec![
        Complex::new(2.0, 0.0),
        Complex::new(1.0, 1.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, -1.0),
        Complex::new(3.0, 0.0),
        Complex::new(1.0, 1.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, -1.0),
        Complex::new(2.0, 0.0),
    ];
    // nrhs=2
    let b = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    assert_solve_laws(&a, &b, 3, 2, 1e-10, |x| x);
}

#[test]
fn test_solve_c32() {
    let a: Vec<Complex<f32>> = vec![
        Complex::new(2.0, 0.0),
        Complex::new(1.0, 1.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, -1.0),
        Complex::new(3.0, 0.0),
        Complex::new(1.0, 1.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, -1.0),
        Complex::new(2.0, 0.0),
    ];
    let b = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    assert_solve_laws(&a, &b, 3, 2, 1e-3, |x| {
        Complex::new(x.re as f64, x.im as f64)
    });
}
