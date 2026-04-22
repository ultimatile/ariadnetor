//! General eigenvalue decomposition tests for all scalar types

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, EigDescriptor, ExecPolicy};
use arnet_native::NativeBackend;
use num_complex::Complex;
use num_traits::One;

/// Verify eig: A * v_j ≈ w_j * v_j for each eigenpair.
/// Converts everything to Complex<f64> for uniform verification.
fn assert_eig_laws<T: Scalar>(
    a_colmaj: &[T],
    n: usize,
    tol: f64,
    to_c64: impl Fn(T) -> Complex<f64>,
    complex_to_c64: impl Fn(T::Complex) -> Complex<f64>,
) {
    let backend = NativeBackend::new();
    // Initialize with sentinel values so Ok(()) replacement is detectable
    let mut w = vec![T::Complex::one(); n];
    let mut v = vec![T::Complex::one(); n * n];

    backend
        .eig(EigDescriptor {
            n,
            a: a_colmaj,
            w: &mut w,
            v: &mut v,
            policy: ExecPolicy::Sequential,
        })
        .unwrap();

    let a64: Vec<Complex<f64>> = a_colmaj.iter().map(|&x| to_c64(x)).collect();
    let w64: Vec<Complex<f64>> = w.iter().map(|&x| complex_to_c64(x)).collect();
    let v64: Vec<Complex<f64>> = v.iter().map(|&x| complex_to_c64(x)).collect();

    for j in 0..n {
        for i in 0..n {
            let mut av = Complex::new(0.0, 0.0);
            for k in 0..n {
                av = av + a64[k * n + i] * v64[j * n + k];
            }
            let wv = w64[j] * v64[j * n + i];
            assert!(
                (av.re - wv.re).abs() < tol && (av.im - wv.im).abs() < tol,
                "A*v != w*v at i={i}, j={j}: av={av:?}, wv={wv:?}"
            );
        }
    }
}

// Upper triangular 3x3: eigenvalues are diagonal entries [1, 4, 6]
// [[1, 2, 3], [0, 4, 5], [0, 0, 6]]
// Column-major: col0=[1,0,0], col1=[2,4,0], col2=[3,5,6]

fn f64_to_c64(x: f64) -> Complex<f64> {
    Complex::new(x, 0.0)
}
fn f32_to_c64(x: f32) -> Complex<f64> {
    Complex::new(x as f64, 0.0)
}
fn c64_to_c64(x: Complex<f64>) -> Complex<f64> {
    x
}
fn c32_to_c64(x: Complex<f32>) -> Complex<f64> {
    Complex::new(x.re as f64, x.im as f64)
}

#[test]
fn test_eig_f64() {
    let a = [1.0f64, 0.0, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0];
    assert_eig_laws(&a, 3, 1e-10, f64_to_c64, c64_to_c64);
}

#[test]
fn test_eig_f32() {
    let a = [1.0f32, 0.0, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0];
    assert_eig_laws(&a, 3, 1e-4, f32_to_c64, c32_to_c64);
}

#[test]
fn test_eig_c64() {
    let a: Vec<Complex<f64>> = vec![
        Complex::new(1.0, 0.5),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(4.0, -0.5),
        Complex::new(0.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(5.0, 0.0),
        Complex::new(6.0, 1.0),
    ];
    assert_eig_laws(&a, 3, 1e-10, c64_to_c64, c64_to_c64);
}

#[test]
fn test_eig_c32() {
    let a: Vec<Complex<f32>> = vec![
        Complex::new(1.0, 0.5),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(4.0, -0.5),
        Complex::new(0.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(5.0, 0.0),
        Complex::new(6.0, 1.0),
    ];
    assert_eig_laws(&a, 3, 1e-3, c32_to_c64, c32_to_c64);
}
