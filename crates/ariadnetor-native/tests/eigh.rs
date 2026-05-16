//! Self-adjoint eigenvalue decomposition tests for all scalar types

use arnet_core::Scalar;
use arnet_core::backend::{BackendError, ComputeBackend, EighDescriptor, ExecPolicy, MemoryOrder};
use arnet_native::NativeBackend;
use num_complex::Complex;
use num_traits::One;

/// Verify eigh: eigenvalues are real, A * v_j ≈ w_j * v_j
/// Converts everything to Complex<f64> for uniform verification.
fn assert_eigh_laws<T: Scalar>(
    a_colmaj: &[T],
    n: usize,
    tol: f64,
    to_c64: impl Fn(T) -> Complex<f64>,
    real_to_f64: impl Fn(T::Real) -> f64,
) {
    let backend = NativeBackend::new();
    // Initialize with sentinel values so Ok(()) replacement is detectable
    let mut w = vec![T::Real::one(); n];
    let mut v = vec![T::one(); n * n];

    backend
        .eigh(EighDescriptor {
            n,
            a: a_colmaj,
            w: &mut w,
            v: &mut v,
            order: MemoryOrder::ColumnMajor,
            policy: ExecPolicy::Sequential,
        })
        .unwrap();

    let a64: Vec<Complex<f64>> = a_colmaj.iter().map(|&x| to_c64(x)).collect();
    let v64: Vec<Complex<f64>> = v.iter().map(|&x| to_c64(x)).collect();

    for j in 0..n {
        let wj = real_to_f64(w[j]);
        for i in 0..n {
            let mut av = Complex::new(0.0, 0.0);
            for k in 0..n {
                av += a64[k * n + i] * v64[j * n + k];
            }
            let wv = Complex::new(wj, 0.0) * v64[j * n + i];
            assert!(
                (av.re - wv.re).abs() < tol && (av.im - wv.im).abs() < tol,
                "A*v != w*v at i={i}, j={j}: av={av:?}, wv={wv:?}"
            );
        }
    }
}

// Symmetric 3x3: [[2,1,0],[1,3,1],[0,1,2]]
// Column-major: col0=[2,1,0], col1=[1,3,1], col2=[0,1,2]

#[test]
fn test_eigh_f64() {
    let a = [2.0f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    assert_eigh_laws(&a, 3, 1e-10, |x| Complex::new(x, 0.0), |x| x);
}

#[test]
fn test_eigh_f32() {
    let a = [2.0f32, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    assert_eigh_laws(&a, 3, 1e-4, |x| Complex::new(x as f64, 0.0), |x| x as f64);
}

#[test]
fn test_eigh_c64() {
    // Hermitian: [[2, 1-i, 0], [1+i, 3, 1-i], [0, 1+i, 2]]
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
    assert_eigh_laws(&a, 3, 1e-10, |x| x, |x| x);
}

#[test]
fn test_eigh_c32() {
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
    assert_eigh_laws(
        &a,
        3,
        1e-3,
        |x| Complex::new(x.re as f64, x.im as f64),
        |x| x as f64,
    );
}

#[test]
fn test_eigh_rejects_row_major_order() {
    let backend = NativeBackend::new();
    let n = 2usize;
    let a = [0.0f64; 4];
    let mut w = [0.0f64; 2];
    let mut v = [0.0f64; 4];

    let desc = EighDescriptor {
        n,
        a: &a,
        w: &mut w,
        v: &mut v,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    let result = backend.eigh(desc);
    assert!(
        matches!(result, Err(BackendError::InvalidArgument(_))),
        "expected InvalidArgument for RowMajor eigh, got {result:?}"
    );
}
