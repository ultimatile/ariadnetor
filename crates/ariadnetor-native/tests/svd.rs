use arnet_core::backend::{ComputeBackend, SvdDescriptor};
use arnet_native::NativeBackend;
use num_complex::Complex;

#[test]
fn test_svd_f64_square() {
    let backend = NativeBackend::new();

    // A = [[1, 2], [3, 4]] (2x2)
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let mut u = [0.0f64; 4]; // 2x2
    let mut s = [0.0f64; 2]; // 2
    let mut vt = [0.0f64; 4]; // 2x2

    let desc = SvdDescriptor {
        m: 2,
        n: 2,
        a: &a,
        u: &mut u,
        s: &mut s,
        vt: &mut vt,
    };
    backend.svd(desc).unwrap();

    // Singular values should be positive and in descending order
    assert!(s[0] > s[1]);
    assert!(s[1] >= 0.0);

    // Reconstruct: A ~ U * diag(S) * Vt
    for i in 0..2 {
        for j in 0..2 {
            let mut val = 0.0;
            for k in 0..2 {
                val += u[i * 2 + k] * s[k] * vt[k * 2 + j];
            }
            assert!(
                (val - a[i * 2 + j]).abs() < 1e-10,
                "Reconstruction mismatch at ({i},{j}): {val} vs {}",
                a[i * 2 + j]
            );
        }
    }
}

#[test]
fn test_svd_f64_rectangular() {
    let backend = NativeBackend::new();

    // A = [[1, 2, 3], [4, 5, 6]] (2x3), k = min(2,3) = 2
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (m, n, k) = (2, 3, 2);
    let mut u = vec![0.0f64; m * k];
    let mut s = vec![0.0f64; k];
    let mut vt = vec![0.0f64; k * n];

    let desc = SvdDescriptor {
        m,
        n,
        a: &a,
        u: &mut u,
        s: &mut s,
        vt: &mut vt,
    };
    backend.svd(desc).unwrap();

    assert!(s[0] > s[1]);

    // Reconstruct A ~ U * diag(S) * Vt
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += u[i * k + l] * s[l] * vt[l * n + j];
            }
            assert!(
                (val - a[i * n + j]).abs() < 1e-10,
                "Reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_svd_f32_basic() {
    let backend = NativeBackend::new();

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let mut u = [0.0f32; 4];
    let mut s = [0.0f32; 2];
    let mut vt = [0.0f32; 4];

    let desc = SvdDescriptor {
        m: 2,
        n: 2,
        a: &a,
        u: &mut u,
        s: &mut s,
        vt: &mut vt,
    };
    backend.svd(desc).unwrap();

    assert!(s[0] > s[1]);

    for i in 0..2 {
        for j in 0..2 {
            let mut val = 0.0f32;
            for k in 0..2 {
                val += u[i * 2 + k] * s[k] * vt[k * 2 + j];
            }
            assert!(
                (val - a[i * 2 + j]).abs() < 1e-4,
                "Reconstruction mismatch at ({i},{j}): {val} vs {}",
                a[i * 2 + j]
            );
        }
    }
}

// --- Complex SVD tests ---

#[test]
fn test_svd_c64_hermitian() {
    let backend = NativeBackend::new();

    // Hermitian matrix: A = [[2, 1-i], [1+i, 3]]
    let a = [
        Complex::new(2.0, 0.0),
        Complex::new(1.0, -1.0),
        Complex::new(1.0, 1.0),
        Complex::new(3.0, 0.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let mut u = vec![Complex::new(0.0, 0.0); m * k];
    let mut s = vec![0.0f64; k];
    let mut vt = vec![Complex::new(0.0, 0.0); k * n];

    let desc = SvdDescriptor {
        m,
        n,
        a: &a,
        u: &mut u,
        s: &mut s,
        vt: &mut vt,
    };
    backend.svd(desc).unwrap();

    // Singular values should be positive and descending
    assert!(s[0] > s[1]);
    assert!(s[1] >= 0.0);

    // Reconstruct: A ~ U * diag(S) * Vt (where Vt = V^H)
    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..k {
                val += u[i * k + l] * s[l] * vt[l * n + j];
            }
            let diff = (val - a[i * n + j]).norm();
            assert!(
                diff < 1e-10,
                "SVD reconstruction mismatch at ({i},{j}): {val} vs {}",
                a[i * n + j]
            );
        }
    }
}

#[test]
fn test_svd_c64_rectangular() {
    let backend = NativeBackend::new();

    // A (2x3) complex
    let a = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(4.0, -1.0),
        Complex::new(2.0, 3.0),
        Complex::new(1.0, 1.0),
    ];
    let (m, n, k) = (2, 3, 2);
    let mut u = vec![Complex::new(0.0, 0.0); m * k];
    let mut s = vec![0.0f64; k];
    let mut vt = vec![Complex::new(0.0, 0.0); k * n];

    let desc = SvdDescriptor {
        m,
        n,
        a: &a,
        u: &mut u,
        s: &mut s,
        vt: &mut vt,
    };
    backend.svd(desc).unwrap();

    assert!(s[0] > s[1]);

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..k {
                val += u[i * k + l] * s[l] * vt[l * n + j];
            }
            let diff = (val - a[i * n + j]).norm();
            assert!(diff < 1e-10, "SVD reconstruction mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_svd_c64_unitary_check() {
    let backend = NativeBackend::new();

    // Verify U^H * U = I for SVD result
    let a = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0),
        Complex::new(2.0, 1.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let mut u = vec![Complex::new(0.0, 0.0); m * k];
    let mut s = vec![0.0f64; k];
    let mut vt = vec![Complex::new(0.0, 0.0); k * n];

    let desc = SvdDescriptor {
        m,
        n,
        a: &a,
        u: &mut u,
        s: &mut s,
        vt: &mut vt,
    };
    backend.svd(desc).unwrap();

    // U^H * U should be identity (k*k)
    for i in 0..k {
        for j in 0..k {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..m {
                val += u[l * k + i].conj() * u[l * k + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                val.norm() - expected < 1e-10,
                "U^H * U not identity at ({i},{j}): {val}"
            );
        }
    }
}

#[test]
fn test_svd_c32_basic() {
    let backend = NativeBackend::new();

    let a = [
        Complex::new(2.0f32, 0.0),
        Complex::new(1.0, -1.0),
        Complex::new(1.0, 1.0),
        Complex::new(3.0, 0.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let mut u = vec![Complex::new(0.0f32, 0.0); m * k];
    let mut s = vec![0.0f32; k];
    let mut vt = vec![Complex::new(0.0f32, 0.0); k * n];

    let desc = SvdDescriptor {
        m,
        n,
        a: &a,
        u: &mut u,
        s: &mut s,
        vt: &mut vt,
    };
    backend.svd(desc).unwrap();

    assert!(s[0] > s[1]);

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0f32, 0.0);
            for l in 0..k {
                val += u[i * k + l] * s[l] * vt[l * n + j];
            }
            let diff = (val - a[i * n + j]).norm();
            assert!(diff < 1e-4, "SVD reconstruction mismatch at ({i},{j})");
        }
    }
}
