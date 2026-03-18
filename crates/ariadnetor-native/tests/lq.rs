use arnet_native::NativeBackend;
use arnet_core::backend::{ComputeBackend, LqDescriptor};
use num_complex::Complex;

#[test]
fn test_lq_f64_square() {
    let backend = NativeBackend::new();

    let a = [1.0f64, 2.0, 3.0, 4.0];
    let (m, n, k) = (2, 2, 2);
    let mut l = [0.0f64; 4];
    let mut q = [0.0f64; 4];

    let desc = LqDescriptor {
        m, n, a: &a,
        l: &mut l, q: &mut q,
    };
    backend.lq(desc).unwrap();

    // Reconstruct: A ~ L * Q
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for kk in 0..k {
                val += l[i * k + kk] * q[kk * n + j];
            }
            assert!((val - a[i * n + j]).abs() < 1e-10,
                "LQ reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * n + j]);
        }
    }
}

#[test]
fn test_lq_f64_rectangular() {
    let backend = NativeBackend::new();

    // A (2x3), k = min(2,3) = 2
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (m, n, k) = (2, 3, 2);
    let mut l = vec![0.0f64; m * k];
    let mut q = vec![0.0f64; k * n];

    let desc = LqDescriptor {
        m, n, a: &a,
        l: &mut l, q: &mut q,
    };
    backend.lq(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for kk in 0..k {
                val += l[i * k + kk] * q[kk * n + j];
            }
            assert!((val - a[i * n + j]).abs() < 1e-10,
                "LQ reconstruction mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_lq_f32_basic() {
    let backend = NativeBackend::new();

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let (m, n, k) = (2, 2, 2);
    let mut l = [0.0f32; 4];
    let mut q = [0.0f32; 4];

    let desc = LqDescriptor {
        m, n, a: &a,
        l: &mut l, q: &mut q,
    };
    backend.lq(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0f32;
            for kk in 0..k {
                val += l[i * k + kk] * q[kk * n + j];
            }
            assert!((val - a[i * n + j]).abs() < 1e-4,
                "LQ reconstruction mismatch at ({i},{j})");
        }
    }
}

// --- Complex LQ tests ---

#[test]
fn test_lq_c64_square() {
    let backend = NativeBackend::new();

    let a = [
        Complex::new(1.0, 2.0), Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0), Complex::new(2.0, 1.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let mut l = vec![Complex::new(0.0, 0.0); m * k];
    let mut q = vec![Complex::new(0.0, 0.0); k * n];

    let desc = LqDescriptor {
        m, n, a: &a,
        l: &mut l, q: &mut q,
    };
    backend.lq(desc).unwrap();

    // Reconstruct: A ~ L * Q
    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for kk in 0..k {
                val += l[i * k + kk] * q[kk * n + j];
            }
            let diff = (val - a[i * n + j]).norm();
            assert!(diff < 1e-10,
                "LQ reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * n + j]);
        }
    }

    // Q should have orthonormal rows: Q * Q^H = I
    for i in 0..k {
        for j in 0..k {
            let mut val = Complex::new(0.0, 0.0);
            for l_idx in 0..n {
                val += q[i * n + l_idx] * q[j * n + l_idx].conj();
            }
            let expected: f64 = if i == j { 1.0 } else { 0.0 };
            assert!((val.norm() - expected).abs() < 1e-10,
                "Q * Q^H not identity at ({i},{j}): {val}");
        }
    }
}

#[test]
fn test_lq_c64_rectangular() {
    let backend = NativeBackend::new();

    // A (2x3) complex
    let a = [
        Complex::new(1.0, 1.0), Complex::new(2.0, -1.0), Complex::new(0.0, 3.0),
        Complex::new(4.0, 0.0), Complex::new(-1.0, 2.0), Complex::new(3.0, 1.0),
    ];
    let (m, n, k) = (2, 3, 2);
    let mut l = vec![Complex::new(0.0, 0.0); m * k];
    let mut q = vec![Complex::new(0.0, 0.0); k * n];

    let desc = LqDescriptor {
        m, n, a: &a,
        l: &mut l, q: &mut q,
    };
    backend.lq(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for kk in 0..k {
                val += l[i * k + kk] * q[kk * n + j];
            }
            let diff = (val - a[i * n + j]).norm();
            assert!(diff < 1e-10,
                "LQ reconstruction mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_lq_c32_basic() {
    let backend = NativeBackend::new();

    let a = [
        Complex::new(1.0f32, 2.0), Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0), Complex::new(2.0, 1.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let mut l = vec![Complex::new(0.0f32, 0.0); m * k];
    let mut q = vec![Complex::new(0.0f32, 0.0); k * n];

    let desc = LqDescriptor {
        m, n, a: &a,
        l: &mut l, q: &mut q,
    };
    backend.lq(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0f32, 0.0);
            for kk in 0..k {
                val += l[i * k + kk] * q[kk * n + j];
            }
            let diff = (val - a[i * n + j]).norm();
            assert!(diff < 1e-4,
                "LQ reconstruction mismatch at ({i},{j})");
        }
    }
}
