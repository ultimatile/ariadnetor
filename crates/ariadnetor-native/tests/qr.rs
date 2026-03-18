use arnet_native::NativeBackend;
use arnet_core::backend::{ComputeBackend, QrDescriptor};
use num_complex::Complex;

#[test]
fn test_qr_f64_square() {
    let backend = NativeBackend::new();

    let a = [1.0f64, 2.0, 3.0, 4.0];
    let (m, n, k) = (2, 2, 2);
    let mut q = [0.0f64; 4];
    let mut r = [0.0f64; 4];

    let desc = QrDescriptor {
        m, n, a: &a,
        q: &mut q, r: &mut r,
    };
    backend.qr(desc).unwrap();

    // Reconstruct: A ~ Q * R
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += q[i * k + l] * r[l * n + j];
            }
            assert!((val - a[i * n + j]).abs() < 1e-10,
                "QR reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * n + j]);
        }
    }
}

#[test]
fn test_qr_f64_rectangular() {
    let backend = NativeBackend::new();

    // A (3x2), k = min(3,2) = 2
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (m, n, k) = (3, 2, 2);
    let mut q = vec![0.0f64; m * k];
    let mut r = vec![0.0f64; k * n];

    let desc = QrDescriptor {
        m, n, a: &a,
        q: &mut q, r: &mut r,
    };
    backend.qr(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += q[i * k + l] * r[l * n + j];
            }
            assert!((val - a[i * n + j]).abs() < 1e-10,
                "QR reconstruction mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_qr_f32_basic() {
    let backend = NativeBackend::new();

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let (m, n, k) = (2, 2, 2);
    let mut q = [0.0f32; 4];
    let mut r = [0.0f32; 4];

    let desc = QrDescriptor {
        m, n, a: &a,
        q: &mut q, r: &mut r,
    };
    backend.qr(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0f32;
            for l in 0..k {
                val += q[i * k + l] * r[l * n + j];
            }
            assert!((val - a[i * n + j]).abs() < 1e-4,
                "QR reconstruction mismatch at ({i},{j})");
        }
    }
}

// --- Complex QR tests ---

#[test]
fn test_qr_c64_square() {
    let backend = NativeBackend::new();

    let a = [
        Complex::new(1.0, 2.0), Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0), Complex::new(2.0, 1.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let mut q = vec![Complex::new(0.0, 0.0); m * k];
    let mut r = vec![Complex::new(0.0, 0.0); k * n];

    let desc = QrDescriptor {
        m, n, a: &a,
        q: &mut q, r: &mut r,
    };
    backend.qr(desc).unwrap();

    // Reconstruct: A ~ Q * R
    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..k {
                val += q[i * k + l] * r[l * n + j];
            }
            let diff = (val - a[i * n + j]).norm();
            assert!(diff < 1e-10,
                "QR reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * n + j]);
        }
    }

    // Q should be unitary: Q^H * Q = I
    for i in 0..k {
        for j in 0..k {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..m {
                val += q[l * k + i].conj() * q[l * k + j];
            }
            let expected: f64 = if i == j { 1.0 } else { 0.0 };
            assert!((val.norm() - expected).abs() < 1e-10,
                "Q^H * Q not identity at ({i},{j}): {val}");
        }
    }
}

#[test]
fn test_qr_c64_rectangular() {
    let backend = NativeBackend::new();

    // A (3x2) complex
    let a = [
        Complex::new(1.0, 1.0), Complex::new(2.0, -1.0),
        Complex::new(3.0, 0.0), Complex::new(0.0, 4.0),
        Complex::new(-1.0, 2.0), Complex::new(5.0, 1.0),
    ];
    let (m, n, k) = (3, 2, 2);
    let mut q = vec![Complex::new(0.0, 0.0); m * k];
    let mut r = vec![Complex::new(0.0, 0.0); k * n];

    let desc = QrDescriptor {
        m, n, a: &a,
        q: &mut q, r: &mut r,
    };
    backend.qr(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..k {
                val += q[i * k + l] * r[l * n + j];
            }
            let diff = (val - a[i * n + j]).norm();
            assert!(diff < 1e-10,
                "QR reconstruction mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_qr_c32_basic() {
    let backend = NativeBackend::new();

    let a = [
        Complex::new(1.0f32, 2.0), Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0), Complex::new(2.0, 1.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let mut q = vec![Complex::new(0.0f32, 0.0); m * k];
    let mut r = vec![Complex::new(0.0f32, 0.0); k * n];

    let desc = QrDescriptor {
        m, n, a: &a,
        q: &mut q, r: &mut r,
    };
    backend.qr(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0f32, 0.0);
            for l in 0..k {
                val += q[i * k + l] * r[l * n + j];
            }
            let diff = (val - a[i * n + j]).norm();
            assert!(diff < 1e-4,
                "QR reconstruction mismatch at ({i},{j})");
        }
    }
}
