//! QR decomposition implementations via faer for all supported scalar types

use arnet_core::backend::{BackendError, QrDescriptor};
use faer::MatRef;
use num_complex::Complex;

/// Thin QR for f64 via faer: A = Q * R
pub(crate) fn qr_f64(desc: QrDescriptor<'_, f64>) -> Result<(), BackendError> {
    let QrDescriptor { m, n, a, q, r } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    // Q (m*k, row-major) -- thin Q
    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[i * k + j] = q_mat[(i, j)];
        }
    }

    // R (k*n, row-major) -- thin R
    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[i * n + j] = r_mat[(i, j)];
        }
    }

    Ok(())
}

/// Thin QR for f32 via faer: A = Q * R
pub(crate) fn qr_f32(desc: QrDescriptor<'_, f32>) -> Result<(), BackendError> {
    let QrDescriptor { m, n, a, q, r } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[i * k + j] = q_mat[(i, j)];
        }
    }

    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[i * n + j] = r_mat[(i, j)];
        }
    }

    Ok(())
}

/// Thin QR for Complex<f64> via faer: A = Q * R
pub(crate) fn qr_c64(desc: QrDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let QrDescriptor { m, n, a, q, r } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[i * k + j] = q_mat[(i, j)];
        }
    }

    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[i * n + j] = r_mat[(i, j)];
        }
    }

    Ok(())
}

/// Thin QR for Complex<f32> via faer: A = Q * R
pub(crate) fn qr_c32(desc: QrDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let QrDescriptor { m, n, a, q, r } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[i * k + j] = q_mat[(i, j)];
        }
    }

    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[i * n + j] = r_mat[(i, j)];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::CpuBackend;
    use arnet_core::backend::{ComputeBackend, QrDescriptor};
    use num_complex::Complex;

    #[test]
    fn test_qr_f64_square() {
        let backend = CpuBackend::new();

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
        let backend = CpuBackend::new();

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
        let backend = CpuBackend::new();

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
        let backend = CpuBackend::new();

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
        let backend = CpuBackend::new();

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
        let backend = CpuBackend::new();

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
}
