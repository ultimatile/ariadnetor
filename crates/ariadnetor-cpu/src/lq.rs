//! LQ decomposition implementations via adjoint -> QR -> adjoint for all supported scalar types

use arnet_core::backend::{BackendError, LqDescriptor};
use faer::MatRef;
use num_complex::Complex;

/// Thin LQ for f64: A = L * Q, computed via QR of A^T
pub(crate) fn lq_f64(desc: LqDescriptor<'_, f64>) -> Result<(), BackendError> {
    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    // Transpose A (m*n, row-major) -> A^T (n*m)
    let a_mat = MatRef::from_row_major_slice(a, m, n);
    let at = a_mat.transpose().to_owned();

    // QR of A^T: A^T = Q_t * R_t where Q_t is n*k, R_t is k*m
    let qr = at.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    // A = (Q_t * R_t)^T = R_t^T * Q_t^T = L * Q
    // L = R_t^T (k*m transposed -> m*k, row-major)
    for i in 0..m {
        for j in 0..k {
            l[i * k + j] = r_t[(j, i)];
        }
    }

    // Q = Q_t^T (n*k transposed -> k*n, row-major)
    for i in 0..k {
        for j in 0..n {
            q[i * n + j] = q_t[(j, i)];
        }
    }

    Ok(())
}

/// Thin LQ for f32: A = L * Q, computed via QR of A^T
pub(crate) fn lq_f32(desc: LqDescriptor<'_, f32>) -> Result<(), BackendError> {
    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    let a_mat = MatRef::from_row_major_slice(a, m, n);
    let at = a_mat.transpose().to_owned();

    let qr = at.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    for i in 0..m {
        for j in 0..k {
            l[i * k + j] = r_t[(j, i)];
        }
    }

    for i in 0..k {
        for j in 0..n {
            q[i * n + j] = q_t[(j, i)];
        }
    }

    Ok(())
}

/// Thin LQ for Complex<f64>: A = L * Q, computed via QR of A^H
pub(crate) fn lq_c64(desc: LqDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    // A^H (n*m) via conjugate transpose
    let a_mat = MatRef::from_row_major_slice(a, m, n);
    let ah = a_mat.adjoint().to_owned();

    // QR of A^H: A^H = Q_t * R_t where Q_t is n*k, R_t is k*m
    let qr = ah.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    // A = (A^H)^H = (Q_t * R_t)^H = R_t^H * Q_t^H = L * Q
    // L = R_t^H (m*k)
    for i in 0..m {
        for j in 0..k {
            l[i * k + j] = r_t[(j, i)].conj();
        }
    }

    // Q = Q_t^H (k*n)
    for i in 0..k {
        for j in 0..n {
            q[i * n + j] = q_t[(j, i)].conj();
        }
    }

    Ok(())
}

/// Thin LQ for Complex<f32>: A = L * Q, computed via QR of A^H
pub(crate) fn lq_c32(desc: LqDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    let a_mat = MatRef::from_row_major_slice(a, m, n);
    let ah = a_mat.adjoint().to_owned();

    let qr = ah.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    for i in 0..m {
        for j in 0..k {
            l[i * k + j] = r_t[(j, i)].conj();
        }
    }

    for i in 0..k {
        for j in 0..n {
            q[i * n + j] = q_t[(j, i)].conj();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::CpuBackend;
    use arnet_core::backend::{ComputeBackend, LqDescriptor};
    use num_complex::Complex;

    #[test]
    fn test_lq_f64_square() {
        let backend = CpuBackend::new();

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
        let backend = CpuBackend::new();

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
        let backend = CpuBackend::new();

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
        let backend = CpuBackend::new();

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
        let backend = CpuBackend::new();

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
        let backend = CpuBackend::new();

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
}
