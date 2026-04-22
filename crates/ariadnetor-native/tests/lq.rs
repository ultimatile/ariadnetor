use arnet_core::backend::{ComputeBackend, ExecPolicy, LqDescriptor};
use arnet_native::NativeBackend;
use num_complex::Complex;

/// Convert a logical matrix (given in row-major order) to column-major layout.
/// The logical matrix has `rows` rows and `cols` columns.
fn to_col_major<T: Copy>(row_major: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut cm = vec![row_major[0]; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            cm[j * rows + i] = row_major[i * cols + j];
        }
    }
    cm
}

#[test]
fn test_lq_f64_square() {
    let backend = NativeBackend::new();

    // Logical A = [[1, 2], [3, 4]] (2x2)
    let a_logical = [1.0f64, 2.0, 3.0, 4.0];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut l = [0.0f64; 4]; // m x k
    let mut q = [0.0f64; 4]; // k x n

    let desc = LqDescriptor {
        m,
        n,
        a: &a,
        l: &mut l,
        q: &mut q,
        policy: ExecPolicy::Sequential,
    };
    backend.lq(desc).unwrap();

    // Reconstruct: A ~ L * Q (column-major)
    //   L(m x k): element (i, kk) at l[kk * m + i]
    //   Q(k x n): element (kk, j) at q[j * k + kk]
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for kk in 0..k {
                val += l[kk * m + i] * q[j * k + kk];
            }
            let expected = a_logical[i * n + j];
            assert!(
                (val - expected).abs() < 1e-10,
                "LQ reconstruction mismatch at ({i},{j}): {val} vs {expected}",
            );
        }
    }
}

#[test]
fn test_lq_f64_rectangular() {
    let backend = NativeBackend::new();

    // Logical A (2x3), k = min(2,3) = 2
    let a_logical = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (m, n, k) = (2, 3, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut l = vec![0.0f64; m * k];
    let mut q = vec![0.0f64; k * n];

    let desc = LqDescriptor {
        m,
        n,
        a: &a,
        l: &mut l,
        q: &mut q,
        policy: ExecPolicy::Sequential,
    };
    backend.lq(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for kk in 0..k {
                val += l[kk * m + i] * q[j * k + kk];
            }
            let expected = a_logical[i * n + j];
            assert!(
                (val - expected).abs() < 1e-10,
                "LQ reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_lq_f32_basic() {
    let backend = NativeBackend::new();

    let a_logical = [1.0f32, 2.0, 3.0, 4.0];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut l = [0.0f32; 4];
    let mut q = [0.0f32; 4];

    let desc = LqDescriptor {
        m,
        n,
        a: &a,
        l: &mut l,
        q: &mut q,
        policy: ExecPolicy::Sequential,
    };
    backend.lq(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0f32;
            for kk in 0..k {
                val += l[kk * m + i] * q[j * k + kk];
            }
            let expected = a_logical[i * n + j];
            assert!(
                (val - expected).abs() < 1e-4,
                "LQ reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

// --- Complex LQ tests ---

#[test]
fn test_lq_c64_square() {
    let backend = NativeBackend::new();

    let a_logical = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0),
        Complex::new(2.0, 1.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut l = vec![Complex::new(0.0, 0.0); m * k];
    let mut q = vec![Complex::new(0.0, 0.0); k * n];

    let desc = LqDescriptor {
        m,
        n,
        a: &a,
        l: &mut l,
        q: &mut q,
        policy: ExecPolicy::Sequential,
    };
    backend.lq(desc).unwrap();

    // Reconstruct: A ~ L * Q (column-major)
    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for kk in 0..k {
                val += l[kk * m + i] * q[j * k + kk];
            }
            let expected = a_logical[i * n + j];
            let diff = (val - expected).norm();
            assert!(
                diff < 1e-10,
                "LQ reconstruction mismatch at ({i},{j}): {val} vs {expected}",
            );
        }
    }

    // Q should have orthonormal rows: Q * Q^H = I
    // Q is column-major k x n: element (i, l_idx) at q[l_idx * k + i]
    for i in 0..k {
        for j in 0..k {
            let mut val = Complex::new(0.0, 0.0);
            for l_idx in 0..n {
                val += q[l_idx * k + i] * q[l_idx * k + j].conj();
            }
            let expected: f64 = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val.norm() - expected).abs() < 1e-10,
                "Q * Q^H not identity at ({i},{j}): {val}"
            );
        }
    }
}

#[test]
fn test_lq_c64_rectangular() {
    let backend = NativeBackend::new();

    // Logical A (2x3) complex
    let a_logical = [
        Complex::new(1.0, 1.0),
        Complex::new(2.0, -1.0),
        Complex::new(0.0, 3.0),
        Complex::new(4.0, 0.0),
        Complex::new(-1.0, 2.0),
        Complex::new(3.0, 1.0),
    ];
    let (m, n, k) = (2, 3, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut l = vec![Complex::new(0.0, 0.0); m * k];
    let mut q = vec![Complex::new(0.0, 0.0); k * n];

    let desc = LqDescriptor {
        m,
        n,
        a: &a,
        l: &mut l,
        q: &mut q,
        policy: ExecPolicy::Sequential,
    };
    backend.lq(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for kk in 0..k {
                val += l[kk * m + i] * q[j * k + kk];
            }
            let expected = a_logical[i * n + j];
            let diff = (val - expected).norm();
            assert!(diff < 1e-10, "LQ reconstruction mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_lq_c32_basic() {
    let backend = NativeBackend::new();

    let a_logical = [
        Complex::new(1.0f32, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0),
        Complex::new(2.0, 1.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut l = vec![Complex::new(0.0f32, 0.0); m * k];
    let mut q = vec![Complex::new(0.0f32, 0.0); k * n];

    let desc = LqDescriptor {
        m,
        n,
        a: &a,
        l: &mut l,
        q: &mut q,
        policy: ExecPolicy::Sequential,
    };
    backend.lq(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0f32, 0.0);
            for kk in 0..k {
                val += l[kk * m + i] * q[j * k + kk];
            }
            let expected = a_logical[i * n + j];
            let diff = (val - expected).norm();
            assert!(diff < 1e-4, "LQ reconstruction mismatch at ({i},{j})");
        }
    }
}
