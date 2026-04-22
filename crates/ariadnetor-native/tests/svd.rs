use arnet_core::backend::{ComputeBackend, ExecPolicy, SvdDescriptor};
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
fn test_svd_f64_square() {
    let backend = NativeBackend::new();

    // Logical A = [[1, 2], [3, 4]] (2x2)
    let a_logical = [1.0f64, 2.0, 3.0, 4.0];
    // Column-major: [1, 3, 2, 4]
    let a = to_col_major(&a_logical, 2, 2);
    let (m, n, k) = (2, 2, 2);
    let mut u = [0.0f64; 4]; // m x k
    let mut s = [0.0f64; 2]; // k
    let mut vt = [0.0f64; 4]; // k x n

    let desc = SvdDescriptor {
        m,
        n,
        a: &a,
        u: &mut u,
        s: &mut s,
        vt: &mut vt,
        policy: ExecPolicy::Sequential,
    };
    backend.svd(desc).unwrap();

    // Singular values should be positive and in descending order
    assert!(s[0] > s[1]);
    assert!(s[1] >= 0.0);

    // Reconstruct: A ~ U * diag(S) * Vt
    // All output matrices are column-major:
    //   U(m x k): element (i, l) at u[l * m + i]
    //   Vt(k x n): element (l, j) at vt[j * k + l]
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += u[l * m + i] * s[l] * vt[j * k + l];
            }
            let expected = a_logical[i * n + j];
            assert!(
                (val - expected).abs() < 1e-10,
                "Reconstruction mismatch at ({i},{j}): {val} vs {expected}",
            );
        }
    }
}

#[test]
fn test_svd_f64_rectangular() {
    let backend = NativeBackend::new();

    // Logical A = [[1, 2, 3], [4, 5, 6]] (2x3), k = min(2,3) = 2
    let a_logical = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (m, n, k) = (2, 3, 2);
    let a = to_col_major(&a_logical, m, n);
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
        policy: ExecPolicy::Sequential,
    };
    backend.svd(desc).unwrap();

    assert!(s[0] > s[1]);

    // Reconstruct A ~ U * diag(S) * Vt (column-major)
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += u[l * m + i] * s[l] * vt[j * k + l];
            }
            let expected = a_logical[i * n + j];
            assert!(
                (val - expected).abs() < 1e-10,
                "Reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_svd_f32_basic() {
    let backend = NativeBackend::new();

    let a_logical = [1.0f32, 2.0, 3.0, 4.0];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut u = [0.0f32; 4];
    let mut s = [0.0f32; 2];
    let mut vt = [0.0f32; 4];

    let desc = SvdDescriptor {
        m,
        n,
        a: &a,
        u: &mut u,
        s: &mut s,
        vt: &mut vt,
        policy: ExecPolicy::Sequential,
    };
    backend.svd(desc).unwrap();

    assert!(s[0] > s[1]);

    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0f32;
            for l in 0..k {
                val += u[l * m + i] * s[l] * vt[j * k + l];
            }
            let expected = a_logical[i * n + j];
            assert!(
                (val - expected).abs() < 1e-4,
                "Reconstruction mismatch at ({i},{j}): {val} vs {expected}",
            );
        }
    }
}

// --- Complex SVD tests ---

#[test]
fn test_svd_c64_hermitian() {
    let backend = NativeBackend::new();

    // Logical Hermitian matrix: A = [[2, 1-i], [1+i, 3]]
    let a_logical = [
        Complex::new(2.0, 0.0),
        Complex::new(1.0, -1.0),
        Complex::new(1.0, 1.0),
        Complex::new(3.0, 0.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
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
        policy: ExecPolicy::Sequential,
    };
    backend.svd(desc).unwrap();

    // Singular values should be positive and descending
    assert!(s[0] > s[1]);
    assert!(s[1] >= 0.0);

    // Reconstruct: A ~ U * diag(S) * Vt (column-major)
    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..k {
                val += u[l * m + i] * s[l] * vt[j * k + l];
            }
            let expected = a_logical[i * n + j];
            let diff = (val - expected).norm();
            assert!(
                diff < 1e-10,
                "SVD reconstruction mismatch at ({i},{j}): {val} vs {expected}",
            );
        }
    }
}

#[test]
fn test_svd_c64_rectangular() {
    let backend = NativeBackend::new();

    // Logical A (2x3) complex
    let a_logical = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(4.0, -1.0),
        Complex::new(2.0, 3.0),
        Complex::new(1.0, 1.0),
    ];
    let (m, n, k) = (2, 3, 2);
    let a = to_col_major(&a_logical, m, n);
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
        policy: ExecPolicy::Sequential,
    };
    backend.svd(desc).unwrap();

    assert!(s[0] > s[1]);

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..k {
                val += u[l * m + i] * s[l] * vt[j * k + l];
            }
            let expected = a_logical[i * n + j];
            let diff = (val - expected).norm();
            assert!(diff < 1e-10, "SVD reconstruction mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_svd_c64_unitary_check() {
    let backend = NativeBackend::new();

    // Verify U^H * U = I for SVD result
    let a_logical = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0),
        Complex::new(2.0, 1.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
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
        policy: ExecPolicy::Sequential,
    };
    backend.svd(desc).unwrap();

    // U^H * U should be identity (k x k)
    // U is column-major m x k: element (row, col) at u[col * m + row]
    for i in 0..k {
        for j in 0..k {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..m {
                // U(l, i) = u[i * m + l], U(l, j) = u[j * m + l]
                val += u[i * m + l].conj() * u[j * m + l];
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

    let a_logical = [
        Complex::new(2.0f32, 0.0),
        Complex::new(1.0, -1.0),
        Complex::new(1.0, 1.0),
        Complex::new(3.0, 0.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
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
        policy: ExecPolicy::Sequential,
    };
    backend.svd(desc).unwrap();

    assert!(s[0] > s[1]);

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0f32, 0.0);
            for l in 0..k {
                val += u[l * m + i] * s[l] * vt[j * k + l];
            }
            let expected = a_logical[i * n + j];
            let diff = (val - expected).norm();
            assert!(diff < 1e-4, "SVD reconstruction mismatch at ({i},{j})");
        }
    }
}
