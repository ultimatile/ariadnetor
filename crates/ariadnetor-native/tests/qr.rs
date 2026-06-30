use ariadnetor_core::backend::{
    BackendError, ComputeBackend, ExecPolicy, MemoryOrder, QrDescriptor,
};
use ariadnetor_native::NativeBackend;
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
fn test_qr_f64_square() {
    let backend = NativeBackend::new();

    // Logical A = [[1, 2], [3, 4]] (2x2)
    let a_logical = [1.0f64, 2.0, 3.0, 4.0];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut q = [0.0f64; 4]; // m x k
    let mut r = [0.0f64; 4]; // k x n

    let desc = QrDescriptor {
        m,
        n,
        a: &a,
        q: &mut q,
        r: &mut r,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.qr(desc).unwrap();

    // Reconstruct: A ~ Q * R (column-major)
    //   Q(m x k): element (i, l) at q[l * m + i]
    //   R(k x n): element (l, j) at r[j * k + l]
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += q[l * m + i] * r[j * k + l];
            }
            let expected = a_logical[i * n + j];
            assert!(
                (val - expected).abs() < 1e-10,
                "QR reconstruction mismatch at ({i},{j}): {val} vs {expected}",
            );
        }
    }
}

#[test]
fn test_qr_f64_rectangular() {
    let backend = NativeBackend::new();

    // Logical A (3x2), k = min(3,2) = 2
    let a_logical = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (m, n, k) = (3, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut q = vec![0.0f64; m * k];
    let mut r = vec![0.0f64; k * n];

    let desc = QrDescriptor {
        m,
        n,
        a: &a,
        q: &mut q,
        r: &mut r,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.qr(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += q[l * m + i] * r[j * k + l];
            }
            let expected = a_logical[i * n + j];
            assert!(
                (val - expected).abs() < 1e-10,
                "QR reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_qr_f32_basic() {
    let backend = NativeBackend::new();

    let a_logical = [1.0f32, 2.0, 3.0, 4.0];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut q = [0.0f32; 4];
    let mut r = [0.0f32; 4];

    let desc = QrDescriptor {
        m,
        n,
        a: &a,
        q: &mut q,
        r: &mut r,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.qr(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0f32;
            for l in 0..k {
                val += q[l * m + i] * r[j * k + l];
            }
            let expected = a_logical[i * n + j];
            assert!(
                (val - expected).abs() < 1e-4,
                "QR reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

// --- Complex QR tests ---

#[test]
fn test_qr_c64_square() {
    let backend = NativeBackend::new();

    let a_logical = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0),
        Complex::new(2.0, 1.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut q = vec![Complex::new(0.0, 0.0); m * k];
    let mut r = vec![Complex::new(0.0, 0.0); k * n];

    let desc = QrDescriptor {
        m,
        n,
        a: &a,
        q: &mut q,
        r: &mut r,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.qr(desc).unwrap();

    // Reconstruct: A ~ Q * R (column-major)
    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..k {
                val += q[l * m + i] * r[j * k + l];
            }
            let expected = a_logical[i * n + j];
            let diff = (val - expected).norm();
            assert!(
                diff < 1e-10,
                "QR reconstruction mismatch at ({i},{j}): {val} vs {expected}",
            );
        }
    }

    // Q should be unitary: Q^H * Q = I
    // Q is column-major m x k: element (row, col) at q[col * m + row]
    for i in 0..k {
        for j in 0..k {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..m {
                val += q[i * m + l].conj() * q[j * m + l];
            }
            let expected: f64 = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val.norm() - expected).abs() < 1e-10,
                "Q^H * Q not identity at ({i},{j}): {val}"
            );
        }
    }
}

#[test]
fn test_qr_c64_rectangular() {
    let backend = NativeBackend::new();

    // Logical A (3x2) complex
    let a_logical = [
        Complex::new(1.0, 1.0),
        Complex::new(2.0, -1.0),
        Complex::new(3.0, 0.0),
        Complex::new(0.0, 4.0),
        Complex::new(-1.0, 2.0),
        Complex::new(5.0, 1.0),
    ];
    let (m, n, k) = (3, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut q = vec![Complex::new(0.0, 0.0); m * k];
    let mut r = vec![Complex::new(0.0, 0.0); k * n];

    let desc = QrDescriptor {
        m,
        n,
        a: &a,
        q: &mut q,
        r: &mut r,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.qr(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for l in 0..k {
                val += q[l * m + i] * r[j * k + l];
            }
            let expected = a_logical[i * n + j];
            let diff = (val - expected).norm();
            assert!(diff < 1e-10, "QR reconstruction mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_qr_c32_basic() {
    let backend = NativeBackend::new();

    let a_logical = [
        Complex::new(1.0f32, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0),
        Complex::new(2.0, 1.0),
    ];
    let (m, n, k) = (2, 2, 2);
    let a = to_col_major(&a_logical, m, n);
    let mut q = vec![Complex::new(0.0f32, 0.0); m * k];
    let mut r = vec![Complex::new(0.0f32, 0.0); k * n];

    let desc = QrDescriptor {
        m,
        n,
        a: &a,
        q: &mut q,
        r: &mut r,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.qr(desc).unwrap();

    for i in 0..m {
        for j in 0..n {
            let mut val = Complex::new(0.0f32, 0.0);
            for l in 0..k {
                val += q[l * m + i] * r[j * k + l];
            }
            let expected = a_logical[i * n + j];
            let diff = (val - expected).norm();
            assert!(diff < 1e-4, "QR reconstruction mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_qr_rejects_row_major_order() {
    let backend = NativeBackend::new();
    let (m, n) = (2usize, 2usize);
    let a = [0.0f64; 4];
    let mut q = [0.0f64; 4];
    let mut r = [0.0f64; 4];

    let desc = QrDescriptor {
        m,
        n,
        a: &a,
        q: &mut q,
        r: &mut r,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    let result = backend.qr(desc);
    assert!(
        matches!(result, Err(BackendError::InvalidArgument(_))),
        "expected InvalidArgument for RowMajor QR, got {result:?}"
    );
}
