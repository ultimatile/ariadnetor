use arnet_linalg::{eig, eigh, eigvals, eigvalsh};
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, MemoryOrder};

/// Create Dense from row-major data, converted to column-major for NativeBackend.
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> Dense<T> {
    let rm = Dense::new(data, shape);
    arnet_tensor::reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)
}

/// Compute row-major flat index for (i, j) in shape [rows, cols]
fn rm(i: usize, j: usize, cols: usize) -> usize {
    i * cols + j
}

// --- EIGH tests ---

#[test]
fn test_eigh_f64_2x2_symmetric() {
    // [[2, 1], [1, 2]] → eigenvalues [1, 3]
    let backend = NativeBackend::new();
    let tensor = cm(vec![2.0_f64, 1.0, 1.0, 2.0], vec![2, 2]);

    let (w, v) = eigh(&backend, &tensor, 1).unwrap();

    assert_eq!(w.shape(), &[2]);
    assert_eq!(v.shape(), &[2, 2]);

    // Eigenvalues ascending: 1, 3
    assert!((w.data()[0] - 1.0).abs() < 1e-10);
    assert!((w.data()[1] - 3.0).abs() < 1e-10);

    // Eigenvectors should be orthogonal
    let v00 = v.get(&[0, 0]);
    let v10 = v.get(&[1, 0]);
    let v01 = v.get(&[0, 1]);
    let v11 = v.get(&[1, 1]);
    let dot = v00 * v01 + v10 * v11;
    assert!(dot.abs() < 1e-10, "Eigenvectors not orthogonal: dot={dot}");
}

#[test]
fn test_eigh_f64_3x3_diagonal() {
    // Diagonal matrix: eigenvalues = diagonal elements (sorted ascending)
    let backend = NativeBackend::new();
    let tensor = cm(
        vec![3.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
        vec![3, 3],
    );

    let (w, _v) = eigh(&backend, &tensor, 1).unwrap();

    assert_eq!(w.shape(), &[3]);
    assert!((w.data()[0] - 1.0).abs() < 1e-10);
    assert!((w.data()[1] - 2.0).abs() < 1e-10);
    assert!((w.data()[2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_eigh_c64_hermitian() {
    use num_complex::Complex;

    // Hermitian: [[2, 1-i], [1+i, 3]]
    let backend = NativeBackend::new();
    let tensor = cm(
        vec![
            Complex::new(2.0, 0.0),
            Complex::new(1.0, -1.0),
            Complex::new(1.0, 1.0),
            Complex::new(3.0, 0.0),
        ],
        vec![2, 2],
    );

    let (w, v) = eigh(&backend, &tensor, 1).unwrap();

    assert_eq!(w.shape(), &[2]);
    assert_eq!(v.shape(), &[2, 2]);

    // Eigenvalues: (5 ± sqrt(9))/2 → 1, 4
    // tr = 5, det = 6 - 2 = 4 → λ² - 5λ + 4 = 0 → λ = 1, 4
    let w0: f64 = w.data()[0];
    let w1: f64 = w.data()[1];
    assert!((w0 - 1.0).abs() < 1e-10);
    assert!((w1 - 4.0).abs() < 1e-10);

    // Eigenvectors should be orthogonal (V^H V = I)
    let v00 = v.get(&[0, 0]);
    let v10 = v.get(&[1, 0]);
    let v01 = v.get(&[0, 1]);
    let v11 = v.get(&[1, 1]);
    let dot = v00.conj() * v01 + v10.conj() * v11;
    assert!(dot.norm() < 1e-10, "Eigenvectors not orthogonal: dot={dot}");
}

#[test]
fn test_eigvalsh_f64() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![2.0_f64, 1.0, 1.0, 2.0], vec![2, 2]);

    let w = eigvalsh(&backend, &tensor, 1).unwrap();

    assert_eq!(w.shape(), &[2]);
    assert!((w.data()[0] - 1.0).abs() < 1e-10);
    assert!((w.data()[1] - 3.0).abs() < 1e-10);
}

#[test]
fn test_eigh_non_square_error() {
    let backend = NativeBackend::new();
    // 2×3 matrix → non-square → error
    let tensor = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let result = eigh(&backend, &tensor, 1);
    assert!(result.is_err());
}

#[test]
fn test_eigh_f32() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![2.0_f32, 1.0, 1.0, 2.0], vec![2, 2]);

    let (w, _v) = eigh(&backend, &tensor, 1).unwrap();

    assert_eq!(w.shape(), &[2]);
    assert!((w.data()[0] - 1.0).abs() < 1e-5);
    assert!((w.data()[1] - 3.0).abs() < 1e-5);
}

#[test]
fn test_eigh_invalid_nrow() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    assert!(eigh(&backend, &tensor, 0).is_err());
    assert!(eigh(&backend, &tensor, 2).is_err());
}

// --- EIG tests ---

#[test]
fn test_eig_f64_2x2_trace_det() {
    // [[1, 2], [3, 4]]
    // trace = 5, det = -2
    // eigenvalues satisfy: λ² - 5λ - 2 = 0
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let (w, v) = eig(&backend, &tensor, 1).unwrap();

    assert_eq!(w.shape(), &[2]);
    assert_eq!(v.shape(), &[2, 2]);

    // sum(eigenvalues) = trace = 5
    let sum = w.data()[0] + w.data()[1];
    assert!((sum.re - 5.0).abs() < 1e-10, "trace mismatch: {sum}");
    assert!(sum.im.abs() < 1e-10, "trace should be real: {sum}");

    // prod(eigenvalues) = det = -2
    let prod = w.data()[0] * w.data()[1];
    assert!((prod.re - (-2.0)).abs() < 1e-10, "det mismatch: {prod}");
    assert!(prod.im.abs() < 1e-10, "det should be real: {prod}");
}

#[test]
fn test_eig_f64_diagonal() {
    // Diagonal: [[3, 0], [0, 7]]
    // eigenvalues = {3, 7}
    let backend = NativeBackend::new();
    let tensor = cm(vec![3.0_f64, 0.0, 0.0, 7.0], vec![2, 2]);

    let (w, _v) = eig(&backend, &tensor, 1).unwrap();

    let mut eigs: Vec<f64> = (0..2).map(|i| w.data()[i].re).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((eigs[0] - 3.0).abs() < 1e-10);
    assert!((eigs[1] - 7.0).abs() < 1e-10);
}

#[test]
fn test_eig_c64_complex_input() {
    use num_complex::Complex;

    // Complex matrix: [[1+i, 2], [0, 3-i]]
    // Upper triangular → eigenvalues = diagonal = {1+i, 3-i}
    let backend = NativeBackend::new();
    let tensor = cm(
        vec![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, -1.0),
        ],
        vec![2, 2],
    );

    let (w, _v) = eig(&backend, &tensor, 1).unwrap();

    assert_eq!(w.shape(), &[2]);

    // Sort by real part for deterministic comparison
    let mut eigs: Vec<Complex<f64>> = (0..2).map(|i| w.data()[i]).collect();
    eigs.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

    assert!((eigs[0].re - 1.0).abs() < 1e-10);
    assert!((eigs[0].im - 1.0).abs() < 1e-10);
    assert!((eigs[1].re - 3.0).abs() < 1e-10);
    assert!((eigs[1].im - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_eigvals_f64() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let w = eigvals(&backend, &tensor, 1).unwrap();

    assert_eq!(w.shape(), &[2]);

    let sum = w.data()[0] + w.data()[1];
    assert!((sum.re - 5.0).abs() < 1e-10);
}

#[test]
fn test_eig_non_square_error() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    assert!(eig(&backend, &tensor, 1).is_err());
}

#[test]
fn test_eig_f32() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);

    let (w, _v) = eig(&backend, &tensor, 1).unwrap();

    assert_eq!(w.shape(), &[2]);

    // trace check
    let sum = w.data()[0] + w.data()[1];
    assert!((sum.re - 5.0).abs() < 1e-4, "trace mismatch: {sum}");
}

#[test]
fn test_eig_invalid_nrow() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    assert!(eig(&backend, &tensor, 0).is_err());
    assert!(eig(&backend, &tensor, 2).is_err());
}
