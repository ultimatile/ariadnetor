use arnet_linalg::{EighResult, contract, eigh, expm, expm_antihermitian, expm_hermitian};
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, MemoryOrder};

/// Create Dense from row-major data, converted to column-major for NativeBackend.
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> Dense<T> {
    let rm = Dense::new(data, shape);
    arnet_tensor::reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)
}

/// Convert column-major Dense back to row-major so `.get()` returns correct values.
fn to_rm<T: Clone>(tensor: &Dense<T>) -> Dense<T> {
    arnet_tensor::reorder(tensor, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor)
}

#[test]
fn test_expm_hermitian_diagonal_f64() {
    let backend = NativeBackend::new();

    // exp(diag(1, 2)) = diag(e, e²)
    let a = cm(vec![1.0_f64, 0.0, 0.0, 2.0], vec![2, 2]);
    let result = expm_hermitian(&backend, &a, 1).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let e = std::f64::consts::E;
    assert!((result.get(&[0, 0]) - e).abs() < 1e-10);
    assert!(result.get(&[0, 1]).abs() < 1e-10);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - e * e).abs() < 1e-10);
}

#[test]
fn test_expm_hermitian_2x2_symmetric() {
    let backend = NativeBackend::new();

    // A = [[0, 1], [1, 0]] (Pauli X), eigenvalues ±1
    // exp(A) = cosh(1)*I + sinh(1)*A
    let a = cm(vec![0.0_f64, 1.0, 1.0, 0.0], vec![2, 2]);
    let result = expm_hermitian(&backend, &a, 1).unwrap();

    let c = 1.0f64.cosh();
    let s = 1.0f64.sinh();
    assert!((result.get(&[0, 0]) - c).abs() < 1e-10);
    assert!((result.get(&[0, 1]) - s).abs() < 1e-10);
    assert!((result.get(&[1, 0]) - s).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - c).abs() < 1e-10);
}

#[test]
fn test_expm_hermitian_c64() {
    use num_complex::Complex;

    let backend = NativeBackend::new();

    // Hermitian: A = [[2, 1-i], [1+i, 3]]
    // eigenvalues from eigh: λ₁ ≈ 1.0, λ₂ ≈ 4.0
    // Verify exp(A) is Hermitian and exp(A) = V diag(exp(λ)) V†
    let a = cm(
        vec![
            Complex::new(2.0, 0.0),
            Complex::new(1.0, -1.0),
            Complex::new(1.0, 1.0),
            Complex::new(3.0, 0.0),
        ],
        vec![2, 2],
    );
    let result = expm_hermitian(&backend, &a, 1).unwrap();

    // exp(A) should be Hermitian: result[i,j] = conj(result[j,i])
    let r00 = result.get(&[0, 0]);
    let r01 = result.get(&[0, 1]);
    let r10 = result.get(&[1, 0]);
    let r11 = result.get(&[1, 1]);

    // Diagonal should be real
    assert!(f64::abs(r00.im) < 1e-10, "r00 not real: {r00}");
    assert!(f64::abs(r11.im) < 1e-10, "r11 not real: {r11}");

    // Off-diagonal: r01 = conj(r10)
    assert!(
        (r01 - r10.conj()).norm() < 1e-10,
        "not Hermitian: r01={r01}, r10={r10}"
    );

    // Verify via eigenvalue comparison: tr(exp(A)) = exp(λ₁) + exp(λ₂)
    let (w, _): EighResult<Complex<f64>> = eigh(&backend, &a, 1).unwrap();
    let expected_trace: f64 = w.data()[0].exp() + w.data()[1].exp();
    let actual_trace: f64 = r00.re + r11.re;
    assert!(
        f64::abs(actual_trace - expected_trace) < 1e-10,
        "trace mismatch: actual={actual_trace}, expected={expected_trace}"
    );
}

#[test]
fn test_expm_hermitian_f32() {
    let backend = NativeBackend::new();

    // exp(diag(1, 2)) = diag(e, e²)
    let a = cm(vec![1.0_f32, 0.0, 0.0, 2.0], vec![2, 2]);
    let result = expm_hermitian(&backend, &a, 1).unwrap();

    let e = std::f32::consts::E;
    assert!((result.get(&[0, 0]) - e).abs() < 1e-4);
    assert!((result.get(&[1, 1]) - e * e).abs() < 1e-4);
}

#[test]
fn test_expm_hermitian_invalid_nonsquare() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert!(expm_hermitian(&backend, &a, 1).is_err());
}

#[test]
fn test_expm_hermitian_invalid_nrow() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    assert!(expm_hermitian(&backend, &a, 0).is_err());
    assert!(expm_hermitian(&backend, &a, 2).is_err());
}

// --- expm_antihermitian tests ---

#[test]
fn test_expm_antihermitian_unitarity_c64() {
    use num_complex::Complex;
    use num_traits::Zero;

    let backend = NativeBackend::new();

    // Anti-Hermitian: A = [[i, 1], [-1, -i]] → A† = [[-i, -1], [1, i]] = -A
    let a = cm(
        vec![
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 0.0),
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, -1.0),
        ],
        vec![2, 2],
    );
    let u = expm_antihermitian(&backend, &a, 1).unwrap();

    // exp(A) should be unitary: U†U = I
    let u_rm = to_rm(&u);
    let mut uh_data = vec![Complex::<f64>::zero(); 4];
    for i in 0..2 {
        for j in 0..2 {
            uh_data[i * 2 + j] = u_rm.get(&[j, i]).conj();
        }
    }
    let u_dagger = cm(uh_data, vec![2, 2]);
    let product = to_rm(&contract(&backend, &u_dagger, &u, "ij,jk->ik").unwrap());

    // Should be identity
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let val = product.get(&[i, j]);
            assert!(
                (val - Complex::new(expected, 0.0)).norm() < 1e-10,
                "U†U[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_expm_antihermitian_pauli_z() {
    use num_complex::Complex;

    let backend = NativeBackend::new();

    // A = -iσ_z * t = [[-it, 0], [0, it]] which is anti-Hermitian
    // exp(A) = [[exp(-it), 0], [0, exp(it)]]
    let t: f64 = 0.5;
    let a = cm(
        vec![
            Complex::new(0.0, -t),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, t),
        ],
        vec![2, 2],
    );
    let result = expm_antihermitian(&backend, &a, 1).unwrap();

    // exp(-it) = cos(t) - i*sin(t)
    let exp_neg_it = Complex::new(t.cos(), -t.sin());
    let exp_pos_it = Complex::new(t.cos(), t.sin());

    assert!(
        (result.get(&[0, 0]) - exp_neg_it).norm() < 1e-10,
        "r00 = {}, expected {exp_neg_it}",
        result.get(&[0, 0])
    );
    assert!(result.get(&[0, 1]).norm() < 1e-10);
    assert!(result.get(&[1, 0]).norm() < 1e-10);
    assert!(
        (result.get(&[1, 1]) - exp_pos_it).norm() < 1e-10,
        "r11 = {}, expected {exp_pos_it}",
        result.get(&[1, 1])
    );
}

#[test]
fn test_expm_antihermitian_real_type_error() {
    let backend = NativeBackend::new();

    // Real types should return error
    let a_f64 = cm(vec![0.0_f64; 4], vec![2, 2]);
    assert!(expm_antihermitian(&backend, &a_f64, 1).is_err());

    let a_f32 = cm(vec![0.0_f32; 4], vec![2, 2]);
    assert!(expm_antihermitian(&backend, &a_f32, 1).is_err());
}

#[test]
fn test_expm_antihermitian_invalid_nonsquare() {
    use num_complex::Complex;
    use num_traits::Zero;

    let backend = NativeBackend::new();
    let a = Dense::new(vec![Complex::<f64>::zero(); 6], vec![2, 3]);
    assert!(expm_antihermitian(&backend, &a, 1).is_err());
}

// --- expm (general) tests ---

#[test]
fn test_expm_diagonal_f64() {
    let backend = NativeBackend::new();

    // exp(diag(1, 2)) = diag(e, e²)
    let a = cm(vec![1.0_f64, 0.0, 0.0, 2.0], vec![2, 2]);
    let result = expm(&backend, &a, 1).unwrap();

    let e = std::f64::consts::E;
    assert!((result.get(&[0, 0]) - e).abs() < 1e-10);
    assert!(result.get(&[0, 1]).abs() < 1e-10);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - e * e).abs() < 1e-10);
}

#[test]
fn test_expm_nilpotent_f64() {
    let backend = NativeBackend::new();

    // N = [[0, 1], [0, 0]] is nilpotent (N² = 0)
    // exp(N) = I + N = [[1, 1], [0, 1]]
    let a = cm(vec![0.0_f64, 1.0, 0.0, 0.0], vec![2, 2]);
    let result = to_rm(&expm(&backend, &a, 1).unwrap());

    assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-10);
    assert!((result.get(&[0, 1]) - 1.0).abs() < 1e-10);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
}

#[test]
fn test_expm_general_2x2_f64() {
    let backend = NativeBackend::new();

    // A = [[1, 2], [3, 4]] — compare with eigendecomposition result
    // eigenvalues: λ = (5 ± √33) / 2
    // tr(exp(A)) = exp(λ₁) + exp(λ₂)
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let result = expm(&backend, &a, 1).unwrap();

    let sqrt33 = 33.0f64.sqrt();
    let l1 = (5.0 - sqrt33) / 2.0;
    let l2 = (5.0 + sqrt33) / 2.0;
    let expected_trace = l1.exp() + l2.exp();
    let actual_trace = result.get(&[0, 0]) + result.get(&[1, 1]);
    assert!(
        (actual_trace - expected_trace).abs() < 1e-8,
        "trace mismatch: actual={actual_trace}, expected={expected_trace}"
    );

    // det(exp(A)) = exp(tr(A)) = exp(5)
    let expected_det = 5.0f64.exp();
    let actual_det =
        result.get(&[0, 0]) * result.get(&[1, 1]) - result.get(&[0, 1]) * result.get(&[1, 0]);
    assert!(
        (actual_det - expected_det).abs() < 1e-6,
        "det mismatch: actual={actual_det}, expected={expected_det}"
    );
}

#[test]
fn test_expm_general_3x3_f64() {
    let backend = NativeBackend::new();

    // A = [[0,1,0],[0,0,1],[0,0,0]] (upper triangular nilpotent, N³=0)
    // exp(A) = I + A + A²/2 = [[1,1,0.5],[0,1,1],[0,0,1]]
    let a = cm(
        vec![0.0_f64, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![3, 3],
    );
    let result = to_rm(&expm(&backend, &a, 1).unwrap());

    assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-10);
    assert!((result.get(&[0, 1]) - 1.0).abs() < 1e-10);
    assert!((result.get(&[0, 2]) - 0.5).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
    assert!((result.get(&[1, 2]) - 1.0).abs() < 1e-10);
    assert!((result.get(&[2, 2]) - 1.0).abs() < 1e-10);
}

#[test]
fn test_expm_complex_f64() {
    use num_complex::Complex;

    let backend = NativeBackend::new();

    // Complex diagonal: exp(diag(i, -i)) = diag(exp(i), exp(-i))
    let a = cm(
        vec![
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, -1.0),
        ],
        vec![2, 2],
    );
    let result = expm(&backend, &a, 1).unwrap();

    // exp(i) = cos(1) + i*sin(1)
    let exp_i = Complex::new(1.0f64.cos(), 1.0f64.sin());
    let exp_neg_i = Complex::new(1.0f64.cos(), -1.0f64.sin());

    assert!((result.get(&[0, 0]) - exp_i).norm() < 1e-10);
    assert!(result.get(&[0, 1]).norm() < 1e-10);
    assert!(result.get(&[1, 0]).norm() < 1e-10);
    assert!((result.get(&[1, 1]) - exp_neg_i).norm() < 1e-10);
}

#[test]
fn test_expm_large_norm_f64() {
    let backend = NativeBackend::new();

    // A = 10*I — triggers scaling (||A||_1 = 10 > θ_13)
    // exp(10*I) = e^10 * I
    let a = cm(vec![10.0_f64, 0.0, 0.0, 10.0], vec![2, 2]);
    let result = expm(&backend, &a, 1).unwrap();

    let e10 = 10.0f64.exp();
    assert!(
        (result.get(&[0, 0]) - e10).abs() / e10 < 1e-10,
        "r00 = {}, expected {e10}",
        result.get(&[0, 0])
    );
    assert!(result.get(&[0, 1]).abs() < 1e-5);
    assert!(result.get(&[1, 0]).abs() < 1e-5);
    assert!(
        (result.get(&[1, 1]) - e10).abs() / e10 < 1e-10,
        "r11 = {}, expected {e10}",
        result.get(&[1, 1])
    );
}

#[test]
fn test_expm_f32() {
    let backend = NativeBackend::new();

    let a = cm(vec![1.0_f32, 0.0, 0.0, 2.0], vec![2, 2]);
    let result = expm(&backend, &a, 1).unwrap();

    let e = std::f32::consts::E;
    assert!((result.get(&[0, 0]) - e).abs() < 1e-4);
    assert!((result.get(&[1, 1]) - e * e).abs() < 1e-3);
}

#[test]
fn test_expm_invalid_nonsquare() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert!(expm(&backend, &a, 1).is_err());
}

#[test]
fn test_expm_invalid_nrow() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    assert!(expm(&backend, &a, 0).is_err());
    assert!(expm(&backend, &a, 2).is_err());
}

// --- Mutation testing: norm_1, Pade orders, scaling/squaring ---

/// Helper to build a Dense matrix from a row-major flat vector.
fn mat(data: Vec<f64>, n: usize) -> Dense<f64> {
    cm(data, vec![n, n])
}

#[test]
fn test_expm_norm_1_known_matrices() {
    // norm_1 is the maximum absolute column sum.
    // We test indirectly: for a diagonal matrix diag(a, b),
    // ||diag(a,b)||_1 = max(|a|, |b|).
    // If we scale each column differently, we can verify the correct column
    // is selected as the max.
    let backend = NativeBackend::new();

    // A = [[1, 3], [2, 1]] → col sums: |1|+|2|=3, |3|+|1|=4 → norm_1 = 4
    // If norm_1 were wrong (e.g. row-based), we'd get different Pade order selection.
    // This matrix has norm_1 = 4 which is above theta_9=2.097 but below theta_13=5.371,
    // so it should use Pade 13 without scaling.
    let a = mat(vec![1.0, 3.0, 2.0, 1.0], 2);
    let result = expm(&backend, &a, 1).unwrap();

    // Verify via known eigenvalues: A has eigenvalues 1±sqrt(6)
    // tr(exp(A)) = exp(1+sqrt(6)) + exp(1-sqrt(6))
    let sq6 = 6.0f64.sqrt();
    let expected_trace = (1.0 + sq6).exp() + (1.0 - sq6).exp();
    let actual_trace = result.get(&[0, 0]) + result.get(&[1, 1]);
    assert!(
        (actual_trace - expected_trace).abs() < 1e-8,
        "trace mismatch: actual={actual_trace}, expected={expected_trace}"
    );
}

#[test]
fn test_expm_norm_1_asymmetric_columns() {
    let backend = NativeBackend::new();

    // A = [[0, 10], [0, 0]] → col sums: 0, 10 → norm_1 = 10
    // This is nilpotent: N^2 = 0, so exp(A) = I + A = [[1, 10], [0, 1]]
    // The norm_1=10 forces scaling/squaring path.
    let a = mat(vec![0.0, 10.0, 0.0, 0.0], 2);
    let result = to_rm(&expm(&backend, &a, 1).unwrap());

    assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-10);
    assert!((result.get(&[0, 1]) - 10.0).abs() < 1e-9);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
}

#[test]
fn test_expm_large_scaling_factor() {
    // Very large norm to test multiple squaring steps
    // A = 50 * I → exp(50*I) = e^50 * I
    let backend = NativeBackend::new();
    let a = mat(vec![50.0, 0.0, 0.0, 50.0], 2);
    let result = expm(&backend, &a, 1).unwrap();

    let e50 = 50.0f64.exp();
    assert!(
        (result.get(&[0, 0]) - e50).abs() / e50 < 1e-9,
        "r00={}, expected {e50}",
        result.get(&[0, 0])
    );
    assert!(result.get(&[0, 1]).abs() / e50 < 1e-10);
    assert!(result.get(&[1, 0]).abs() / e50 < 1e-10);
    assert!(
        (result.get(&[1, 1]) - e50).abs() / e50 < 1e-9,
        "r11={}, expected {e50}",
        result.get(&[1, 1])
    );
}

#[test]
fn test_expm_scaling_boundary_norm_equals_theta13() {
    // When norm_1 == theta_13, s should be 0 (no scaling needed)
    // theta_13 ≈ 5.371920351148152
    // Use A = theta_13 * [[1, 0], [0, 0]] → norm_1 = theta_13
    // exp(A) = [[exp(theta_13), 0], [0, 1]]
    let backend = NativeBackend::new();
    let theta13 = 5.371_920_351_148_152;
    let a = mat(vec![theta13, 0.0, 0.0, 0.0], 2);
    let result = expm(&backend, &a, 1).unwrap();

    let expected = theta13.exp();
    assert!(
        (result.get(&[0, 0]) - expected).abs() / expected < 1e-10,
        "r00={}, expected {expected}",
        result.get(&[0, 0])
    );
    assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
}

#[test]
fn test_expm_3x3_rotation() {
    // A general 3x3 matrix test to exercise Pade on larger sizes.
    // Use a skew-symmetric matrix (generates rotation).
    // A = [[0, -a, b], [a, 0, -c], [-b, c, 0]]
    // exp(A) should be orthogonal: R^T R = I
    let backend = NativeBackend::new();
    #[rustfmt::skip]
    let a = cm(
        vec![
            0.0_f64, -0.3,  0.5,
            0.3,      0.0, -0.7,
           -0.5,      0.7,  0.0,
        ],
        vec![3, 3]
    );
    let r = expm(&backend, &a, 1).unwrap();

    // Check R^T R ≈ I
    for i in 0..3 {
        for j in 0..3 {
            let mut dot = 0.0;
            for k in 0..3 {
                dot += r.get(&[k, i]) * r.get(&[k, j]);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-10,
                "R^T R[{i},{j}] = {dot}, expected {expected}"
            );
        }
    }

    // det(exp(A)) = exp(tr(A)) = exp(0) = 1 for skew-symmetric
    let det = r.get(&[0, 0]) * (r.get(&[1, 1]) * r.get(&[2, 2]) - r.get(&[1, 2]) * r.get(&[2, 1]))
        - r.get(&[0, 1]) * (r.get(&[1, 0]) * r.get(&[2, 2]) - r.get(&[1, 2]) * r.get(&[2, 0]))
        + r.get(&[0, 2]) * (r.get(&[1, 0]) * r.get(&[2, 1]) - r.get(&[1, 1]) * r.get(&[2, 0]));
    assert!(
        (det - 1.0).abs() < 1e-10,
        "det(exp(A)) = {det}, expected 1.0"
    );
}

#[test]
fn test_expm_different_pade_orders_agree() {
    // Scale the same base matrix to hit different Pade orders, and verify
    // they all give consistent results (exp(c*A) via different paths).
    // Use A = [[0,1],[-1,0]] (rotation generator).
    // exp(t*A) = [[cos(t), sin(t)], [-sin(t), cos(t)]]
    let backend = NativeBackend::new();

    // Norms that hit each Pade order threshold:
    // order 3: t=0.01, order 5: t=0.1, order 7: t=0.5
    // order 9: t=1.5, order 13 no scale: t=3.0, order 13 w/ scale: t=8.0
    let test_values = [0.01, 0.1, 0.5, 1.5, 3.0, 8.0];
    for &t in &test_values {
        let a = mat(vec![0.0, t, -t, 0.0], 2);
        let result = to_rm(&expm(&backend, &a, 1).unwrap());
        let c = t.cos();
        let s = t.sin();
        let expected = [[c, s], [-s, c]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &exp_ij) in row.iter().enumerate() {
                assert!(
                    (result.get(&[i, j]) - exp_ij).abs() < 1e-10,
                    "t={t}: r[{i},{j}]={}, expected {exp_ij}",
                    result.get(&[i, j]),
                );
            }
        }
    }
}

#[test]
fn test_expm_1x1_matrix() {
    // Edge case: 1x1 matrix, must be passed as [1,1] shape with nrow=1
    // Requires rank-2 tensor. exp([[x]]) = [[exp(x)]]
    let backend = NativeBackend::new();
    let a = cm(vec![2.0_f64], vec![1, 1]);
    let result = expm(&backend, &a, 1).unwrap();
    assert!(
        (result.get(&[0, 0]) - 2.0f64.exp()).abs() < 1e-12,
        "r00={}, expected {}",
        result.get(&[0, 0]),
        2.0f64.exp()
    );
}

#[test]
fn test_expm_3x3_large_lower_rows() {
    // Catches norm_1 index mutation (i*n → i/n): with n=3, mutated norm_1
    // reads only row 0 (tiny), missing the large entries in rows 1-2.
    // Correct norm_1 ≈ 20 (scaling needed), mutated ≈ 0.003 (Padé 3, wrong).
    let backend = NativeBackend::new();
    #[rustfmt::skip]
    let a = cm(
        vec![
            0.001_f64, 0.001,  0.001,
            0.0,       0.0,    0.0,
            10.0,      10.0,   10.0,
        ],
        vec![3, 3]
    );
    let result = expm(&backend, &a, 1).unwrap();

    // tr(exp(A)) = sum of exp(eigenvalues)
    // A has eigenvalues ≈ 0.001, 0, 10.001 (nearly diagonal-dominant)
    // tr(exp(A)) ≈ 1.001 + 1 + exp(10.001) ≈ 2 + 22028.5
    let trace = result.get(&[0, 0]) + result.get(&[1, 1]) + result.get(&[2, 2]);

    // det(exp(A)) = exp(tr(A)) = exp(0.001 + 0 + 10) ≈ exp(10.001)
    let expected_det = (10.001f64).exp();
    let actual_det = result.get(&[0, 0])
        * (result.get(&[1, 1]) * result.get(&[2, 2]) - result.get(&[1, 2]) * result.get(&[2, 1]))
        - result.get(&[0, 1])
            * (result.get(&[1, 0]) * result.get(&[2, 2])
                - result.get(&[1, 2]) * result.get(&[2, 0]))
        + result.get(&[0, 2])
            * (result.get(&[1, 0]) * result.get(&[2, 1])
                - result.get(&[1, 1]) * result.get(&[2, 0]));

    assert!(
        (actual_det - expected_det).abs() / expected_det < 1e-6,
        "det(exp(A)) = {actual_det}, expected {expected_det}"
    );
    assert!(
        trace > 22000.0,
        "trace too small: {trace} (expected > 22000)"
    );
}

#[test]
fn test_expm_satisfies_exp_property() {
    // exp(A+B) = exp(A)*exp(B) when [A,B]=0 (commuting matrices)
    // Use A = diag(1, 2), B = diag(3, 4) → both diagonal → they commute
    let backend = NativeBackend::new();
    let a = mat(vec![1.0, 0.0, 0.0, 2.0], 2);
    let b = mat(vec![3.0, 0.0, 0.0, 4.0], 2);
    let ab = mat(vec![4.0, 0.0, 0.0, 6.0], 2);

    let exp_a = expm(&backend, &a, 1).unwrap();
    let exp_b = expm(&backend, &b, 1).unwrap();
    let exp_ab = expm(&backend, &ab, 1).unwrap();
    let product = contract(&backend, &exp_a, &exp_b, "ij,jk->ik").unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (product.get(&[i, j]) - exp_ab.get(&[i, j])).abs()
                    / (1.0 + exp_ab.get(&[i, j]).abs())
                    < 1e-9,
                "exp(A+B) != exp(A)*exp(B) at ({i},{j})"
            );
        }
    }
}
