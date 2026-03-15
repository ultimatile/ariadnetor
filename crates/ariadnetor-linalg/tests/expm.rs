use arnet_cpu::CpuBackend;
use arnet_linalg::{contract, eigh, expm, expm_antihermitian, expm_hermitian, EighResult};
use arnet_tensor::DenseTensor;

#[test]
fn test_expm_hermitian_diagonal_f64() {
    let backend = CpuBackend::new();

    // exp(diag(1, 2)) = diag(e, e²)
    let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
    let result = expm_hermitian(&backend, &a, 1).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let e = std::f64::consts::E;
    assert!((result.get(&[0, 0]) - e).abs() < 1e-10);
    assert!(result.get(&[0, 1]).abs() < 1e-10);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - e * e).abs() < 1e-10);
}

#[test]
fn test_expm_hermitian_zero_f64() {
    let backend = CpuBackend::new();

    // exp(0) = I
    let a = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
    let result = expm_hermitian(&backend, &a, 1).unwrap();

    assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-10);
    assert!(result.get(&[0, 1]).abs() < 1e-10);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
}

#[test]
fn test_expm_hermitian_identity_f64() {
    let backend = CpuBackend::new();

    // exp(I) = e * I
    let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let result = expm_hermitian(&backend, &a, 1).unwrap();

    let e = std::f64::consts::E;
    assert!((result.get(&[0, 0]) - e).abs() < 1e-10);
    assert!(result.get(&[0, 1]).abs() < 1e-10);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - e).abs() < 1e-10);
}

#[test]
fn test_expm_hermitian_2x2_symmetric() {
    let backend = CpuBackend::new();

    // A = [[0, 1], [1, 0]] (Pauli X), eigenvalues ±1
    // exp(A) = cosh(1)*I + sinh(1)*A
    let a = DenseTensor::<f64>::from_data(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]);
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

    let backend = CpuBackend::new();

    // Hermitian: A = [[2, 1-i], [1+i, 3]]
    // eigenvalues from eigh: λ₁ ≈ 1.0, λ₂ ≈ 4.0
    // Verify exp(A) is Hermitian and exp(A) = V diag(exp(λ)) V†
    let a = DenseTensor::from_data(
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
    assert!((r01 - r10.conj()).norm() < 1e-10, "not Hermitian: r01={r01}, r10={r10}");

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
    let backend = CpuBackend::new();

    // exp(diag(1, 2)) = diag(e, e²)
    let a = DenseTensor::<f32>::from_data(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
    let result = expm_hermitian(&backend, &a, 1).unwrap();

    let e = std::f32::consts::E;
    assert!((result.get(&[0, 0]) - e).abs() < 1e-4);
    assert!((result.get(&[1, 1]) - e * e).abs() < 1e-4);
}

#[test]
fn test_expm_hermitian_invalid_nonsquare() {
    let backend = CpuBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert!(expm_hermitian(&backend, &a, 1).is_err());
}

#[test]
fn test_expm_hermitian_invalid_nrow() {
    let backend = CpuBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    assert!(expm_hermitian(&backend, &a, 0).is_err());
    assert!(expm_hermitian(&backend, &a, 2).is_err());
}

// --- expm_antihermitian tests ---

#[test]
fn test_expm_antihermitian_unitarity_c64() {
    use num_complex::Complex;
    use num_traits::Zero;

    let backend = CpuBackend::new();

    // Anti-Hermitian: A = [[i, 1], [-1, -i]] → A† = [[-i, -1], [1, i]] = -A
    let a = DenseTensor::from_data(
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
    let mut uh_data = vec![Complex::<f64>::zero(); 4];
    for i in 0..2 {
        for j in 0..2 {
            uh_data[i * 2 + j] = u.data()[j * 2 + i].conj();
        }
    }
    let u_dagger = DenseTensor::from_data(uh_data, vec![2, 2]);
    let product = contract(&backend, &u_dagger, &u, "ij,jk->ik").unwrap();

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

    let backend = CpuBackend::new();

    // A = -iσ_z * t = [[-it, 0], [0, it]] which is anti-Hermitian
    // exp(A) = [[exp(-it), 0], [0, exp(it)]]
    let t: f64 = 0.5;
    let a = DenseTensor::from_data(
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
fn test_expm_antihermitian_zero_c64() {
    use num_complex::Complex;
    use num_traits::Zero;

    let backend = CpuBackend::new();

    // exp(0) = I
    let a = DenseTensor::from_data(vec![Complex::<f64>::zero(); 4], vec![2, 2]);
    let result = expm_antihermitian(&backend, &a, 1).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let val = result.get(&[i, j]);
            assert!(
                (val - Complex::new(expected, 0.0)).norm() < 1e-10,
                "result[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_expm_antihermitian_real_type_error() {
    let backend = CpuBackend::new();

    // Real types should return error
    let a_f64 = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
    assert!(expm_antihermitian(&backend, &a_f64, 1).is_err());

    let a_f32 = DenseTensor::<f32>::from_data(vec![0.0; 4], vec![2, 2]);
    assert!(expm_antihermitian(&backend, &a_f32, 1).is_err());
}

#[test]
fn test_expm_antihermitian_invalid_nonsquare() {
    use num_complex::Complex;
    use num_traits::Zero;

    let backend = CpuBackend::new();
    let a = DenseTensor::from_data(vec![Complex::<f64>::zero(); 6], vec![2, 3]);
    assert!(expm_antihermitian(&backend, &a, 1).is_err());
}

// --- expm (general) tests ---

#[test]
fn test_expm_diagonal_f64() {
    let backend = CpuBackend::new();

    // exp(diag(1, 2)) = diag(e, e²)
    let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
    let result = expm(&backend, &a, 1).unwrap();

    let e = std::f64::consts::E;
    assert!((result.get(&[0, 0]) - e).abs() < 1e-10);
    assert!(result.get(&[0, 1]).abs() < 1e-10);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - e * e).abs() < 1e-10);
}

#[test]
fn test_expm_zero_f64() {
    let backend = CpuBackend::new();

    // exp(0) = I
    let a = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
    let result = expm(&backend, &a, 1).unwrap();

    assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-10);
    assert!(result.get(&[0, 1]).abs() < 1e-10);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
}

#[test]
fn test_expm_identity_f64() {
    let backend = CpuBackend::new();

    // exp(I) = e * I
    let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let result = expm(&backend, &a, 1).unwrap();

    let e = std::f64::consts::E;
    assert!((result.get(&[0, 0]) - e).abs() < 1e-10);
    assert!(result.get(&[0, 1]).abs() < 1e-10);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - e).abs() < 1e-10);
}

#[test]
fn test_expm_nilpotent_f64() {
    let backend = CpuBackend::new();

    // N = [[0, 1], [0, 0]] is nilpotent (N² = 0)
    // exp(N) = I + N = [[1, 1], [0, 1]]
    let a = DenseTensor::<f64>::from_data(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]);
    let result = expm(&backend, &a, 1).unwrap();

    assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-10);
    assert!((result.get(&[0, 1]) - 1.0).abs() < 1e-10);
    assert!(result.get(&[1, 0]).abs() < 1e-10);
    assert!((result.get(&[1, 1]) - 1.0).abs() < 1e-10);
}

#[test]
fn test_expm_general_2x2_f64() {
    let backend = CpuBackend::new();

    // A = [[1, 2], [3, 4]] — compare with eigendecomposition result
    // eigenvalues: λ = (5 ± √33) / 2
    // tr(exp(A)) = exp(λ₁) + exp(λ₂)
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
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
    let actual_det = result.get(&[0, 0]) * result.get(&[1, 1])
        - result.get(&[0, 1]) * result.get(&[1, 0]);
    assert!(
        (actual_det - expected_det).abs() < 1e-6,
        "det mismatch: actual={actual_det}, expected={expected_det}"
    );
}

#[test]
fn test_expm_general_3x3_f64() {
    let backend = CpuBackend::new();

    // A = [[0,1,0],[0,0,1],[0,0,0]] (upper triangular nilpotent, N³=0)
    // exp(A) = I + A + A²/2 = [[1,1,0.5],[0,1,1],[0,0,1]]
    let a = DenseTensor::<f64>::from_data(
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![3, 3],
    );
    let result = expm(&backend, &a, 1).unwrap();

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

    let backend = CpuBackend::new();

    // Complex diagonal: exp(diag(i, -i)) = diag(exp(i), exp(-i))
    let a = DenseTensor::from_data(
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
    let backend = CpuBackend::new();

    // A = 10*I — triggers scaling (||A||_1 = 10 > θ_13)
    // exp(10*I) = e^10 * I
    let a = DenseTensor::<f64>::from_data(vec![10.0, 0.0, 0.0, 10.0], vec![2, 2]);
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
    let backend = CpuBackend::new();

    let a = DenseTensor::<f32>::from_data(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
    let result = expm(&backend, &a, 1).unwrap();

    let e = std::f32::consts::E;
    assert!((result.get(&[0, 0]) - e).abs() < 1e-4);
    assert!((result.get(&[1, 1]) - e * e).abs() < 1e-3);
}

#[test]
fn test_expm_invalid_nonsquare() {
    let backend = CpuBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert!(expm(&backend, &a, 1).is_err());
}

#[test]
fn test_expm_invalid_nrow() {
    let backend = CpuBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    assert!(expm(&backend, &a, 0).is_err());
    assert!(expm(&backend, &a, 2).is_err());
}
