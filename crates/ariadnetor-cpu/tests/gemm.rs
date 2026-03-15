use arnet_cpu::CpuBackend;
use arnet_core::backend::{ComputeBackend, GemmDescriptor};
use num_complex::Complex;

#[test]
fn test_gemm_f64_identity() {
    let backend = CpuBackend::new();

    // A = [[1, 0], [0, 1]] (2x2 identity)
    let a = [1.0f64, 0.0, 0.0, 1.0];
    let b = [5.0f64, 6.0, 7.0, 8.0];
    let mut c = [0.0f64; 4];

    let desc = GemmDescriptor {
        m: 2, n: 2, k: 2,
        alpha: 1.0, a: &a, b: &b,
        beta: 0.0, c: &mut c,
        trans_a: false, trans_b: false,
    };
    backend.gemm(desc).unwrap();
    assert_eq!(c, [5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_gemm_f64_basic() {
    let backend = CpuBackend::new();

    // A = [[1, 2], [3, 4]] (2x2), B = [[5, 6], [7, 8]] (2x2)
    // C = A * B = [[19, 22], [43, 50]]
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [5.0f64, 6.0, 7.0, 8.0];
    let mut c = [0.0f64; 4];

    let desc = GemmDescriptor {
        m: 2, n: 2, k: 2,
        alpha: 1.0, a: &a, b: &b,
        beta: 0.0, c: &mut c,
        trans_a: false, trans_b: false,
    };
    backend.gemm(desc).unwrap();
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_gemm_f64_alpha_beta() {
    let backend = CpuBackend::new();

    // C = 2.0 * A * B + 3.0 * C_init
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [5.0f64, 6.0, 7.0, 8.0];
    let mut c = [1.0f64; 4]; // C_init = all ones

    let desc = GemmDescriptor {
        m: 2, n: 2, k: 2,
        alpha: 2.0, a: &a, b: &b,
        beta: 3.0, c: &mut c,
        trans_a: false, trans_b: false,
    };
    backend.gemm(desc).unwrap();
    // C = 2 * [19, 22, 43, 50] + 3 * [1, 1, 1, 1] = [41, 47, 89, 103]
    assert_eq!(c, [41.0, 47.0, 89.0, 103.0]);
}

#[test]
fn test_gemm_f64_rectangular() {
    let backend = CpuBackend::new();

    // A (2x3) * B (3x2) = C (2x2)
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = [0.0f64; 4];

    let desc = GemmDescriptor {
        m: 2, n: 2, k: 3,
        alpha: 1.0, a: &a, b: &b,
        beta: 0.0, c: &mut c,
        trans_a: false, trans_b: false,
    };
    backend.gemm(desc).unwrap();
    // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
    // = [58, 64, 139, 154]
    assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_gemm_f32_basic() {
    let backend = CpuBackend::new();

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];

    let desc = GemmDescriptor {
        m: 2, n: 2, k: 2,
        alpha: 1.0, a: &a, b: &b,
        beta: 0.0, c: &mut c,
        trans_a: false, trans_b: false,
    };
    backend.gemm(desc).unwrap();
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
}

// --- Complex GEMM tests ---

#[test]
fn test_gemm_c64_basic() {
    let backend = CpuBackend::new();

    // A = [[1+i, 2+i], [3+i, 4+i]], B = [[5+i, 6+i], [7+i, 8+i]]
    let a = [
        Complex::new(1.0, 1.0), Complex::new(2.0, 1.0),
        Complex::new(3.0, 1.0), Complex::new(4.0, 1.0),
    ];
    let b = [
        Complex::new(5.0, 1.0), Complex::new(6.0, 1.0),
        Complex::new(7.0, 1.0), Complex::new(8.0, 1.0),
    ];
    let mut c = [Complex::new(0.0, 0.0); 4];

    let desc = GemmDescriptor {
        m: 2, n: 2, k: 2,
        alpha: Complex::new(1.0, 0.0), a: &a, b: &b,
        beta: Complex::new(0.0, 0.0), c: &mut c,
        trans_a: false, trans_b: false,
    };
    backend.gemm(desc).unwrap();

    // C[0,0] = (1+i)(5+i) + (2+i)(7+i) = (4+6i) + (13+9i) = 17+15i
    assert!((c[0].re - 17.0f64).abs() < 1e-10);
    assert!((c[0].im - 15.0f64).abs() < 1e-10);
}

#[test]
fn test_gemm_c64_alpha_beta() {
    let backend = CpuBackend::new();

    // C = alpha * A * B + beta * C_init with complex alpha, beta
    let a = [
        Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
    ];
    let b = [
        Complex::new(3.0, 4.0), Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0), Complex::new(3.0, 4.0),
    ];
    let mut c = [
        Complex::new(1.0, 1.0), Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0), Complex::new(1.0, 1.0),
    ];

    // alpha = 2, beta = i -> C = 2*I*B + i*C_init = 2*B + i*C_init
    let desc = GemmDescriptor {
        m: 2, n: 2, k: 2,
        alpha: Complex::new(2.0, 0.0), a: &a, b: &b,
        beta: Complex::new(0.0, 1.0), c: &mut c,
        trans_a: false, trans_b: false,
    };
    backend.gemm(desc).unwrap();

    // C[0,0] = 2*(3+4i) + i*(1+i) = (6+8i) + (i+i^2) = (6+8i) + (-1+i) = 5+9i
    assert!((c[0].re - 5.0f64).abs() < 1e-10);
    assert!((c[0].im - 9.0f64).abs() < 1e-10);
}

#[test]
fn test_gemm_c32_basic() {
    let backend = CpuBackend::new();

    let a = [
        Complex::new(1.0f32, 1.0), Complex::new(2.0, 1.0),
        Complex::new(3.0, 1.0), Complex::new(4.0, 1.0),
    ];
    let b = [
        Complex::new(5.0f32, 1.0), Complex::new(6.0, 1.0),
        Complex::new(7.0, 1.0), Complex::new(8.0, 1.0),
    ];
    let mut c = [Complex::new(0.0f32, 0.0); 4];

    let desc = GemmDescriptor {
        m: 2, n: 2, k: 2,
        alpha: Complex::new(1.0, 0.0), a: &a, b: &b,
        beta: Complex::new(0.0, 0.0), c: &mut c,
        trans_a: false, trans_b: false,
    };
    backend.gemm(desc).unwrap();

    assert!((c[0].re - 17.0).abs() < 1e-4);
    assert!((c[0].im - 15.0).abs() < 1e-4);
}
