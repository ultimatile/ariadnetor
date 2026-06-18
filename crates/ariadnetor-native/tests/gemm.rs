use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, ExecPolicy, GemmDescriptor, MemoryOrder};
use arnet_native::NativeBackend;
use num_complex::Complex;
use rstest::rstest;

#[test]
fn test_gemm_f64_identity() {
    let backend = NativeBackend::new();

    // A = [[1, 0], [0, 1]] (2x2 identity)
    let a = [1.0f64, 0.0, 0.0, 1.0];
    let b = [5.0f64, 6.0, 7.0, 8.0];
    let mut c = [0.0f64; 4];

    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: 1.0,
        a: &a,
        b: &b,
        beta: 0.0,
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();
    assert_eq!(c, [5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_gemm_f64_basic() {
    let backend = NativeBackend::new();

    // A = [[1, 2], [3, 4]] (2x2), B = [[5, 6], [7, 8]] (2x2)
    // C = A * B = [[19, 22], [43, 50]]
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [5.0f64, 6.0, 7.0, 8.0];
    let mut c = [0.0f64; 4];

    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: 1.0,
        a: &a,
        b: &b,
        beta: 0.0,
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_gemm_f64_alpha_beta() {
    let backend = NativeBackend::new();

    // C = 2.0 * A * B + 3.0 * C_init
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [5.0f64, 6.0, 7.0, 8.0];
    let mut c = [1.0f64; 4]; // C_init = all ones

    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: 2.0,
        a: &a,
        b: &b,
        beta: 3.0,
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();
    // C = 2 * [19, 22, 43, 50] + 3 * [1, 1, 1, 1] = [41, 47, 89, 103]
    assert_eq!(c, [41.0, 47.0, 89.0, 103.0]);
}

#[test]
fn test_gemm_f64_rectangular() {
    let backend = NativeBackend::new();

    // A (2x3) * B (3x2) = C (2x2)
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = [0.0f64; 4];

    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 3,
        alpha: 1.0,
        a: &a,
        b: &b,
        beta: 0.0,
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();
    // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
    // = [58, 64, 139, 154]
    assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_gemm_f32_basic() {
    let backend = NativeBackend::new();

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];

    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: 1.0,
        a: &a,
        b: &b,
        beta: 0.0,
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_gemm_f32_alpha_beta() {
    let backend = NativeBackend::new();
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut c = [2.0f32; 4];

    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: 2.0,
        a: &a,
        b: &b,
        beta: 3.0,
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();
    // 2*[19,22,43,50] + 3*[2,2,2,2] = [44,50,92,106]
    assert_eq!(c, [44.0, 50.0, 92.0, 106.0]);
}

// --- Complex GEMM tests ---

#[test]
fn test_gemm_c64_basic() {
    let backend = NativeBackend::new();

    // A = [[1+i, 2+i], [3+i, 4+i]], B = [[5+i, 6+i], [7+i, 8+i]]
    let a = [
        Complex::new(1.0, 1.0),
        Complex::new(2.0, 1.0),
        Complex::new(3.0, 1.0),
        Complex::new(4.0, 1.0),
    ];
    let b = [
        Complex::new(5.0, 1.0),
        Complex::new(6.0, 1.0),
        Complex::new(7.0, 1.0),
        Complex::new(8.0, 1.0),
    ];
    let mut c = [Complex::new(0.0, 0.0); 4];

    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: Complex::new(1.0, 0.0),
        a: &a,
        b: &b,
        beta: Complex::new(0.0, 0.0),
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();

    // C[0,0] = (1+i)(5+i) + (2+i)(7+i) = (4+6i) + (13+9i) = 17+15i
    assert!((c[0].re - 17.0f64).abs() < 1e-10);
    assert!((c[0].im - 15.0f64).abs() < 1e-10);
}

#[test]
fn test_gemm_c64_alpha_beta() {
    let backend = NativeBackend::new();

    // C = alpha * A * B + beta * C_init with complex alpha, beta
    let a = [
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    ];
    let b = [
        Complex::new(3.0, 4.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(3.0, 4.0),
    ];
    let mut c = [
        Complex::new(1.0, 1.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 1.0),
    ];

    // alpha = 2, beta = i -> C = 2*I*B + i*C_init = 2*B + i*C_init
    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: Complex::new(2.0, 0.0),
        a: &a,
        b: &b,
        beta: Complex::new(0.0, 1.0),
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();

    // C[0,0] = 2*(3+4i) + i*(1+i) = (6+8i) + (i+i^2) = (6+8i) + (-1+i) = 5+9i
    assert!((c[0].re - 5.0f64).abs() < 1e-10);
    assert!((c[0].im - 9.0f64).abs() < 1e-10);
}

#[test]
fn test_gemm_c32_basic() {
    let backend = NativeBackend::new();

    let a = [
        Complex::new(1.0f32, 1.0),
        Complex::new(2.0, 1.0),
        Complex::new(3.0, 1.0),
        Complex::new(4.0, 1.0),
    ];
    let b = [
        Complex::new(5.0f32, 1.0),
        Complex::new(6.0, 1.0),
        Complex::new(7.0, 1.0),
        Complex::new(8.0, 1.0),
    ];
    let mut c = [Complex::new(0.0f32, 0.0); 4];

    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: Complex::new(1.0, 0.0),
        a: &a,
        b: &b,
        beta: Complex::new(0.0, 0.0),
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();

    assert!((c[0].re - 17.0).abs() < 1e-4);
    assert!((c[0].im - 15.0).abs() < 1e-4);
}

#[test]
fn test_gemm_c32_alpha_beta() {
    let backend = NativeBackend::new();
    let a = [
        Complex::new(1.0f32, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    ];
    let b = [
        Complex::new(3.0f32, 4.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(3.0, 4.0),
    ];
    let mut c = [
        Complex::new(2.0f32, 3.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(2.0, 3.0),
    ];

    // C = 2*I*B + i*C_init
    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: Complex::new(2.0, 0.0),
        a: &a,
        b: &b,
        beta: Complex::new(0.0, 1.0),
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::RowMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();

    // C[0,0] = 2*(3+4i) + i*(2+3i) = (6+8i) + (-3+2i) = 3+10i
    assert!((c[0].re - 3.0).abs() < 1e-4);
    assert!((c[0].im - 10.0).abs() < 1e-4);
}

// --- ColumnMajor tests ---
// A = [[1,2],[3,4]] col-major: [1,3,2,4]
// B = [[5,6],[7,8]] col-major: [5,7,6,8]
// C = A*B = [[19,22],[43,50]] col-major: [19,43,22,50]

#[test]
fn test_gemm_f64_colmajor() {
    let backend = NativeBackend::new();
    let a = [1.0f64, 3.0, 2.0, 4.0];
    let b = [5.0f64, 7.0, 6.0, 8.0];
    let mut c = [2.0f64; 4]; // != 1.0 to distinguish * from /

    // C = 2*A*B + 3*C_init
    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: 2.0,
        a: &a,
        b: &b,
        beta: 3.0,
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();
    // 2*[19,43,22,50] + 3*[2,2,2,2] = [44,92,50,106]
    assert_eq!(c, [44.0, 92.0, 50.0, 106.0]);
}

#[test]
fn test_gemm_f32_colmajor() {
    let backend = NativeBackend::new();
    let a = [1.0f32, 3.0, 2.0, 4.0];
    let b = [5.0f32, 7.0, 6.0, 8.0];
    let mut c = [2.0f32; 4];

    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: 2.0,
        a: &a,
        b: &b,
        beta: 3.0,
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();
    assert_eq!(c, [44.0, 92.0, 50.0, 106.0]);
}

#[test]
fn test_gemm_c64_colmajor() {
    let backend = NativeBackend::new();
    let a = [
        Complex::new(1.0, 1.0),
        Complex::new(3.0, 1.0),
        Complex::new(2.0, 1.0),
        Complex::new(4.0, 1.0),
    ];
    let b = [
        Complex::new(5.0, 1.0),
        Complex::new(7.0, 1.0),
        Complex::new(6.0, 1.0),
        Complex::new(8.0, 1.0),
    ];
    let mut c = [Complex::new(2.0, 3.0); 4];

    // C = 2*A*B + i*C_init
    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: Complex::new(2.0, 0.0),
        a: &a,
        b: &b,
        beta: Complex::new(0.0, 1.0),
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();

    // C[0,0] = 2*(17+15i) + i*(2+3i) = (34+30i) + (-3+2i) = 31+32i
    assert!((c[0].re - 31.0f64).abs() < 1e-10);
    assert!((c[0].im - 32.0f64).abs() < 1e-10);
}

#[test]
fn test_gemm_c32_colmajor() {
    let backend = NativeBackend::new();
    let a = [
        Complex::new(1.0f32, 1.0),
        Complex::new(3.0, 1.0),
        Complex::new(2.0, 1.0),
        Complex::new(4.0, 1.0),
    ];
    let b = [
        Complex::new(5.0f32, 1.0),
        Complex::new(7.0, 1.0),
        Complex::new(6.0, 1.0),
        Complex::new(8.0, 1.0),
    ];
    let mut c = [Complex::new(2.0f32, 3.0); 4];

    let desc = GemmDescriptor {
        m: 2,
        n: 2,
        k: 2,
        alpha: Complex::new(2.0, 0.0),
        a: &a,
        b: &b,
        beta: Complex::new(0.0, 1.0),
        c: &mut c,
        trans_a: false,
        trans_b: false,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    };
    backend.gemm(desc).unwrap();

    // C[0,0] = 2*(17+15i) + i*(2+3i) = (34+30i) + (-3+2i) = 31+32i
    assert!((c[0].re - 31.0).abs() < 1e-3);
    assert!((c[0].im - 32.0).abs() < 1e-3);
}

// --- Transpose × layout contract (issue #103) ---
//
// GEMM computes C[m×n] = op(A)[m×k] * op(B)[k×n], where `trans_X` means the
// stored buffer holds the transpose of `op(X)` and `order` sets each buffer's
// memory layout. The result must be independent of layout and correct for
// every transpose-flag combination. The tests below check all 8 combinations
// of (trans_a, trans_b, order) against a naive triple-loop reference, for each
// scalar kernel — the transpose-dimension fix was duplicated across all four,
// so each is covered independently.

/// Flat offset of logical element `(i, j)` in a `rows`×`cols` matrix laid out per `order`.
fn layout_index(i: usize, j: usize, rows: usize, cols: usize, order: MemoryOrder) -> usize {
    match order {
        MemoryOrder::RowMajor => i * cols + j,
        MemoryOrder::ColumnMajor => j * rows + i,
    }
}

/// Flatten a logical `rows`×`cols` matrix (row-major in `data`) into a buffer
/// laid out per `order`. This is the inverse of the decode the kernel performs.
fn encode<T: Scalar>(data: &[T], rows: usize, cols: usize, order: MemoryOrder) -> Vec<T> {
    let mut buf = vec![T::zero(); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            buf[layout_index(i, j, rows, cols, order)] = data[i * cols + j];
        }
    }
    buf
}

/// Inverse of [`encode`]: read an `order`-laid-out buffer back into row-major form.
fn decode<T: Scalar>(buf: &[T], rows: usize, cols: usize, order: MemoryOrder) -> Vec<T> {
    let mut out = vec![T::zero(); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = buf[layout_index(i, j, rows, cols, order)];
        }
    }
    out
}

/// Transpose a row-major `rows`×`cols` logical matrix into a row-major `cols`×`rows` one.
fn transpose_logical<T: Scalar>(data: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut t = vec![T::zero(); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = data[i * cols + j];
        }
    }
    t
}

/// Verify GEMM matches a naive reference for one `(trans_a, trans_b, order)` combination.
fn check_gemm_combination<T: Scalar>(
    trans_a: bool,
    trans_b: bool,
    order: MemoryOrder,
    mk: fn(f64) -> T,
) {
    // All dimensions distinct: m != k and n != k make the column-major
    // transposed paths dimension-sensitive, so a regression of the transpose-
    // dimension fix is caught. A square example would make the two dimension
    // orderings identical and silently hide such a bug.
    let (m, n, k) = (2usize, 4usize, 3usize);

    let op_a: Vec<T> = (1..=m * k).map(|x| mk(x as f64)).collect(); // m×k, row-major
    let op_b: Vec<T> = (1..=k * n).map(|x| mk(x as f64)).collect(); // k×n, row-major

    // Reference C = op(A) * op(B), m×n.
    let mut reference = vec![T::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = T::zero();
            for p in 0..k {
                acc = acc + op_a[i * k + p] * op_b[p * n + j];
            }
            reference[i * n + j] = acc;
        }
    }

    // The A buffer stores op(A) (m×k) or its transpose (k×m), encoded per `order`.
    let a_buf = if trans_a {
        encode(&transpose_logical(&op_a, m, k), k, m, order)
    } else {
        encode(&op_a, m, k, order)
    };
    let b_buf = if trans_b {
        encode(&transpose_logical(&op_b, k, n), n, k, order)
    } else {
        encode(&op_b, k, n, order)
    };

    let mut c_buf = vec![T::zero(); m * n];
    let desc = GemmDescriptor {
        m,
        n,
        k,
        alpha: T::one(),
        a: &a_buf,
        b: &b_buf,
        beta: T::zero(),
        c: &mut c_buf,
        trans_a,
        trans_b,
        order,
        policy: ExecPolicy::Sequential,
    };
    NativeBackend::new().gemm(desc).unwrap();

    // `Scalar` has no `Sub`; form the difference via `Add` + `Mul<Real>`.
    // Inputs are integer-valued, so the arithmetic is exact and an epsilon
    // tolerance suffices; `Scalar::abs` covers both real and complex scalars.
    let got = decode(&c_buf, m, n, order);
    let neg_one = -<T::Real as num_traits::One>::one();
    for (g, r) in got.iter().zip(reference.iter()) {
        let err = (*g + r.scale_real(neg_one)).abs();
        assert!(
            err <= <T::Real as num_traits::Float>::epsilon(),
            "GEMM mismatch: trans_a={trans_a}, trans_b={trans_b}, order={order:?}"
        );
    }
}

#[rstest]
fn gemm_layout_transpose_invariance_f64(
    #[values(false, true)] trans_a: bool,
    #[values(false, true)] trans_b: bool,
    #[values(MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)] order: MemoryOrder,
) {
    check_gemm_combination::<f64>(trans_a, trans_b, order, |x| x);
}

#[rstest]
fn gemm_layout_transpose_invariance_f32(
    #[values(false, true)] trans_a: bool,
    #[values(false, true)] trans_b: bool,
    #[values(MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)] order: MemoryOrder,
) {
    check_gemm_combination::<f32>(trans_a, trans_b, order, |x| x as f32);
}

#[rstest]
fn gemm_layout_transpose_invariance_c64(
    #[values(false, true)] trans_a: bool,
    #[values(false, true)] trans_b: bool,
    #[values(MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)] order: MemoryOrder,
) {
    check_gemm_combination::<Complex<f64>>(trans_a, trans_b, order, |x| Complex::new(x, 0.0));
}

#[rstest]
fn gemm_layout_transpose_invariance_c32(
    #[values(false, true)] trans_a: bool,
    #[values(false, true)] trans_b: bool,
    #[values(MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)] order: MemoryOrder,
) {
    check_gemm_combination::<Complex<f32>>(trans_a, trans_b, order, |x| {
        Complex::new(x as f32, 0.0)
    });
}
