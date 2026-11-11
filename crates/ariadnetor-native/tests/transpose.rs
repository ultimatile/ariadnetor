use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{
    BackendError, ComputeBackend, DeviceType, ExecPolicy, MemoryOrder, TransposeDescriptor,
};
use ariadnetor_native::NativeBackend;
use num_complex::Complex;

#[test]
fn test_backend_metadata() {
    let backend = NativeBackend::new();
    assert_eq!(backend.name(), "cpu");
    assert_eq!(backend.device_type(), DeviceType::Cpu);
    assert!(backend.is_available());
}

// --- Transpose tests ---

#[test]
fn test_transpose_f64_2d() {
    let backend = NativeBackend::new();

    // 2x3 matrix -> 3x2
    let input = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = [0.0f64; 6];

    let desc = TransposeDescriptor {
        input: &input,
        output: &mut output,
        shape: &[2, 3],
        perm: &[1, 0],
        order: MemoryOrder::RowMajor,
        conj: false,
        policy: ExecPolicy::Sequential,
    };
    backend.transpose(desc).unwrap();
    // [[1,2,3],[4,5,6]] transposed = [[1,4],[2,5],[3,6]]
    assert_eq!(output, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_f64_3d() {
    let backend = NativeBackend::new();

    // Shape [2,3,4], perm [1,0,2] -> shape [3,2,4]
    let input: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let mut output = vec![0.0f64; 24];

    let desc = TransposeDescriptor {
        input: &input,
        output: &mut output,
        shape: &[2, 3, 4],
        perm: &[1, 0, 2],
        order: MemoryOrder::RowMajor,
        conj: false,
        policy: ExecPolicy::Sequential,
    };
    backend.transpose(desc).unwrap();

    // Verify a few elements: input[i][j][k] should equal output[j][i][k]
    // input[0][1][2] = 0*12 + 1*4 + 2 = 6 -> output[1][0][2] = 1*8 + 0*4 + 2 = 10
    assert_eq!(output[10], 6.0);
    // input[1][0][3] = 1*12 + 0*4 + 3 = 15 -> output[0][1][3] = 0*8 + 1*4 + 3 = 7
    assert_eq!(output[7], 15.0);
}

#[test]
fn test_transpose_f32_2d() {
    let backend = NativeBackend::new();

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = [0.0f32; 6];

    let desc = TransposeDescriptor {
        input: &input,
        output: &mut output,
        shape: &[2, 3],
        perm: &[1, 0],
        order: MemoryOrder::RowMajor,
        conj: false,
        policy: ExecPolicy::Sequential,
    };
    backend.transpose(desc).unwrap();
    assert_eq!(output, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_complex_f64_2d() {
    let backend = NativeBackend::new();

    let input = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
        Complex::new(9.0, 10.0),
        Complex::new(11.0, 12.0),
    ];
    let mut output = [Complex::new(0.0, 0.0); 6];

    let desc = TransposeDescriptor {
        input: &input,
        output: &mut output,
        shape: &[2, 3],
        perm: &[1, 0],
        order: MemoryOrder::RowMajor,
        conj: false,
        policy: ExecPolicy::Sequential,
    };
    backend.transpose(desc).unwrap();
    assert_eq!(output[0], Complex::new(1.0, 2.0));
    assert_eq!(output[1], Complex::new(7.0, 8.0));
    assert_eq!(output[2], Complex::new(3.0, 4.0));
    assert_eq!(output[3], Complex::new(9.0, 10.0));
}

// --- Reference-checked transposes ---
//
// Every case below is compared with `reference`, which walks output
// coordinates one at a time and shares no code with the kernels. The perms
// are not involutions and the extents differ, so a kernel that used the
// inverse permutation, or read the wrong axis's extent, would fail.

trait Sample: Scalar + std::fmt::Debug + PartialEq {
    fn sample(i: usize) -> Self;
}

impl Sample for f64 {
    fn sample(i: usize) -> Self {
        i as f64 + 0.5
    }
}

impl Sample for f32 {
    fn sample(i: usize) -> Self {
        i as f32 + 0.5
    }
}

impl Sample for Complex<f64> {
    fn sample(i: usize) -> Self {
        Complex::new(i as f64, -(2.0 * i as f64) - 1.0)
    }
}

impl Sample for Complex<f32> {
    fn sample(i: usize) -> Self {
        Complex::new(i as f32, -(2.0 * i as f32) - 1.0)
    }
}

fn contiguous_strides(shape: &[usize], order: MemoryOrder) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut step = 1;
    let axes: Vec<usize> = match order {
        MemoryOrder::RowMajor => (0..shape.len()).rev().collect(),
        MemoryOrder::ColumnMajor => (0..shape.len()).collect(),
    };
    for axis in axes {
        strides[axis] = step;
        step *= shape[axis];
    }
    strides
}

/// `out[y] = input[x]` with `x[perm[a]] = y[a]`, conjugated when `conj`.
fn reference<T: Sample>(
    input: &[T],
    shape: &[usize],
    perm: &[usize],
    order: MemoryOrder,
    conj: bool,
) -> Vec<T> {
    let new_shape: Vec<usize> = perm.iter().map(|&a| shape[a]).collect();
    let in_strides = contiguous_strides(shape, order);
    let out_strides = contiguous_strides(&new_shape, order);
    let total: usize = shape.iter().product();
    let mut out = vec![T::sample(0); total];
    let mut y = vec![0usize; shape.len()];
    for _ in 0..total {
        let mut src = 0;
        let mut dst = 0;
        for a in 0..y.len() {
            src += y[a] * in_strides[perm[a]];
            dst += y[a] * out_strides[a];
        }
        out[dst] = if conj { input[src].conj() } else { input[src] };
        // Odometer increment over the output coordinates.
        for a in 0..y.len() {
            y[a] += 1;
            if y[a] < new_shape[a] {
                break;
            }
            y[a] = 0;
        }
    }
    out
}

fn run_transpose<T: Sample>(
    input: &[T],
    shape: &[usize],
    perm: &[usize],
    order: MemoryOrder,
    conj: bool,
    policy: ExecPolicy,
) -> Vec<T> {
    let mut output = vec![T::sample(usize::MAX / 4); input.len()];
    let desc = TransposeDescriptor {
        input,
        output: &mut output,
        shape,
        perm,
        order,
        conj,
        policy,
    };
    NativeBackend::new().transpose(desc).unwrap();
    output
}

const ORDERS: [MemoryOrder; 2] = [MemoryOrder::RowMajor, MemoryOrder::ColumnMajor];
const POLICIES: [ExecPolicy; 3] = [
    ExecPolicy::Sequential,
    ExecPolicy::Parallel(0),
    ExecPolicy::Parallel(2),
];

/// Rank 0 through 5, two zero-size shapes, and a 15_360-element case that
/// the naive kernel's parallel path splits into several chunks.
const CASES: &[(&[usize], &[usize])] = &[
    (&[], &[]),
    (&[5], &[0]),
    (&[3, 4], &[1, 0]),
    (&[2, 3, 4], &[2, 0, 1]),
    (&[2, 3, 4], &[1, 2, 0]),
    (&[3, 1, 4, 2], &[1, 3, 0, 2]),
    (&[2, 3, 1, 4, 5], &[4, 0, 3, 1, 2]),
    (&[0, 3, 2], &[2, 0, 1]),
    (&[3, 0], &[1, 0]),
    (&[16, 24, 40], &[2, 0, 1]),
];

fn check_against_reference<T: Sample>() {
    for &(shape, perm) in CASES {
        let total: usize = shape.iter().product();
        let input: Vec<T> = (0..total).map(T::sample).collect();
        for order in ORDERS {
            for conj in [false, true] {
                let expected = reference(&input, shape, perm, order, conj);
                for policy in POLICIES {
                    let got = run_transpose(&input, shape, perm, order, conj, policy);
                    assert_eq!(
                        got, expected,
                        "shape {shape:?} perm {perm:?} {order:?} conj {conj} {policy:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn f64_matches_reference() {
    check_against_reference::<f64>();
}

#[test]
fn f32_matches_reference() {
    check_against_reference::<f32>();
}

#[test]
fn c64_matches_reference() {
    check_against_reference::<Complex<f64>>();
}

#[test]
fn c32_matches_reference() {
    check_against_reference::<Complex<f32>>();
}

/// Above 1M elements strided-perm may split the copy across the pool; how
/// wide it goes depends on the host, so only the output is asserted.
fn check_large<T: Sample>() {
    let shape = [64usize, 48, 400];
    let perm = [2usize, 0, 1];
    let total: usize = shape.iter().product();
    let input: Vec<T> = (0..total).map(T::sample).collect();
    for order in ORDERS {
        let expected = reference(&input, &shape, &perm, order, false);
        for policy in [ExecPolicy::Sequential, ExecPolicy::Parallel(0)] {
            let got = run_transpose(&input, &shape, &perm, order, false, policy);
            assert!(got == expected, "{order:?} {policy:?}");
        }
    }
}

#[test]
fn large_f64_matches_reference() {
    check_large::<f64>();
}

#[test]
fn large_c64_matches_reference() {
    check_large::<Complex<f64>>();
}

// --- Descriptor validation ---

/// `(shape, perm, input length, output length)` for descriptors every entry
/// must reject before writing anything.
const INVALID: &[(&[usize], &[usize], usize, usize)] = &[
    (&[2, 3], &[0], 6, 6),
    (&[2, 3], &[1, 0, 2], 6, 6),
    (&[2, 3], &[0, 2], 6, 6),
    // Column-major `[0, 0]` on a square shape stays in bounds, so only the
    // permutation check can catch it.
    (&[3, 3], &[0, 0], 9, 9),
    (&[usize::MAX, 2], &[1, 0], 0, 0),
    (&[2, 3], &[1, 0], 5, 6),
    (&[2, 3], &[1, 0], 7, 6),
    (&[2, 3], &[1, 0], 6, 5),
    (&[2, 3], &[1, 0], 6, 7),
];

fn check_rejects<T: Sample>() {
    let sentinel = T::sample(999);
    for &(shape, perm, in_len, out_len) in INVALID {
        let input: Vec<T> = (0..in_len).map(T::sample).collect();
        for order in ORDERS {
            for conj in [false, true] {
                let mut output = vec![sentinel; out_len];
                let desc = TransposeDescriptor {
                    input: &input,
                    output: &mut output,
                    shape,
                    perm,
                    order,
                    conj,
                    policy: ExecPolicy::Sequential,
                };
                let err = NativeBackend::new().transpose(desc).unwrap_err();
                assert!(
                    matches!(err, BackendError::InvalidArgument(_)),
                    "shape {shape:?} perm {perm:?} lens {in_len}/{out_len}: {err:?}"
                );
                assert!(output.iter().all(|&v| v == sentinel));
            }
        }
    }
}

#[test]
fn f64_rejects_invalid_descriptors() {
    check_rejects::<f64>();
}

#[test]
fn f32_rejects_invalid_descriptors() {
    check_rejects::<f32>();
}

#[test]
fn c64_rejects_invalid_descriptors() {
    check_rejects::<Complex<f64>>();
}

#[test]
fn c32_rejects_invalid_descriptors() {
    check_rejects::<Complex<f32>>();
}
