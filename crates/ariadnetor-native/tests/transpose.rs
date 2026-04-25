use arnet_core::backend::{
    ComputeBackend, DeviceType, ExecPolicy, MemoryOrder, TransposeDescriptor,
};
use arnet_native::NativeBackend;
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

// --- Parallel-policy correctness ---
//
// These tests verify that the parallel branch of the naive fallback
// (and, with the `hptt` feature on, the HPTT path for f64) produces the
// same output as the sequential reference for every input. They are
// intentionally written feature-flag-agnostic: under `--features hptt`
// f64 routes through HPTT and these tests cover the HPTT parallel path;
// under `--no-default-features` f64 routes through `naive_parallel` and
// the same assertions cover the new Rayon kernel.

fn run_transpose_f64(
    backend: &NativeBackend,
    input: &[f64],
    shape: &[usize],
    perm: &[usize],
    order: MemoryOrder,
    policy: ExecPolicy,
) -> Vec<f64> {
    let total = shape.iter().product();
    let mut output = vec![0.0f64; total];
    let desc = TransposeDescriptor {
        input,
        output: &mut output,
        shape,
        perm,
        order,
        conj: false,
        policy,
    };
    backend.transpose(desc).unwrap();
    output
}

#[test]
fn parallel_policy_matches_sequential_2d() {
    let backend = NativeBackend::new();
    let input: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let shape = [2usize, 3];
    let perm = [1usize, 0];

    let seq = run_transpose_f64(
        &backend,
        &input,
        &shape,
        &perm,
        MemoryOrder::RowMajor,
        ExecPolicy::Sequential,
    );
    let par0 = run_transpose_f64(
        &backend,
        &input,
        &shape,
        &perm,
        MemoryOrder::RowMajor,
        ExecPolicy::Parallel(0),
    );
    let par2 = run_transpose_f64(
        &backend,
        &input,
        &shape,
        &perm,
        MemoryOrder::RowMajor,
        ExecPolicy::Parallel(2),
    );
    assert_eq!(seq, par0);
    assert_eq!(seq, par2);
}

#[test]
fn parallel_policy_matches_sequential_3d() {
    let backend = NativeBackend::new();
    let input: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let shape = [2usize, 3, 4];
    let perm = [1usize, 0, 2];

    let seq = run_transpose_f64(
        &backend,
        &input,
        &shape,
        &perm,
        MemoryOrder::RowMajor,
        ExecPolicy::Sequential,
    );
    let par = run_transpose_f64(
        &backend,
        &input,
        &shape,
        &perm,
        MemoryOrder::RowMajor,
        ExecPolicy::Parallel(0),
    );
    assert_eq!(seq, par);
}

// Large input forces multiple Rayon chunks under naive_parallel
// (chunk size has a 4096 floor, so 128*128 = 16384 elements always
// produces ≥4 chunks regardless of `current_num_threads()`).
#[test]
fn parallel_policy_handles_chunk_boundaries() {
    let backend = NativeBackend::new();
    let n = 128usize;
    let input: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
    let shape = [n, n];
    let perm = [1usize, 0];

    let seq = run_transpose_f64(
        &backend,
        &input,
        &shape,
        &perm,
        MemoryOrder::RowMajor,
        ExecPolicy::Sequential,
    );
    let par = run_transpose_f64(
        &backend,
        &input,
        &shape,
        &perm,
        MemoryOrder::RowMajor,
        ExecPolicy::Parallel(0),
    );
    assert_eq!(seq, par);
}

// Column-major exercises the other stride-computation branch in
// naive_parallel; tests RowMajor+ColumnMajor parity end-to-end.
#[test]
fn parallel_policy_matches_sequential_column_major() {
    let backend = NativeBackend::new();
    let input: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let shape = [2usize, 3, 4];
    let perm = [2usize, 0, 1];

    let seq = run_transpose_f64(
        &backend,
        &input,
        &shape,
        &perm,
        MemoryOrder::ColumnMajor,
        ExecPolicy::Sequential,
    );
    let par = run_transpose_f64(
        &backend,
        &input,
        &shape,
        &perm,
        MemoryOrder::ColumnMajor,
        ExecPolicy::Parallel(0),
    );
    assert_eq!(seq, par);
}

// Conjugation under parallel policy: complex elements must each be
// conjugated exactly once, regardless of which chunk owns them.
// Uses 128*128 = 16_384 elements so the input always exceeds
// `MIN_CHUNK = 4096` and the kernel produces multiple Rayon chunks,
// exercising the "conjugation across chunk boundaries" path.
#[test]
fn parallel_policy_conjugates_each_element_once() {
    let backend = NativeBackend::new();
    let n = 128usize;
    let input: Vec<Complex<f64>> = (0..n * n)
        .map(|i| Complex::new(i as f64, (2 * i + 1) as f64))
        .collect();
    let shape = [n, n];
    let perm = [1usize, 0];

    let mut out_seq = vec![Complex::new(0.0, 0.0); n * n];
    let desc_seq = TransposeDescriptor {
        input: &input,
        output: &mut out_seq,
        shape: &shape,
        perm: &perm,
        order: MemoryOrder::RowMajor,
        conj: true,
        policy: ExecPolicy::Sequential,
    };
    backend.transpose(desc_seq).unwrap();

    let mut out_par = vec![Complex::new(0.0, 0.0); n * n];
    let desc_par = TransposeDescriptor {
        input: &input,
        output: &mut out_par,
        shape: &shape,
        perm: &perm,
        order: MemoryOrder::RowMajor,
        conj: true,
        policy: ExecPolicy::Parallel(0),
    };
    backend.transpose(desc_par).unwrap();

    assert_eq!(out_seq, out_par);
}
