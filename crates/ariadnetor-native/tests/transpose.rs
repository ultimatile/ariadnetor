use arnet_native::NativeBackend;
use arnet_core::backend::{ComputeBackend, DeviceType, TransposeDescriptor};
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
    };
    backend.transpose(desc).unwrap();
    assert_eq!(output, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_complex_f64_2d() {
    let backend = NativeBackend::new();

    let input = [
        Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0), Complex::new(9.0, 10.0), Complex::new(11.0, 12.0),
    ];
    let mut output = [Complex::new(0.0, 0.0); 6];

    let desc = TransposeDescriptor {
        input: &input,
        output: &mut output,
        shape: &[2, 3],
        perm: &[1, 0],
    };
    backend.transpose(desc).unwrap();
    assert_eq!(output[0], Complex::new(1.0, 2.0));
    assert_eq!(output[1], Complex::new(7.0, 8.0));
    assert_eq!(output[2], Complex::new(3.0, 4.0));
    assert_eq!(output[3], Complex::new(9.0, 10.0));
}
