use arnet_tensor::DenseTensor;
use num_complex::Complex;

#[test]
fn test_tensor_creation() {
    let tensor = DenseTensor::<f64>::zeros(vec![3, 4]);
    assert_eq!(tensor.shape(), &[3, 4]);
    assert_eq!(tensor.len(), 12);
}

#[test]
fn test_tensor_from_data() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = DenseTensor::<f64>::from_data(data.clone(), vec![2, 2]);
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.data(), &data[..]);
}

#[test]
fn test_tensor_indexing() {
    let mut tensor = DenseTensor::<f64>::zeros(vec![3, 4]);
    tensor.set(&[1, 2], 42.0);
    assert_eq!(tensor.get(&[1, 2]), 42.0);
}

#[test]
fn test_tensor_fill() {
    let mut tensor = DenseTensor::<f64>::zeros(vec![2, 3]);
    tensor.fill(3.14);
    for &val in tensor.data() {
        assert_eq!(val, 3.14);
    }
}

#[test]
fn test_ones() {
    let tensor = DenseTensor::<f64>::ones(vec![2, 3]);
    for &val in tensor.data() {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_copy_on_write() {
    let tensor1 = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let mut tensor2 = tensor1.clone(); // Share data

    // Modification triggers CoW
    tensor2.set(&[0, 0], 999.0);

    // tensor1 should be unchanged
    assert_eq!(tensor1.get(&[0, 0]), 1.0);
    assert_eq!(tensor2.get(&[0, 0]), 999.0);
}

// Test different numeric types
#[test]
fn test_f32_tensor() {
    let tensor = DenseTensor::<f32>::zeros(vec![2, 3]);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.len(), 6);

    let tensor = DenseTensor::<f32>::ones(vec![2, 2]);
    for &val in tensor.data() {
        assert_eq!(val, 1.0f32);
    }
}

#[test]
fn test_complex_f64_tensor() {
    let tensor = DenseTensor::<Complex<f64>>::zeros(vec![2, 2]);
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.len(), 4);
    for &val in tensor.data() {
        assert_eq!(val, Complex::new(0.0, 0.0));
    }

    let tensor = DenseTensor::<Complex<f64>>::ones(vec![2, 2]);
    for &val in tensor.data() {
        assert_eq!(val, Complex::new(1.0, 0.0));
    }
}

#[test]
fn test_complex_f32_tensor() {
    let mut tensor = DenseTensor::<Complex<f32>>::zeros(vec![2, 2]);
    let c = Complex::new(3.0f32, 4.0f32);
    tensor.set(&[0, 0], c);
    assert_eq!(tensor.get(&[0, 0]), c);
}

#[test]
fn test_constant_with_complex() {
    let c = Complex::new(1.5, 2.5);
    let tensor = DenseTensor::constant(vec![3, 3], c);
    for &val in tensor.data() {
        assert_eq!(val, c);
    }
}

#[test]
fn test_ffi_pointer_types() {
    // Test that as_ptr works for different types
    let tensor_f64 = DenseTensor::<f64>::zeros(vec![10]);
    let _ptr_f64: *const f64 = tensor_f64.as_ptr();

    let tensor_f32 = DenseTensor::<f32>::zeros(vec![10]);
    let _ptr_f32: *const f32 = tensor_f32.as_ptr();

    let tensor_c64 = DenseTensor::<Complex<f64>>::zeros(vec![10]);
    let _ptr_c64: *const Complex<f64> = tensor_c64.as_ptr();

    let tensor_c32 = DenseTensor::<Complex<f32>>::zeros(vec![10]);
    let _ptr_c32: *const Complex<f32> = tensor_c32.as_ptr();
}

// Permute tests
#[test]
fn test_permute_2d_transpose() {
    let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    // Transpose: [2, 3] -> [3, 2]
    let result = tensor.permute(&[1, 0]);

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.get(&[0, 0]), 1.0);
    assert_eq!(result.get(&[1, 0]), 2.0);
    assert_eq!(result.get(&[2, 0]), 3.0);
    assert_eq!(result.get(&[0, 1]), 4.0);
    assert_eq!(result.get(&[1, 1]), 5.0);
    assert_eq!(result.get(&[2, 1]), 6.0);
}

#[test]
fn test_permute_3d() {
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let tensor = DenseTensor::<f64>::from_data(data, vec![2, 3, 4]);

    // Permute: [2, 3, 4] -> [4, 2, 3]
    let result = tensor.permute(&[2, 0, 1]);

    assert_eq!(result.shape(), &[4, 2, 3]);
    assert_eq!(result.len(), 24);

    // Verify a few elements
    assert_eq!(result.get(&[0, 0, 0]), tensor.get(&[0, 0, 0]));
    assert_eq!(result.get(&[1, 0, 0]), tensor.get(&[0, 0, 1]));
    assert_eq!(result.get(&[2, 0, 0]), tensor.get(&[0, 0, 2]));
}

#[test]
fn test_permute_identity() {
    let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // Identity permutation
    let result = tensor.permute(&[0, 1]);

    assert_eq!(result.shape(), tensor.shape());
    assert_eq!(result.data(), tensor.data());
}

#[test]
fn test_permute_f32() {
    let tensor = DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let result = tensor.permute(&[1, 0]);
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.get(&[0, 0]), 1.0f32);
    assert_eq!(result.get(&[1, 0]), 2.0f32);
}

#[test]
#[should_panic(expected = "Permutation length 3 doesn't match tensor rank 2")]
fn test_permute_invalid_length() {
    let tensor = DenseTensor::<f64>::zeros(vec![2, 3]);
    tensor.permute(&[0, 1, 2]); // Wrong length
}

#[test]
#[should_panic(expected = "Permutation index 2 out of range")]
fn test_permute_invalid_index() {
    let tensor = DenseTensor::<f64>::zeros(vec![2, 3]);
    tensor.permute(&[0, 2]); // Index 2 out of range
}

#[test]
#[should_panic(expected = "Duplicate index 1 in permutation")]
fn test_permute_duplicate_index() {
    let tensor = DenseTensor::<f64>::zeros(vec![2, 3]);
    tensor.permute(&[1, 1]); // Duplicate index
}

#[test]
fn test_permute_complex_basic() {
    let data: Vec<Complex<f64>> = (0..4)
        .map(|i| Complex::new(i as f64, (i + 1) as f64))
        .collect();
    let tensor = DenseTensor::from_data(data, vec![2, 2]);

    let result = tensor.permute(&[1, 0]);
    let result_naive = tensor.permute_naive(&[1, 0]);

    assert_eq!(result.shape(), result_naive.shape());
    assert_eq!(result.data(), result_naive.data());
}
