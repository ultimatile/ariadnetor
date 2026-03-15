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

#[test]
fn test_eye_f64_3x3() {
    let id = DenseTensor::<f64>::eye(3);
    assert_eq!(id.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(id.get(&[i, j]), expected);
        }
    }
}

#[test]
fn test_eye_c64() {
    let id = DenseTensor::<Complex<f64>>::eye(2);
    assert_eq!(id.shape(), &[2, 2]);
    assert_eq!(id.get(&[0, 0]), Complex::new(1.0, 0.0));
    assert_eq!(id.get(&[0, 1]), Complex::new(0.0, 0.0));
    assert_eq!(id.get(&[1, 0]), Complex::new(0.0, 0.0));
    assert_eq!(id.get(&[1, 1]), Complex::new(1.0, 0.0));
}

#[test]
fn test_eye_1x1() {
    let id = DenseTensor::<f64>::eye(1);
    assert_eq!(id.shape(), &[1, 1]);
    assert_eq!(id.get(&[0, 0]), 1.0);
}

#[test]
fn test_reshape_2x3_to_3x2() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r = t.reshape(vec![3, 2]);
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.data(), t.data());
}

#[test]
fn test_reshape_chain() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r1 = t.reshape(vec![6]);
    assert_eq!(r1.shape(), &[6]);
    let r2 = r1.reshape(vec![1, 6]);
    assert_eq!(r2.shape(), &[1, 6]);
    assert_eq!(r2.data(), t.data());
}

#[test]
fn test_reshape_preserves_cow() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let r = t.reshape(vec![4]);
    // Both share the same underlying Arc — modifying one triggers CoW
    assert_eq!(t.as_ptr(), r.as_ptr());
}

#[test]
#[should_panic(expected = "total elements must match")]
fn test_reshape_mismatch_panics() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let _r = t.reshape(vec![2, 2]);
}

#[test]
fn test_conj_real() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]);
    let c = t.conj();
    assert_eq!(c.data(), t.data());
}

#[test]
fn test_conj_complex() {
    let t = DenseTensor::from_data(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)],
        vec![2],
    );
    let c = t.conj();
    assert_eq!(c.get(&[0]), Complex::new(1.0, -2.0));
    assert_eq!(c.get(&[1]), Complex::new(3.0, 4.0));
}

#[test]
fn test_to_complex_from_real() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0], vec![2]);
    let c = t.to_complex();
    assert_eq!(c.shape(), &[2]);
    assert_eq!(c.get(&[0]), Complex::new(1.0, 0.0));
    assert_eq!(c.get(&[1]), Complex::new(2.0, 0.0));
}

#[test]
fn test_to_complex_from_complex() {
    let t = DenseTensor::from_data(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
        vec![2],
    );
    let c = t.to_complex();
    assert_eq!(c.data(), t.data());
}

#[test]
fn test_real_complex() {
    let t = DenseTensor::from_data(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)],
        vec![2],
    );
    let r = t.real();
    assert_eq!(r.shape(), &[2]);
    assert_eq!(r.get(&[0]), 1.0);
    assert_eq!(r.get(&[1]), 3.0);
}

#[test]
fn test_imag_complex() {
    let t = DenseTensor::from_data(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)],
        vec![2],
    );
    let im = t.imag();
    assert_eq!(im.shape(), &[2]);
    assert_eq!(im.get(&[0]), 2.0);
    assert_eq!(im.get(&[1]), -4.0);
}

#[test]
fn test_real_imag_real_type() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0], vec![2]);
    let r = t.real();
    assert_eq!(r.data(), t.data());
    let im = t.imag();
    assert_eq!(im.get(&[0]), 0.0);
    assert_eq!(im.get(&[1]), 0.0);
}

#[test]
fn test_map_double() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let r = t.map(|&x| x * 2.0);
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(r.data(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_map_type_conversion() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0], vec![2]);
    let c = t.map(|&x| Complex::new(x, 0.0));
    assert_eq!(c.shape(), &[2]);
    assert_eq!(c.get(&[0]), Complex::new(1.0, 0.0));
    assert_eq!(c.get(&[1]), Complex::new(2.0, 0.0));
}

#[test]
fn test_map_mut_negate() {
    let mut t = DenseTensor::<f64>::from_data(vec![1.0, -2.0, 3.0], vec![3]);
    t.map_mut(|&x| -x);
    assert_eq!(t.data(), &[-1.0, 2.0, -3.0]);
}

#[test]
fn test_map_mut_cow() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0], vec![2]);
    let mut t2 = t.clone(); // shares Arc
    assert_eq!(t.as_ptr(), t2.as_ptr()); // same underlying data
    t2.map_mut(|&x| x * 10.0);
    // t2 triggered CoW, t unchanged
    assert_eq!(t.data(), &[1.0, 2.0]);
    assert_eq!(t2.data(), &[10.0, 20.0]);
}

#[test]
fn test_map_with_index_sum_of_indices() {
    let t = DenseTensor::<f64>::from_data(vec![0.0; 6], vec![2, 3]);
    let r = t.map_with_index(|coords, _| (coords[0] + coords[1]) as f64);
    assert_eq!(r.shape(), &[2, 3]);
    // [0+0, 0+1, 0+2, 1+0, 1+1, 1+2] = [0, 1, 2, 1, 2, 3]
    assert_eq!(r.data(), &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
}

// --- slice / expand / replace_slice tests ---

#[test]
fn test_slice_2x2_from_3x3() {
    // [[1,2,3],[4,5,6],[7,8,9]] → slice rows 0..2, cols 1..3 → [[2,3],[5,6]]
    let t = DenseTensor::<f64>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    );
    let s = t.slice(&[(0, 2), (1, 3)]);
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.data(), &[2.0, 3.0, 5.0, 6.0]);
}

#[test]
fn test_slice_row() {
    let t = DenseTensor::<f64>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    // Extract second row
    let s = t.slice(&[(1, 2), (0, 3)]);
    assert_eq!(s.shape(), &[1, 3]);
    assert_eq!(s.data(), &[4.0, 5.0, 6.0]);
}

#[test]
fn test_slice_3d() {
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let t = DenseTensor::from_data(data, vec![2, 3, 4]);
    let s = t.slice(&[(0, 1), (1, 3), (2, 4)]);
    assert_eq!(s.shape(), &[1, 2, 2]);
    // Elements: t[0,1,2]=6, t[0,1,3]=7, t[0,2,2]=10, t[0,2,3]=11
    assert_eq!(s.data(), &[6.0, 7.0, 10.0, 11.0]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_slice_out_of_bounds() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let _s = t.slice(&[(0, 3), (0, 2)]);
}

#[test]
fn test_expand_symmetric() {
    // [[1,2],[3,4]] with padding (1,1) on each axis → 4×4
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let e = t.expand(&[(1, 1), (1, 1)]);
    assert_eq!(e.shape(), &[4, 4]);
    // Row 0: all zeros
    assert_eq!(e.get(&[0, 0]), 0.0);
    assert_eq!(e.get(&[0, 1]), 0.0);
    // Row 1: 0, 1, 2, 0
    assert_eq!(e.get(&[1, 0]), 0.0);
    assert_eq!(e.get(&[1, 1]), 1.0);
    assert_eq!(e.get(&[1, 2]), 2.0);
    assert_eq!(e.get(&[1, 3]), 0.0);
    // Row 2: 0, 3, 4, 0
    assert_eq!(e.get(&[2, 1]), 3.0);
    assert_eq!(e.get(&[2, 2]), 4.0);
    // Row 3: all zeros
    assert_eq!(e.get(&[3, 3]), 0.0);
}

#[test]
fn test_expand_asymmetric() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0], vec![2]);
    let e = t.expand(&[(2, 1)]);
    assert_eq!(e.shape(), &[5]);
    assert_eq!(e.data(), &[0.0, 0.0, 1.0, 2.0, 0.0]);
}

#[test]
fn test_replace_slice_center() {
    // 3×3 zeros, write [[5,6],[7,8]] at position (0,1)
    let mut t = DenseTensor::<f64>::from_data(vec![0.0; 9], vec![3, 3]);
    let sub = DenseTensor::<f64>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    t.replace_slice(&sub, &[0, 1]);
    assert_eq!(t.get(&[0, 1]), 5.0);
    assert_eq!(t.get(&[0, 2]), 6.0);
    assert_eq!(t.get(&[1, 1]), 7.0);
    assert_eq!(t.get(&[1, 2]), 8.0);
    // Untouched elements remain zero
    assert_eq!(t.get(&[0, 0]), 0.0);
    assert_eq!(t.get(&[2, 2]), 0.0);
}

#[test]
#[should_panic(expected = "exceeds boundary")]
fn test_replace_slice_out_of_bounds() {
    let mut t = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
    let sub = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    t.replace_slice(&sub, &[1, 1]); // 1+2 > 2
}

#[test]
fn test_slice_expand_round_trip() {
    let t = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let expanded = t.expand(&[(1, 1), (1, 1)]); // 4×4
    let recovered = expanded.slice(&[(1, 3), (1, 3)]); // back to 2×2
    assert_eq!(recovered.data(), t.data());
}
