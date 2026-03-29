use arnet_tensor::{DenseTensor, MemoryOrder};
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
    let tensor =
        DenseTensor::<f64>::from_data_with_order(data.clone(), vec![2, 2], MemoryOrder::RowMajor);
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
    let tensor1 = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
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
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let r = t.reshape(vec![3, 2]);
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.data(), t.data());
}

#[test]
fn test_reshape_chain() {
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let r1 = t.reshape(vec![6]);
    assert_eq!(r1.shape(), &[6]);
    let r2 = r1.reshape(vec![1, 6]);
    assert_eq!(r2.shape(), &[1, 6]);
    assert_eq!(r2.data(), t.data());
}

#[test]
fn test_reshape_preserves_cow() {
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let r = t.reshape(vec![4]);
    // Both share the same underlying Arc — modifying one triggers CoW
    assert_eq!(t.as_ptr(), r.as_ptr());
}

#[test]
#[should_panic(expected = "total elements must match")]
fn test_reshape_mismatch_panics() {
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let _r = t.reshape(vec![2, 2]);
}

#[test]
fn test_conj_real() {
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, -2.0, 3.0, -4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let c = t.conj();
    assert_eq!(c.data(), t.data());
}

#[test]
fn test_conj_complex() {
    let t = DenseTensor::from_data_with_order(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)],
        vec![2],
        MemoryOrder::RowMajor,
    );
    let c = t.conj();
    assert_eq!(c.get(&[0]), Complex::new(1.0, -2.0));
    assert_eq!(c.get(&[1]), Complex::new(3.0, 4.0));
}

#[test]
fn test_to_complex_from_real() {
    let t =
        DenseTensor::<f64>::from_data_with_order(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let c = t.to_complex();
    assert_eq!(c.shape(), &[2]);
    assert_eq!(c.get(&[0]), Complex::new(1.0, 0.0));
    assert_eq!(c.get(&[1]), Complex::new(2.0, 0.0));
}

#[test]
fn test_to_complex_from_complex() {
    let t = DenseTensor::from_data_with_order(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
        vec![2],
        MemoryOrder::RowMajor,
    );
    let c = t.to_complex();
    assert_eq!(c.data(), t.data());
}

#[test]
fn test_real_complex() {
    let t = DenseTensor::from_data_with_order(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)],
        vec![2],
        MemoryOrder::RowMajor,
    );
    let r = t.real();
    assert_eq!(r.shape(), &[2]);
    assert_eq!(r.get(&[0]), 1.0);
    assert_eq!(r.get(&[1]), 3.0);
}

#[test]
fn test_imag_complex() {
    let t = DenseTensor::from_data_with_order(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)],
        vec![2],
        MemoryOrder::RowMajor,
    );
    let im = t.imag();
    assert_eq!(im.shape(), &[2]);
    assert_eq!(im.get(&[0]), 2.0);
    assert_eq!(im.get(&[1]), -4.0);
}

#[test]
fn test_real_imag_real_type() {
    let t =
        DenseTensor::<f64>::from_data_with_order(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let r = t.real();
    assert_eq!(r.data(), t.data());
    let im = t.imag();
    assert_eq!(im.get(&[0]), 0.0);
    assert_eq!(im.get(&[1]), 0.0);
}

#[test]
fn test_map_double() {
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let r = t.map(|&x| x * 2.0);
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(r.data(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_map_type_conversion() {
    let t =
        DenseTensor::<f64>::from_data_with_order(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let c = t.map(|&x| Complex::new(x, 0.0));
    assert_eq!(c.shape(), &[2]);
    assert_eq!(c.get(&[0]), Complex::new(1.0, 0.0));
    assert_eq!(c.get(&[1]), Complex::new(2.0, 0.0));
}

#[test]
fn test_map_mut_negate() {
    let mut t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, -2.0, 3.0],
        vec![3],
        MemoryOrder::RowMajor,
    );
    t.map_mut(|&x| -x);
    assert_eq!(t.data(), &[-1.0, 2.0, -3.0]);
}

#[test]
fn test_map_mut_cow() {
    let t =
        DenseTensor::<f64>::from_data_with_order(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let mut t2 = t.clone(); // shares Arc
    assert_eq!(t.as_ptr(), t2.as_ptr()); // same underlying data
    t2.map_mut(|&x| x * 10.0);
    // t2 triggered CoW, t unchanged
    assert_eq!(t.data(), &[1.0, 2.0]);
    assert_eq!(t2.data(), &[10.0, 20.0]);
}

#[test]
fn test_map_with_index_sum_of_indices() {
    let t =
        DenseTensor::<f64>::from_data_with_order(vec![0.0; 6], vec![2, 3], MemoryOrder::RowMajor);
    let r = t.map_with_index(|coords, _| (coords[0] + coords[1]) as f64);
    assert_eq!(r.shape(), &[2, 3]);
    // [0+0, 0+1, 0+2, 1+0, 1+1, 1+2] = [0, 1, 2, 1, 2, 3]
    assert_eq!(r.data(), &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
}

// --- slice / expand / replace_slice tests ---

#[test]
fn test_slice_2x2_from_3x3() {
    // [[1,2,3],[4,5,6],[7,8,9]] → slice rows 0..2, cols 1..3 → [[2,3],[5,6]]
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
        MemoryOrder::RowMajor,
    );
    let s = t.slice(&[(0, 2), (1, 3)]);
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.data(), &[2.0, 3.0, 5.0, 6.0]);
}

#[test]
fn test_slice_row() {
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    // Extract second row
    let s = t.slice(&[(1, 2), (0, 3)]);
    assert_eq!(s.shape(), &[1, 3]);
    assert_eq!(s.data(), &[4.0, 5.0, 6.0]);
}

#[test]
fn test_slice_3d() {
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let t = DenseTensor::from_data_with_order(data, vec![2, 3, 4], MemoryOrder::RowMajor);
    let s = t.slice(&[(0, 1), (1, 3), (2, 4)]);
    assert_eq!(s.shape(), &[1, 2, 2]);
    // Elements: t[0,1,2]=6, t[0,1,3]=7, t[0,2,2]=10, t[0,2,3]=11
    assert_eq!(s.data(), &[6.0, 7.0, 10.0, 11.0]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_slice_out_of_bounds() {
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let _s = t.slice(&[(0, 3), (0, 2)]);
}

#[test]
fn test_expand_symmetric() {
    // [[1,2],[3,4]] with padding (1,1) on each axis → 4×4
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
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
    let t =
        DenseTensor::<f64>::from_data_with_order(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let e = t.expand(&[(2, 1)]);
    assert_eq!(e.shape(), &[5]);
    assert_eq!(e.data(), &[0.0, 0.0, 1.0, 2.0, 0.0]);
}

#[test]
fn test_replace_slice_center() {
    // 3×3 zeros, write [[5,6],[7,8]] at position (0,1)
    let mut t =
        DenseTensor::<f64>::from_data_with_order(vec![0.0; 9], vec![3, 3], MemoryOrder::RowMajor);
    let sub = DenseTensor::<f64>::from_data_with_order(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
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
    let mut t =
        DenseTensor::<f64>::from_data_with_order(vec![0.0; 4], vec![2, 2], MemoryOrder::RowMajor);
    let sub = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    t.replace_slice(&sub, &[1, 1]); // 1+2 > 2
}

#[test]
fn test_slice_expand_round_trip() {
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let expanded = t.expand(&[(1, 1), (1, 1)]); // 4×4
    let recovered = expanded.slice(&[(1, 3), (1, 3)]); // back to 2×2
    assert_eq!(recovered.data(), t.data());
}

// --- concatenate / stack tests ---

#[test]
fn test_concatenate_axis0() {
    let a = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let b = DenseTensor::<f64>::from_data_with_order(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let c = DenseTensor::concatenate(&[&a, &b], 0);
    assert_eq!(c.shape(), &[4, 3]);
    assert_eq!(
        c.data(),
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        ]
    );
}

#[test]
fn test_concatenate_axis1() {
    let a = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let b = DenseTensor::<f64>::from_data_with_order(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let c = DenseTensor::concatenate(&[&a, &b], 1);
    assert_eq!(c.shape(), &[2, 4]);
    assert_eq!(c.data(), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn test_concatenate_single() {
    let a = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0],
        vec![3],
        MemoryOrder::RowMajor,
    );
    let c = DenseTensor::concatenate(&[&a], 0);
    assert_eq!(c.data(), a.data());
}

#[test]
#[should_panic(expected = "concatenate")]
fn test_concatenate_shape_mismatch() {
    let a = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let b = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let _c = DenseTensor::concatenate(&[&a, &b], 0);
}

#[test]
fn test_stack_axis0() {
    let a = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let b = DenseTensor::<f64>::from_data_with_order(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let s = DenseTensor::stack(&[&a, &b], 0);
    assert_eq!(s.shape(), &[2, 2, 3]);
    assert_eq!(
        s.data(),
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        ]
    );
}

#[test]
fn test_stack_axis2() {
    let a = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let b = DenseTensor::<f64>::from_data_with_order(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let s = DenseTensor::stack(&[&a, &b], 2);
    assert_eq!(s.shape(), &[2, 3, 2]);
    // [0,0,:] = [1,7], [0,1,:] = [2,8], [0,2,:] = [3,9]
    // [1,0,:] = [4,10], [1,1,:] = [5,11], [1,2,:] = [6,12]
    assert_eq!(
        s.data(),
        &[
            1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0
        ]
    );
}

#[test]
fn test_stack_single() {
    let a =
        DenseTensor::<f64>::from_data_with_order(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let s = DenseTensor::stack(&[&a], 0);
    assert_eq!(s.shape(), &[1, 2]);
    assert_eq!(s.data(), &[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "stack")]
fn test_stack_shape_mismatch() {
    let a =
        DenseTensor::<f64>::from_data_with_order(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let b = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0],
        vec![3],
        MemoryOrder::RowMajor,
    );
    let _s = DenseTensor::stack(&[&a, &b], 0);
}

// --- iter tests ---

#[test]
fn test_iter_contiguous_row_major() {
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let elems: Vec<f64> = t.iter().copied().collect();
    assert_eq!(elems, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_iter_contiguous_column_major() {
    let t = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    // iter() walks storage order, not logical order
    let elems: Vec<f64> = t.iter().copied().collect();
    assert_eq!(elems, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_iter_element_sum_order_independent() {
    // Same logical tensor in RowMajor and ColumnMajor — sum must match
    let rm = DenseTensor::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let cm = rm.to_contiguous(MemoryOrder::ColumnMajor);

    let sum_rm: f64 = rm.iter().sum();
    let sum_cm: f64 = cm.iter().sum();
    assert_eq!(sum_rm, sum_cm);
}

#[test]
fn test_iter_exact_size() {
    let t = DenseTensor::<f64>::zeros(vec![3, 4]);
    assert_eq!(t.iter().len(), 12);
}

#[test]
fn test_iter_empty() {
    let t = DenseTensor::<f64>::zeros(vec![0]);
    assert_eq!(t.iter().count(), 0);
}

#[test]
fn test_iter_scalar() {
    let t = DenseTensor::<f64>::from_data_with_order(vec![42.0], vec![1], MemoryOrder::RowMajor);
    let elems: Vec<f64> = t.iter().copied().collect();
    assert_eq!(elems, vec![42.0]);
}

// --- random tests (require "random" feature) ---

#[cfg(feature = "random")]
mod random_tests {
    use arnet_tensor::DenseTensor;
    use rand::SeedableRng;

    #[test]
    fn test_random_f64_shape() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let t = DenseTensor::<f64>::random(vec![3, 4], &mut rng);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.len(), 12);
    }

    #[test]
    fn test_random_f64_reproducible() {
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(123);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);
        let t1 = DenseTensor::<f64>::random(vec![2, 3], &mut rng1);
        let t2 = DenseTensor::<f64>::random(vec![2, 3], &mut rng2);
        assert_eq!(t1.data(), t2.data());
    }

    #[test]
    fn test_random_different_seeds() {
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(1);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(2);
        let t1 = DenseTensor::<f64>::random(vec![4], &mut rng1);
        let t2 = DenseTensor::<f64>::random(vec![4], &mut rng2);
        assert_ne!(t1.data(), t2.data());
    }

    #[test]
    fn test_random_f32() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let t = DenseTensor::<f32>::random(vec![5], &mut rng);
        assert_eq!(t.shape(), &[5]);
    }
}
