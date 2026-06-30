use ariadnetor_tensor::{DenseTensorData, MemoryOrder};
use num_complex::Complex;

#[test]
fn test_tensor_creation() {
    let tensor = DenseTensorData::<f64>::zeros_in_order(vec![3, 4], MemoryOrder::ColumnMajor);
    assert_eq!(tensor.shape(), &[3, 4]);
    assert_eq!(tensor.len(), 12);
}

#[test]
fn test_tensor_from_data() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor =
        DenseTensorData::<f64>::from_raw_parts(data.clone(), vec![2, 2], MemoryOrder::ColumnMajor);
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.data(), &data[..]);
}

#[test]
fn test_data_mut() {
    let mut tensor = DenseTensorData::<f64>::zeros_in_order(vec![3, 4], MemoryOrder::ColumnMajor);
    tensor.data_mut()[5] = 42.0;
    assert_eq!(tensor.data()[5], 42.0);
    assert_eq!(tensor.data()[0], 0.0);
}

#[test]
fn test_tensor_fill() {
    let mut tensor = DenseTensorData::<f64>::zeros_in_order(vec![2, 3], MemoryOrder::ColumnMajor);
    tensor.fill(3.15);
    for &val in tensor.data() {
        assert_eq!(val, 3.15);
    }
}

#[test]
fn test_ones() {
    let tensor = DenseTensorData::<f64>::ones_in_order(vec![2, 3], MemoryOrder::ColumnMajor);
    for &val in tensor.data() {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_copy_on_write() {
    let tensor1 = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let mut tensor2 = tensor1.clone();

    // Modification triggers CoW
    tensor2.data_mut()[0] = 999.0;

    // tensor1 should be unchanged
    assert_eq!(tensor1.data()[0], 1.0);
    assert_eq!(tensor2.data()[0], 999.0);
}

#[test]
fn test_f32_tensor() {
    let tensor = DenseTensorData::<f32>::zeros_in_order(vec![2, 3], MemoryOrder::ColumnMajor);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.len(), 6);

    let tensor = DenseTensorData::<f32>::ones_in_order(vec![2, 2], MemoryOrder::ColumnMajor);
    for &val in tensor.data() {
        assert_eq!(val, 1.0f32);
    }
}

#[test]
fn test_complex_f64_tensor() {
    let tensor =
        DenseTensorData::<Complex<f64>>::zeros_in_order(vec![2, 2], MemoryOrder::ColumnMajor);
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.len(), 4);
    for &val in tensor.data() {
        assert_eq!(val, Complex::new(0.0, 0.0));
    }

    let tensor =
        DenseTensorData::<Complex<f64>>::ones_in_order(vec![2, 2], MemoryOrder::ColumnMajor);
    for &val in tensor.data() {
        assert_eq!(val, Complex::new(1.0, 0.0));
    }
}

#[test]
fn test_complex_f32_tensor() {
    let mut tensor =
        DenseTensorData::<Complex<f32>>::zeros_in_order(vec![2, 2], MemoryOrder::ColumnMajor);
    let c = Complex::new(3.0f32, 4.0f32);
    tensor.data_mut()[0] = c;
    assert_eq!(tensor.data()[0], c);
}

#[test]
fn test_filled_with_complex() {
    let c = Complex::new(1.5, 2.5);
    let tensor = DenseTensorData::filled_in_order(vec![3, 3], c, MemoryOrder::ColumnMajor);
    for &val in tensor.data() {
        assert_eq!(val, c);
    }
}

#[test]
fn test_ffi_pointer_types() {
    // FFI clients consume the raw `*const T` from the data slice;
    // pin that the pointer types match the element type for every
    // primitive scalar combination the public surface supports.
    let tensor_f64 = DenseTensorData::<f64>::zeros_in_order(vec![10], MemoryOrder::ColumnMajor);
    let _ptr_f64: *const f64 = tensor_f64.data().as_ptr();

    let tensor_f32 = DenseTensorData::<f32>::zeros_in_order(vec![10], MemoryOrder::ColumnMajor);
    let _ptr_f32: *const f32 = tensor_f32.data().as_ptr();

    let tensor_c64 =
        DenseTensorData::<Complex<f64>>::zeros_in_order(vec![10], MemoryOrder::ColumnMajor);
    let _ptr_c64: *const Complex<f64> = tensor_c64.data().as_ptr();

    let tensor_c32 =
        DenseTensorData::<Complex<f32>>::zeros_in_order(vec![10], MemoryOrder::ColumnMajor);
    let _ptr_c32: *const Complex<f32> = tensor_c32.data().as_ptr();
}

#[test]
fn test_eye_f64_3x3() {
    let id = DenseTensorData::<f64>::eye_in_order(3, MemoryOrder::RowMajor);
    assert_eq!(id.shape(), &[3, 3]);
    // eye() stores data in RM: data[i*n + i] = 1.0
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(id.data()[i * 3 + j], expected);
        }
    }
}

#[test]
fn test_eye_c64() {
    let id = DenseTensorData::<Complex<f64>>::eye_in_order(2, MemoryOrder::RowMajor);
    assert_eq!(id.shape(), &[2, 2]);
    assert_eq!(id.data()[0], Complex::new(1.0, 0.0)); // (0,0)
    assert_eq!(id.data()[1], Complex::new(0.0, 0.0)); // (0,1)
    assert_eq!(id.data()[2], Complex::new(0.0, 0.0)); // (1,0)
    assert_eq!(id.data()[3], Complex::new(1.0, 0.0)); // (1,1)
}

#[test]
fn test_eye_1x1() {
    let id = DenseTensorData::<f64>::eye_in_order(1, MemoryOrder::RowMajor);
    assert_eq!(id.shape(), &[1, 1]);
    assert_eq!(id.data()[0], 1.0);
}

#[test]
fn test_reshape_2x3_to_3x2() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let r = t.reshape(vec![3, 2]);
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.data(), t.data());
}

#[test]
fn test_reshape_chain() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let r1 = t.reshape(vec![6]);
    assert_eq!(r1.shape(), &[6]);
    let r2 = r1.reshape(vec![1, 6]);
    assert_eq!(r2.shape(), &[1, 6]);
    assert_eq!(r2.data(), t.data());
}

#[test]
fn test_reshape_preserves_cow() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let r = t.reshape(vec![4]);
    assert_eq!(t.data().as_ptr(), r.data().as_ptr());
}

#[test]
#[should_panic(expected = "total elements must match")]
fn test_reshape_mismatch_panics() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let _r = t.reshape(vec![2, 2]);
}

#[test]
fn test_conj_real() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, -2.0, 3.0, -4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let c = t.conj();
    assert_eq!(c.data(), t.data());
}

#[test]
fn test_conj_complex() {
    let t = DenseTensorData::from_raw_parts(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)],
        vec![2],
        MemoryOrder::ColumnMajor,
    );
    let c = t.conj();
    assert_eq!(c.data()[0], Complex::new(1.0, -2.0));
    assert_eq!(c.data()[1], Complex::new(3.0, 4.0));
}

#[test]
fn test_to_complex_from_real() {
    let t =
        DenseTensorData::<f64>::from_raw_parts(vec![1.0, 2.0], vec![2], MemoryOrder::ColumnMajor);
    let c = t.to_complex();
    assert_eq!(c.shape(), &[2]);
    assert_eq!(c.data()[0], Complex::new(1.0, 0.0));
    assert_eq!(c.data()[1], Complex::new(2.0, 0.0));
}

#[test]
fn test_to_complex_from_complex() {
    let t = DenseTensorData::from_raw_parts(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
        vec![2],
        MemoryOrder::ColumnMajor,
    );
    let c = t.to_complex();
    assert_eq!(c.data(), t.data());
}

#[test]
fn test_real_complex() {
    let t = DenseTensorData::from_raw_parts(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)],
        vec![2],
        MemoryOrder::ColumnMajor,
    );
    let r = t.real();
    assert_eq!(r.shape(), &[2]);
    assert_eq!(r.data()[0], 1.0);
    assert_eq!(r.data()[1], 3.0);
}

#[test]
fn test_imag_complex() {
    let t = DenseTensorData::from_raw_parts(
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)],
        vec![2],
        MemoryOrder::ColumnMajor,
    );
    let im = t.imag();
    assert_eq!(im.shape(), &[2]);
    assert_eq!(im.data()[0], 2.0);
    assert_eq!(im.data()[1], -4.0);
}

#[test]
fn test_real_imag_real_type() {
    let t =
        DenseTensorData::<f64>::from_raw_parts(vec![1.0, 2.0], vec![2], MemoryOrder::ColumnMajor);
    let r = t.real();
    assert_eq!(r.data(), t.data());
    let im = t.imag();
    assert_eq!(im.data()[0], 0.0);
    assert_eq!(im.data()[1], 0.0);
}

#[test]
fn test_map_double() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let r = t.map(|&x| x * 2.0);
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(r.data(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_map_type_conversion() {
    let t =
        DenseTensorData::<f64>::from_raw_parts(vec![1.0, 2.0], vec![2], MemoryOrder::ColumnMajor);
    let c = t.map(|&x| Complex::new(x, 0.0));
    assert_eq!(c.shape(), &[2]);
    assert_eq!(c.data()[0], Complex::new(1.0, 0.0));
    assert_eq!(c.data()[1], Complex::new(2.0, 0.0));
}

#[test]
fn test_map_mut_negate() {
    let mut t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, -2.0, 3.0],
        vec![3],
        MemoryOrder::ColumnMajor,
    );
    t.map_mut(|&x| -x);
    assert_eq!(t.data(), &[-1.0, 2.0, -3.0]);
}

#[test]
fn test_map_mut_cow() {
    let t =
        DenseTensorData::<f64>::from_raw_parts(vec![1.0, 2.0], vec![2], MemoryOrder::ColumnMajor);
    let mut t2 = t.clone();
    assert_eq!(t.data().as_ptr(), t2.data().as_ptr());
    t2.map_mut(|&x| x * 10.0);
    assert_eq!(t.data(), &[1.0, 2.0]);
    assert_eq!(t2.data(), &[10.0, 20.0]);
}

#[test]
fn test_map_with_index_sum_of_indices() {
    // `map_with_index` walks coords in the iteration `order` while
    // reading the storage linearly, so the storage must already be in
    // that same order.
    let t = DenseTensorData::<f64>::from_raw_parts(vec![0.0; 6], vec![2, 3], MemoryOrder::RowMajor);
    let r = t.map_with_index(|coords, _| (coords[0] + coords[1]) as f64);
    assert_eq!(r.shape(), &[2, 3]);
    // RM: [0+0, 0+1, 0+2, 1+0, 1+1, 1+2] = [0, 1, 2, 1, 2, 3]
    assert_eq!(r.data(), &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
}

// --- slice / expand / replace_slice tests ---

#[test]
fn test_slice_2x2_from_3x3() {
    // RM [[1,2,3],[4,5,6],[7,8,9]] → slice rows 0..2, cols 1..3 → [[2,3],[5,6]]
    let t = DenseTensorData::<f64>::from_raw_parts(
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
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let s = t.slice(&[(1, 2), (0, 3)]);
    assert_eq!(s.shape(), &[1, 3]);
    assert_eq!(s.data(), &[4.0, 5.0, 6.0]);
}

#[test]
fn test_slice_3d() {
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let t = DenseTensorData::from_raw_parts(data, vec![2, 3, 4], MemoryOrder::RowMajor);
    let s = t.slice(&[(0, 1), (1, 3), (2, 4)]);
    assert_eq!(s.shape(), &[1, 2, 2]);
    // RM elements: t[0,1,2]=6, t[0,1,3]=7, t[0,2,2]=10, t[0,2,3]=11
    assert_eq!(s.data(), &[6.0, 7.0, 10.0, 11.0]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_slice_out_of_bounds() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let _s = t.slice(&[(0, 3), (0, 2)]);
}

#[test]
fn test_expand_symmetric() {
    // RM [[1,2],[3,4]] with padding (1,1) on each axis → 4x4
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let e = t.expand(&[(1, 1), (1, 1)]);
    assert_eq!(e.shape(), &[4, 4]);
    // Expected RM flat: row0=[0,0,0,0], row1=[0,1,2,0], row2=[0,3,4,0], row3=[0,0,0,0]
    assert_eq!(
        e.data(),
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
    );
}

#[test]
fn test_expand_asymmetric() {
    let t = DenseTensorData::<f64>::from_raw_parts(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let e = t.expand(&[(2, 1)]);
    assert_eq!(e.shape(), &[5]);
    assert_eq!(e.data(), &[0.0, 0.0, 1.0, 2.0, 0.0]);
}

#[test]
fn test_replace_slice_center() {
    // 3x3 zeros, write [[5,6],[7,8]] at position (0,1) in RM
    let mut t =
        DenseTensorData::<f64>::from_raw_parts(vec![0.0; 9], vec![3, 3], MemoryOrder::RowMajor);
    let sub = DenseTensorData::<f64>::from_raw_parts(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    t.replace_slice(&sub, &[0, 1]);
    // RM: row0=[0,5,6], row1=[0,7,8], row2=[0,0,0]
    assert_eq!(t.data(), &[0.0, 5.0, 6.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0]);
}

#[test]
#[should_panic(expected = "exceeds boundary")]
fn test_replace_slice_out_of_bounds() {
    let mut t =
        DenseTensorData::<f64>::from_raw_parts(vec![0.0; 4], vec![2, 2], MemoryOrder::RowMajor);
    let sub = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    t.replace_slice(&sub, &[1, 1]);
}

#[test]
fn test_slice_expand_round_trip() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let expanded = t.expand(&[(1, 1), (1, 1)]);
    let recovered = expanded.slice(&[(1, 3), (1, 3)]);
    assert_eq!(recovered.data(), t.data());
}

// --- concatenate / stack tests ---

#[test]
fn test_concatenate_axis0() {
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let b = DenseTensorData::<f64>::from_raw_parts(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let c = DenseTensorData::concatenate(&[&a, &b], 0);
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
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let b = DenseTensorData::<f64>::from_raw_parts(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let c = DenseTensorData::concatenate(&[&a, &b], 1);
    assert_eq!(c.shape(), &[2, 4]);
    assert_eq!(c.data(), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn test_concatenate_single() {
    let a =
        DenseTensorData::<f64>::from_raw_parts(vec![1.0, 2.0, 3.0], vec![3], MemoryOrder::RowMajor);
    let c = DenseTensorData::concatenate(&[&a], 0);
    assert_eq!(c.data(), a.data());
}

#[test]
#[should_panic(expected = "concatenate")]
fn test_concatenate_shape_mismatch() {
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let b = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let _c = DenseTensorData::concatenate(&[&a, &b], 0);
}

#[test]
fn test_stack_axis0() {
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let b = DenseTensorData::<f64>::from_raw_parts(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let s = DenseTensorData::stack(&[&a, &b], 0);
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
    let a = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let b = DenseTensorData::<f64>::from_raw_parts(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let s = DenseTensorData::stack(&[&a, &b], 2);
    assert_eq!(s.shape(), &[2, 3, 2]);
    assert_eq!(
        s.data(),
        &[
            1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0
        ]
    );
}

#[test]
fn test_stack_single() {
    let a = DenseTensorData::<f64>::from_raw_parts(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let s = DenseTensorData::stack(&[&a], 0);
    assert_eq!(s.shape(), &[1, 2]);
    assert_eq!(s.data(), &[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "stack")]
fn test_stack_shape_mismatch() {
    let a = DenseTensorData::<f64>::from_raw_parts(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let b =
        DenseTensorData::<f64>::from_raw_parts(vec![1.0, 2.0, 3.0], vec![3], MemoryOrder::RowMajor);
    let _s = DenseTensorData::stack(&[&a, &b], 0);
}

// --- iter tests ---

#[test]
fn test_iter_walks_storage_order() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let elems: Vec<f64> = t.iter().copied().collect();
    assert_eq!(elems, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_iter_exact_size() {
    let t = DenseTensorData::<f64>::zeros_in_order(vec![3, 4], MemoryOrder::ColumnMajor);
    assert_eq!(t.iter().len(), 12);
}

#[test]
fn test_iter_empty() {
    let t = DenseTensorData::<f64>::zeros_in_order(vec![0], MemoryOrder::ColumnMajor);
    assert_eq!(t.iter().count(), 0);
}

mod random_tests {
    use ariadnetor_tensor::{DenseTensorData, MemoryOrder};
    use rand::SeedableRng;

    #[test]
    fn test_random_f64_shape() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let t =
            DenseTensorData::<f64>::random_in_order(vec![3, 4], MemoryOrder::ColumnMajor, &mut rng);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.len(), 12);
    }

    #[test]
    fn test_random_f64_reproducible() {
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(123);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);
        let t1 = DenseTensorData::<f64>::random_in_order(
            vec![2, 3],
            MemoryOrder::ColumnMajor,
            &mut rng1,
        );
        let t2 = DenseTensorData::<f64>::random_in_order(
            vec![2, 3],
            MemoryOrder::ColumnMajor,
            &mut rng2,
        );
        assert_eq!(t1.data(), t2.data());
    }

    #[test]
    fn test_random_different_seeds() {
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(1);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(2);
        let t1 =
            DenseTensorData::<f64>::random_in_order(vec![4], MemoryOrder::ColumnMajor, &mut rng1);
        let t2 =
            DenseTensorData::<f64>::random_in_order(vec![4], MemoryOrder::ColumnMajor, &mut rng2);
        assert_ne!(t1.data(), t2.data());
    }

    #[test]
    fn test_random_f32() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let t =
            DenseTensorData::<f32>::random_in_order(vec![5], MemoryOrder::ColumnMajor, &mut rng);
        assert_eq!(t.shape(), &[5]);
    }
}
