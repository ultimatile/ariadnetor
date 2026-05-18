use arnet_linalg::contract;
use arnet_native::NativeBackend;
use arnet_tensor::{DenseTensorData, MemoryOrder};

/// Create Dense from row-major data, converted to column-major for NativeBackend.
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> DenseTensorData<T> {
    let rm = DenseTensorData::from_raw_parts(data, shape, MemoryOrder::RowMajor);
    arnet_tensor::reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)
}

/// Convert column-major Dense back to row-major so `.get()` returns correct values.
fn to_rm<T: Clone>(tensor: &DenseTensorData<T>) -> DenseTensorData<T> {
    arnet_tensor::reorder(tensor, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor)
}

#[test]
fn test_contract_matmul() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = to_rm(&contract(&backend, &a, &b, "ik,kj->ij").unwrap());

    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[1, 0]), 43.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}

#[test]
fn test_contract_tensor_contraction() {
    let backend = NativeBackend::new();
    let a = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );

    let c = to_rm(&contract(&backend, &a, &b, "ijk,jkl->il").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_ne!(c.get(&[0, 0]), 0.0);
}

#[test]
fn test_contract_f32() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = to_rm(&contract(&backend, &a, &b, "ik,kj->ij").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0f32);
}

#[test]
fn test_contract_with_permutation() {
    let backend = NativeBackend::new();
    let a = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let c = to_rm(&contract(&backend, &a, &b, "ikj,kj->i").unwrap());

    assert_eq!(c.shape(), &[2]);
    assert_ne!(c.data()[0], 0.0);
}

#[test]
fn test_contract_rectangular() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);

    let c = contract(&backend, &a, &b, "ik,kj->ij").unwrap();

    assert_eq!(c.shape(), &[2, 3]);
}

#[test]
fn test_contract_invalid_notation() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = contract(&backend, &a, &b, "ik,kj->im");
    assert!(result.is_err());
}

#[test]
fn test_contract_rank_mismatch() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = contract(&backend, &a, &b, "ijk,kl->ijl");
    assert!(result.is_err());
}

// ============================================================================
// Output index reordering tests
// ============================================================================

#[test]
fn test_contract_output_reorder_swap() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = cm((1..=12).map(|x| x as f64).collect(), vec![3, 4]);

    let normal = to_rm(&contract(&backend, &a, &b, "ik,kj->ij").unwrap());
    let swapped = to_rm(&contract(&backend, &a, &b, "ik,kj->ji").unwrap());

    assert_eq!(normal.shape(), &[2, 4]);
    assert_eq!(swapped.shape(), &[4, 2]);

    for i in 0..2 {
        for j in 0..4 {
            assert_eq!(swapped.get(&[j, i]), normal.get(&[i, j]));
        }
    }
}

#[test]
fn test_contract_output_reorder_3d() {
    let backend = NativeBackend::new();
    let a = cm((1..=12).map(|x| x as f64).collect(), vec![2, 3, 2]);
    let b = cm((1..=8).map(|x| x as f64).collect(), vec![2, 4]);

    let normal = to_rm(&contract(&backend, &a, &b, "abc,cd->abd").unwrap());
    let reordered = to_rm(&contract(&backend, &a, &b, "abc,cd->dba").unwrap());

    assert_eq!(normal.shape(), &[2, 3, 4]);
    assert_eq!(reordered.shape(), &[4, 3, 2]);

    for a_idx in 0..2 {
        for b_idx in 0..3 {
            for d in 0..4 {
                assert_eq!(
                    reordered.get(&[d, b_idx, a_idx]),
                    normal.get(&[a_idx, b_idx, d]),
                );
            }
        }
    }
}

#[test]
fn test_contract_rejects_batch_indices() {
    let backend = NativeBackend::new();
    let a = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = cm(
        vec![1.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        vec![2, 2, 2],
    );

    let result = contract(&backend, &a, &b, "bik,bkj->bij");
    assert!(result.is_err());
}

#[test]
fn test_contract_output_memory_order() {
    let backend = NativeBackend::new();
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let _c = contract(&backend, &a, &b, "ik,kj->ij").unwrap();
    let _c_reordered = contract(&backend, &a, &b, "ik,kj->ji").unwrap();
}
