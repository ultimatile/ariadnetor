use arnet_linalg::contract;
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, DenseTensor, MemoryOrder};

/// Build a `DenseTensor` from row-major data, reordered to column-major
/// (the NativeBackend's preferred order). Mirrors the original
/// `cm()` helper that produced a `Dense`.
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> DenseTensor<T, NativeBackend> {
    let rm = Dense::new(data, shape, MemoryOrder::RowMajor);
    let cm = arnet_tensor::reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);
    DenseTensor::with_backend(cm.into_tensor_data(), NativeBackend::shared())
}

/// Reorder a `DenseTensor` result back to row-major so element-wise
/// `.get()` assertions return the values one would index in RM.
fn to_rm<T: Clone>(tensor: &DenseTensor<T, NativeBackend>) -> DenseTensor<T, NativeBackend> {
    let dense = tensor.data().as_dense();
    let rm = arnet_tensor::reorder(&dense, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);
    DenseTensor::with_backend(rm.into_tensor_data(), NativeBackend::shared())
}

#[test]
fn test_contract_matmul() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = to_rm(&contract(&a, &b, "ik,kj->ij").unwrap());

    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[1, 0]), 43.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}

#[test]
fn test_contract_tensor_contraction() {
    let a = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );

    let c = to_rm(&contract(&a, &b, "ijk,jkl->il").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_ne!(c.get(&[0, 0]), 0.0);
}

#[test]
fn test_contract_f32() {
    let a = cm(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = to_rm(&contract(&a, &b, "ik,kj->ij").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0f32);
}

#[test]
fn test_contract_with_permutation() {
    let a = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let c = to_rm(&contract(&a, &b, "ikj,kj->i").unwrap());

    assert_eq!(c.shape(), &[2]);
    assert_ne!(c.data_slice()[0], 0.0);
}

#[test]
fn test_contract_rectangular() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);

    let c = contract(&a, &b, "ik,kj->ij").unwrap();

    assert_eq!(c.shape(), &[2, 3]);
}

#[test]
fn test_contract_invalid_notation() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = contract(&a, &b, "ik,kj->im");
    assert!(result.is_err());
}

#[test]
fn test_contract_rank_mismatch() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = contract(&a, &b, "ijk,kl->ijl");
    assert!(result.is_err());
}

// ============================================================================
// Output index reordering tests
// ============================================================================

#[test]
fn test_contract_output_reorder_swap() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = cm((1..=12).map(|x| x as f64).collect(), vec![3, 4]);

    let normal = to_rm(&contract(&a, &b, "ik,kj->ij").unwrap());
    let swapped = to_rm(&contract(&a, &b, "ik,kj->ji").unwrap());

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
    let a = cm((1..=12).map(|x| x as f64).collect(), vec![2, 3, 2]);
    let b = cm((1..=8).map(|x| x as f64).collect(), vec![2, 4]);

    let normal = to_rm(&contract(&a, &b, "abc,cd->abd").unwrap());
    let reordered = to_rm(&contract(&a, &b, "abc,cd->dba").unwrap());

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
    let a = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = cm(
        vec![1.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        vec![2, 2, 2],
    );

    let result = contract(&a, &b, "bik,bkj->bij");
    assert!(result.is_err());
}

#[test]
fn test_contract_output_memory_order() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let _c = contract(&a, &b, "ik,kj->ij").unwrap();
    let _c_reordered = contract(&a, &b, "ik,kj->ji").unwrap();
}
