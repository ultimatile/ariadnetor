use ariadnetor_core::backend::ComputeBackend;
use ariadnetor_linalg::{DenseHostOps, LinalgError};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseTensor, DenseTensorData, MemoryOrder};

/// Build a `DenseTensor` from row-major data, reordered to column-major
/// (the NativeBackend's preferred order).
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> DenseTensor<T> {
    let rm = DenseTensorData::from_raw_parts(data, shape, MemoryOrder::RowMajor);
    let cm = ariadnetor_tensor::reorder_data(&rm, MemoryOrder::ColumnMajor);
    DenseTensor::from_data(cm)
}

/// Reorder a `DenseTensor` result back to row-major so element-wise
/// `.get()` assertions return the values one would index in RM.
fn to_rm<T: Clone>(tensor: &DenseTensor<T>) -> DenseTensor<T> {
    let rm = ariadnetor_tensor::reorder_data(tensor.data(), MemoryOrder::RowMajor);
    DenseTensor::from_data(rm)
}

#[test]
fn test_contract_matmul() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = to_rm(&a.contract(&b, "ik,kj->ij").unwrap());

    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get([0, 0]), 19.0);
    assert_eq!(c.get([0, 1]), 22.0);
    assert_eq!(c.get([1, 0]), 43.0);
    assert_eq!(c.get([1, 1]), 50.0);
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

    let c = to_rm(&a.contract(&b, "ijk,jkl->il").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_ne!(c.get([0, 0]), 0.0);
}

#[test]
fn test_contract_f32() {
    let a = cm(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = to_rm(&a.contract(&b, "ik,kj->ij").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get([0, 0]), 19.0f32);
}

#[test]
fn test_contract_with_permutation() {
    let a = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let c = to_rm(&a.contract(&b, "ikj,kj->i").unwrap());

    assert_eq!(c.shape(), &[2]);
    assert_ne!(c.data_slice()[0], 0.0);
}

#[test]
fn test_contract_rectangular() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);

    let c = a.contract(&b, "ik,kj->ij").unwrap();

    assert_eq!(c.shape(), &[2, 3]);
}

#[test]
fn test_contract_invalid_notation() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = a.contract(&b, "ik,kj->im");
    assert!(result.is_err());
}

#[test]
fn test_contract_rank_mismatch() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = a.contract(&b, "ijk,kl->ijl");
    assert!(result.is_err());
}

#[test]
fn test_contract_rejects_mismatched_contracted_extents() {
    // Notation path: the paired contracted index `k` has extent 3 on the left and
    // 4 on the right. Must return InvalidArgument, not panic in the GEMM reshape.
    let a = DenseTensor::<f64>::zeros(vec![2, 3]);
    let b = DenseTensor::<f64>::zeros(vec![4, 5]);
    assert!(matches!(
        a.contract(&b, "ik,kj->ij"),
        Err(LinalgError::InvalidArgument(_))
    ));
}

#[test]
fn test_contract_zero_length_contracted_axis() {
    // Notation path: a zero-extent contracted axis is an empty sum, so the result
    // is the zero tensor of the free shape rather than a panic.
    let a = DenseTensor::<f64>::zeros(vec![2, 0]);
    let b = DenseTensor::<f64>::zeros(vec![0, 3]);
    let c = a.contract(&b, "ik,kj->ij").unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    assert!(c.data_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn test_contract_zero_length_free_axis() {
    // Notation path: a zero-extent free axis yields an empty tensor (the shape
    // carries the zero), again rather than a panic.
    let a = DenseTensor::<f64>::zeros(vec![0, 3]);
    let b = DenseTensor::<f64>::zeros(vec![3, 4]);
    let c = a.contract(&b, "ik,kj->ij").unwrap();
    assert_eq!(c.shape(), &[0, 4]);
    assert_eq!(c.data_slice().len(), 0);
}

// ============================================================================
// Output index reordering tests
// ============================================================================

#[test]
fn test_contract_output_reorder_swap() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = cm((1..=12).map(|x| x as f64).collect(), vec![3, 4]);

    let normal = to_rm(&a.contract(&b, "ik,kj->ij").unwrap());
    let swapped = to_rm(&a.contract(&b, "ik,kj->ji").unwrap());

    assert_eq!(normal.shape(), &[2, 4]);
    assert_eq!(swapped.shape(), &[4, 2]);

    for i in 0..2 {
        for j in 0..4 {
            assert_eq!(swapped.get([j, i]), normal.get([i, j]));
        }
    }
}

#[test]
fn test_contract_output_reorder_3d() {
    let a = cm((1..=12).map(|x| x as f64).collect(), vec![2, 3, 2]);
    let b = cm((1..=8).map(|x| x as f64).collect(), vec![2, 4]);

    let normal = to_rm(&a.contract(&b, "abc,cd->abd").unwrap());
    let reordered = to_rm(&a.contract(&b, "abc,cd->dba").unwrap());

    assert_eq!(normal.shape(), &[2, 3, 4]);
    assert_eq!(reordered.shape(), &[4, 3, 2]);

    for a_idx in 0..2 {
        for b_idx in 0..3 {
            for d in 0..4 {
                assert_eq!(
                    reordered.get([d, b_idx, a_idx]),
                    normal.get([a_idx, b_idx, d]),
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

    let result = a.contract(&b, "bik,bkj->bij");
    assert!(result.is_err());
}

#[test]
fn test_contract_output_memory_order() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = a.contract(&b, "ik,kj->ij").unwrap();
    let c_reordered = a.contract(&b, "ik,kj->ji").unwrap();

    // Pins the contract docstring: output is returned in the host backend's
    // `preferred_order()`, independent of the einsum's output index permutation
    // (`->ji` exercises non-trivial permutation; both calls must still land in
    // the backend's preferred order). The host-ext `contract` routes through
    // `NativeBackend`, so query its preferred order directly.
    let expected = NativeBackend::new().preferred_order();
    assert_eq!(c.order(), expected);
    assert_eq!(c_reordered.order(), expected);
}
