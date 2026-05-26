//! Tests for einsum: single-tensor, 2-tensor, and N-tensor operations

use arnet_linalg::einsum;
use arnet_native::NativeBackend;
use arnet_tensor::{DenseTensor, DenseTensorData, MemoryOrder};

/// Create Dense from row-major data, converted to column-major for NativeBackend.
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> DenseTensor<T, NativeBackend> {
    let rm = DenseTensorData::from_raw_parts(data, shape, MemoryOrder::RowMajor);
    let cm = arnet_tensor::reorder_data(&rm, MemoryOrder::ColumnMajor);
    DenseTensor::with_backend(cm, NativeBackend::shared())
}

/// Convert column-major Dense back to row-major so `.get()` returns correct values.
fn to_rm<T: Clone>(tensor: &DenseTensor<T, NativeBackend>) -> DenseTensor<T, NativeBackend> {
    let rm = arnet_tensor::reorder_data(tensor.data(), MemoryOrder::RowMajor);
    DenseTensor::with_backend(rm, NativeBackend::shared())
}

// ============================================================================
// Transpose / permutation (no repeated indices)
// ============================================================================

#[test]
fn test_einsum_transpose_2d() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let b = to_rm(&einsum(&[&a], "ij->ji").unwrap());

    assert_eq!(b.shape(), &[3, 2]);
    assert_eq!(b.get(&[0, 0]), 1.0);
    assert_eq!(b.get(&[0, 1]), 4.0);
    assert_eq!(b.get(&[1, 0]), 2.0);
    assert_eq!(b.get(&[1, 1]), 5.0);
    assert_eq!(b.get(&[2, 0]), 3.0);
    assert_eq!(b.get(&[2, 1]), 6.0);
}

#[test]
fn test_einsum_permutation_3d() {
    let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    let a = cm(data, vec![2, 3, 4]);

    let b = einsum(&[&a], "ijk->kji").unwrap();
    let a_rm = to_rm(&a);
    let b_rm = to_rm(&b);

    assert_eq!(b.shape(), &[4, 3, 2]);
    assert_eq!(b_rm.get(&[0, 0, 0]), a_rm.get(&[0, 0, 0]));
    assert_eq!(b_rm.get(&[3, 2, 1]), a_rm.get(&[1, 2, 3]));
    assert_eq!(b_rm.get(&[2, 1, 0]), a_rm.get(&[0, 1, 2]));
}

#[test]
fn test_einsum_identity_permutation() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let b = einsum(&[&a], "ij->ij").unwrap();

    assert_eq!(b.shape(), &[2, 2]);
    assert_eq!(b.data_slice(), a.data_slice());
}

// ============================================================================
// Trace (repeated indices not in output)
// ============================================================================

#[test]
fn test_einsum_full_trace() {
    let a = cm(
        vec![1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        vec![3, 3],
    );

    let b = einsum(&[&a], "ii->").unwrap();

    assert_eq!(b.shape(), &[1]);
    assert_eq!(b.data_slice()[0], 6.0);
}

#[test]
fn test_einsum_partial_trace() {
    let a = cm(
        vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
    );

    let b = einsum(&[&a], "iij->j").unwrap();

    assert_eq!(b.shape(), &[3]);
    // B[j] = A[0,0,j] + A[1,1,j] = (1+10, 2+11, 3+12) = (11, 13, 15)
    assert_eq!(b.data_slice()[0], 11.0);
    assert_eq!(b.data_slice()[1], 13.0);
    assert_eq!(b.data_slice()[2], 15.0);
}

// ============================================================================
// Trace + transpose
// ============================================================================

#[test]
fn test_einsum_trace_then_transpose() {
    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let a = cm(data, vec![2, 3, 2]);

    let b = einsum(&[&a], "iji->j").unwrap();

    assert_eq!(b.shape(), &[3]);
    // B[j] = A[0,j,0] + A[1,j,1]
    let a_rm = to_rm(&a);
    let b0 = a_rm.get(&[0, 0, 0]) + a_rm.get(&[1, 0, 1]);
    let b1 = a_rm.get(&[0, 1, 0]) + a_rm.get(&[1, 1, 1]);
    let b2 = a_rm.get(&[0, 2, 0]) + a_rm.get(&[1, 2, 1]);
    assert!((b.data_slice()[0] - b0).abs() < 1e-10);
    assert!((b.data_slice()[1] - b1).abs() < 1e-10);
    assert!((b.data_slice()[2] - b2).abs() < 1e-10);
}

#[test]
fn test_einsum_trace_and_permute() {
    let data: Vec<f64> = (1..=48).map(|x| x as f64).collect();
    let a = cm(data, vec![2, 3, 4, 2]);

    let b = to_rm(&einsum(&[&a], "ijki->kj").unwrap());
    let a_rm = to_rm(&a);

    assert_eq!(b.shape(), &[4, 3]);
    // B[k,j] = A[0,j,k,0] + A[1,j,k,1]
    let b00 = a_rm.get(&[0, 0, 0, 0]) + a_rm.get(&[1, 0, 0, 1]);
    assert_eq!(b.get(&[0, 0]), b00);

    let b21 = a_rm.get(&[0, 1, 2, 0]) + a_rm.get(&[1, 1, 2, 1]);
    assert_eq!(b.get(&[2, 1]), b21);
}

// ============================================================================
// 2-input delegation (verify einsum dispatches to contract)
// ============================================================================

#[test]
fn test_einsum_two_input_matmul() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = to_rm(&einsum(&[&a, &b], "ij,jk->ik").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[1, 0]), 43.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}

// ============================================================================
// N-tensor pairwise reduction (3+ tensors)
// ============================================================================

#[test]
fn test_einsum_3_tensor_chain() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = cm((1..=12).map(|x| x as f64).collect(), vec![3, 4]);
    let c = cm((1..=8).map(|x| x as f64).collect(), vec![4, 2]);

    let d = einsum(&[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert_eq!(d.shape(), &[2, 2]);

    // Verify against manual 2-step contraction (both in CM, compare data directly)
    use arnet_linalg::contract;
    let ab = contract(&a, &b, "ij,jk->ik").unwrap();
    let expected = contract(&ab, &c, "ik,kl->il").unwrap();
    for i in 0..d.len() {
        assert!((d.data_slice()[i] - expected.data_slice()[i]).abs() < 1e-10);
    }
}

#[test]
fn test_einsum_3_tensor_implicit_output() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = cm(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]);

    let d = einsum(&[&a, &b, &c], "ij,jk,kl").unwrap();
    let d_explicit = einsum(&[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert_eq!(d.shape(), d_explicit.shape());
    // Compare data arrays directly (both in same CM layout)
    for i in 0..d.len() {
        assert!((d.data_slice()[i] - d_explicit.data_slice()[i]).abs() < 1e-10);
    }
}

#[test]
fn test_einsum_4_tensor_chain() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = cm(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]);
    let d = cm(vec![2.0_f64, 1.0, 1.0, 2.0], vec![2, 2]);

    let result = einsum(&[&a, &b, &c, &d], "ij,jk,kl,lm->im").unwrap();

    assert_eq!(result.shape(), &[2, 2]);

    // Verify against sequential contraction (both in CM, compare data directly)
    use arnet_linalg::contract;
    let ab = contract(&a, &b, "ij,jk->ik").unwrap();
    let abc = contract(&ab, &c, "ik,kl->il").unwrap();
    let expected = contract(&abc, &d, "il,lm->im").unwrap();
    for i in 0..result.len() {
        assert!((result.data_slice()[i] - expected.data_slice()[i]).abs() < 1e-10);
    }
}

#[test]
fn test_einsum_3_tensor_trace_of_product() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = cm(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]);

    let result = einsum(&[&a, &b, &c], "ij,jk,ki->").unwrap();

    assert_eq!(result.shape(), &[1]);

    // A·B = [[19,22],[43,50]], A·B·C = [[19,22],[43,50]], tr = 19+50 = 69
    assert_eq!(result.data_slice()[0], 69.0);
}

#[test]
fn test_einsum_2_tensor_hadamard() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![2.0_f64, 3.0, 4.0, 5.0], vec![2, 2]);

    let c = to_rm(&einsum(&[&a, &b], "ij,ij->ij").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 2.0);
    assert_eq!(c.get(&[0, 1]), 6.0);
    assert_eq!(c.get(&[1, 0]), 12.0);
    assert_eq!(c.get(&[1, 1]), 20.0);
}

// ============================================================================
// Error cases
// ============================================================================

#[test]
fn test_einsum_wrong_tensor_count() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let result = einsum(&[&a], "ij,jk->ik");
    assert!(result.is_err());
}

#[test]
fn test_einsum_rank_mismatch() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let result = einsum(&[&a], "ijk->kji");
    assert!(result.is_err());
}

#[test]
fn test_einsum_diagonal_extraction_unsupported() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let result = einsum(&[&a], "ii->i");
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("diagonal extraction"));
}

#[test]
fn test_einsum_reduction_unsupported() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let result = einsum(&[&a], "ij->i");
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("reduction"));
}

// ============================================================================
// Output index reordering (via einsum → contract)
// ============================================================================

#[test]
fn test_einsum_output_reorder_matmul() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let normal = to_rm(&einsum(&[&a, &b], "ij,jk->ik").unwrap());
    let swapped = to_rm(&einsum(&[&a, &b], "ij,jk->ki").unwrap());

    assert_eq!(swapped.shape(), &[2, 2]);
    for i in 0..2 {
        for k in 0..2 {
            assert_eq!(swapped.get(&[k, i]), normal.get(&[i, k]));
        }
    }
}

#[test]
fn test_einsum_output_reorder_rectangular() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = cm((1..=12).map(|x| x as f64).collect(), vec![3, 4]);

    let normal = to_rm(&einsum(&[&a, &b], "ij,jk->ik").unwrap());
    let swapped = to_rm(&einsum(&[&a, &b], "ij,jk->ki").unwrap());

    assert_eq!(normal.shape(), &[2, 4]);
    assert_eq!(swapped.shape(), &[4, 2]);

    for i in 0..2 {
        for k in 0..4 {
            assert_eq!(swapped.get(&[k, i]), normal.get(&[i, k]));
        }
    }
}

// ============================================================================
// Batch and Hadamard routing (einsum_pair dispatcher)
// ============================================================================

#[test]
fn test_einsum_batched_matmul() {
    let a = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = cm(
        vec![1.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        vec![2, 2, 2],
    );

    let normal = to_rm(&einsum(&[&a, &b], "bik,bkj->bij").unwrap());

    assert_eq!(normal.shape(), &[2, 2, 2]);
    // Batch 0: [[1,2],[3,4]] × I = [[1,2],[3,4]]
    assert_eq!(normal.get(&[0, 0, 0]), 1.0);
    assert_eq!(normal.get(&[0, 0, 1]), 2.0);
    assert_eq!(normal.get(&[0, 1, 0]), 3.0);
    assert_eq!(normal.get(&[0, 1, 1]), 4.0);
    // Batch 1: [[5,6],[7,8]] × 2I = [[10,12],[14,16]]
    assert_eq!(normal.get(&[1, 0, 0]), 10.0);
    assert_eq!(normal.get(&[1, 0, 1]), 12.0);
    assert_eq!(normal.get(&[1, 1, 0]), 14.0);
    assert_eq!(normal.get(&[1, 1, 1]), 16.0);
}

#[test]
fn test_einsum_batched_output_reorder_bji() {
    let a = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = cm(
        vec![1.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        vec![2, 2, 2],
    );

    let normal = to_rm(&einsum(&[&a, &b], "bik,bkj->bij").unwrap());
    let swapped = to_rm(&einsum(&[&a, &b], "bik,bkj->bji").unwrap());

    assert_eq!(swapped.shape(), &[2, 2, 2]);
    for batch in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(swapped.get(&[batch, j, i]), normal.get(&[batch, i, j]));
            }
        }
    }
}

#[test]
fn test_einsum_batched_output_reorder_jbi() {
    let a = cm(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = cm(
        vec![1.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        vec![2, 2, 2],
    );

    let normal = to_rm(&einsum(&[&a, &b], "bik,bkj->bij").unwrap());
    let reordered = to_rm(&einsum(&[&a, &b], "bik,bkj->jbi").unwrap());

    assert_eq!(reordered.shape(), &[2, 2, 2]);
    for batch in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(reordered.get(&[j, batch, i]), normal.get(&[batch, i, j]));
            }
        }
    }
}

#[test]
fn test_einsum_multi_with_intermediate_batch() {
    let t1 = cm((1..=12).map(|x| x as f64).collect(), vec![2, 2, 3]);
    let t2 = cm((1..=18).map(|x| x as f64).collect(), vec![2, 3, 3]);
    let t3 = cm((1..=6).map(|x| x as f64).collect(), vec![2, 3]);

    let result = to_rm(&einsum(&[&t1, &t2, &t3], "aij,ajk,ak->ai").unwrap());
    let t1_rm = to_rm(&t1);
    let t2_rm = to_rm(&t2);
    let t3_rm = to_rm(&t3);

    assert_eq!(result.shape(), &[2, 2]);
    for a in 0..2 {
        for i in 0..2 {
            let mut expected = 0.0;
            for j in 0..3 {
                for k in 0..3 {
                    expected += t1_rm.get(&[a, i, j]) * t2_rm.get(&[a, j, k]) * t3_rm.get(&[a, k]);
                }
            }
            assert!(
                (result.get(&[a, i]) - expected).abs() < 1e-10,
                "mismatch at [{a},{i}]: got {} expected {expected}",
                result.get(&[a, i])
            );
        }
    }
}

#[test]
fn test_einsum_batched_scalar_reduction() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0_f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = einsum(&[&a, &b], "bi,bi->b").unwrap();

    assert_eq!(result.shape(), &[2]);
    // Batch 0: 1*5 + 2*6 = 17
    assert_eq!(result.data_slice()[0], 17.0);
    // Batch 1: 3*7 + 4*8 = 53
    assert_eq!(result.data_slice()[1], 53.0);
}

#[test]
fn test_einsum_batched_multi_contracted_different_order() {
    let a = cm((1..=24).map(|x| x as f64).collect(), vec![2, 2, 3, 2]);
    let b_tensor = cm((1..=24).map(|x| x as f64).collect(), vec![2, 2, 3, 2]);

    let result = to_rm(&einsum(&[&a, &b_tensor], "bkli,bjlk->bij").unwrap());
    let a_rm = to_rm(&a);
    let b_rm = to_rm(&b_tensor);

    assert_eq!(result.shape(), &[2, 2, 2]);
    for bi in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                let mut expected = 0.0;
                for k in 0..2 {
                    for l in 0..3 {
                        expected += a_rm.get(&[bi, k, l, i]) * b_rm.get(&[bi, j, l, k]);
                    }
                }
                assert!(
                    (result.get(&[bi, i, j]) - expected).abs() < 1e-10,
                    "mismatch at [{bi},{i},{j}]: got {} expected {expected}",
                    result.get(&[bi, i, j])
                );
            }
        }
    }
}
