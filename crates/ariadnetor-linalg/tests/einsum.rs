//! Tests for einsum: single-tensor, 2-tensor, and N-tensor operations

use arnet_linalg::einsum;
use arnet_native::NativeBackend;
use arnet_tensor::DenseTensor;

// ============================================================================
// Transpose / permutation (no repeated indices)
// ============================================================================

#[test]
fn test_einsum_transpose_2d() {
    let backend = NativeBackend::new();
    // 2×3 matrix
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let b = einsum(&backend, &[&a], "ij->ji").unwrap();

    assert_eq!(b.shape(), &[3, 2]);
    // Row-major [1,2,3,4,5,6] transposed → [1,4,2,5,3,6]
    assert_eq!(b.get(&[0, 0]), 1.0);
    assert_eq!(b.get(&[0, 1]), 4.0);
    assert_eq!(b.get(&[1, 0]), 2.0);
    assert_eq!(b.get(&[1, 1]), 5.0);
    assert_eq!(b.get(&[2, 0]), 3.0);
    assert_eq!(b.get(&[2, 1]), 6.0);
}

#[test]
fn test_einsum_permutation_3d() {
    let backend = NativeBackend::new();
    // 2×3×4 tensor
    let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    let a = DenseTensor::from_data(data, vec![2, 3, 4]);

    let b = einsum(&backend, &[&a], "ijk->kji").unwrap();

    assert_eq!(b.shape(), &[4, 3, 2]);
    // A[0,0,0] = 1 → B[0,0,0] = 1
    assert_eq!(b.get(&[0, 0, 0]), a.get(&[0, 0, 0]));
    // A[1,2,3] → B[3,2,1]
    assert_eq!(b.get(&[3, 2, 1]), a.get(&[1, 2, 3]));
    // A[0,1,2] → B[2,1,0]
    assert_eq!(b.get(&[2, 1, 0]), a.get(&[0, 1, 2]));
}

#[test]
fn test_einsum_identity_permutation() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // Identity permutation: no actual transpose needed
    let b = einsum(&backend, &[&a], "ij->ij").unwrap();

    assert_eq!(b.shape(), &[2, 2]);
    assert_eq!(b.data(), a.data());
}

// ============================================================================
// Trace (repeated indices not in output)
// ============================================================================

#[test]
fn test_einsum_full_trace() {
    let backend = NativeBackend::new();
    // 3×3 matrix
    let a = DenseTensor::from_data(
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        vec![3, 3],
    );

    let b = einsum(&backend, &[&a], "ii->").unwrap();

    // Trace = 1 + 2 + 3 = 6
    assert_eq!(b.shape(), &[1]);
    assert_eq!(b.get(&[0]), 6.0);
}

#[test]
fn test_einsum_partial_trace() {
    let backend = NativeBackend::new();
    // 2×2×3 tensor: A[i,i,j] → sum over diagonal of i, keep j
    let a = DenseTensor::from_data(
        vec![
            1.0, 2.0, 3.0, // [0,0,:]
            4.0, 5.0, 6.0, // [0,1,:]
            7.0, 8.0, 9.0, // [1,0,:]
            10.0, 11.0, 12.0, // [1,1,:]
        ],
        vec![2, 2, 3],
    );

    let b = einsum(&backend, &[&a], "iij->j").unwrap();

    assert_eq!(b.shape(), &[3]);
    // B[j] = A[0,0,j] + A[1,1,j]
    // B[0] = 1 + 10 = 11
    // B[1] = 2 + 11 = 13
    // B[2] = 3 + 12 = 15
    assert_eq!(b.get(&[0]), 11.0);
    assert_eq!(b.get(&[1]), 13.0);
    assert_eq!(b.get(&[2]), 15.0);
}

// ============================================================================
// Trace + transpose
// ============================================================================

#[test]
fn test_einsum_trace_then_transpose() {
    let backend = NativeBackend::new();
    // 2×3×2 tensor: "iji->j" traces i (positions 0,2), keeps j
    // This is a valid trace+result case
    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let a = DenseTensor::from_data(data, vec![2, 3, 2]);

    let b = einsum(&backend, &[&a], "iji->j").unwrap();

    assert_eq!(b.shape(), &[3]);
    // B[j] = A[0,j,0] + A[1,j,1]
    // A[0,0,0]=1, A[1,0,1]=8 → B[0]=9
    // A[0,1,0]=3, A[1,1,1]=10 → B[1]=13
    // A[0,2,0]=5, A[1,2,1]=12 → B[2]=17
    assert_eq!(b.get(&[0]), 9.0);
    assert_eq!(b.get(&[1]), 13.0);
    assert_eq!(b.get(&[2]), 17.0);
}

#[test]
fn test_einsum_trace_and_permute() {
    let backend = NativeBackend::new();
    // 2×3×4×2 tensor: "ijki->kj" traces i (positions 0,3), keeps j,k → permute to k,j
    let data: Vec<f64> = (1..=48).map(|x| x as f64).collect();
    let a = DenseTensor::from_data(data, vec![2, 3, 4, 2]);

    let b = einsum(&backend, &[&a], "ijki->kj").unwrap();

    assert_eq!(b.shape(), &[4, 3]);
    // After trace: C[j,k] = A[0,j,k,0] + A[1,j,k,1]
    // A is row-major [2,3,4,2]: A[i,j,k,l] = data[i*24 + j*8 + k*2 + l]
    // C[0,0] = A[0,0,0,0] + A[1,0,0,1] = 1 + 26 = 27
    // Then B[k,j] = C[j,k], so B[0,0] = C[0,0] = 27
    assert_eq!(b.get(&[0, 0]), 27.0);

    // C[1,2] = A[0,1,2,0] + A[1,1,2,1] = 13 + 38 = 51
    // B[2,1] = C[1,2] = 51
    assert_eq!(b.get(&[2, 1]), 51.0);
}

// ============================================================================
// 2-input delegation (verify einsum dispatches to contract)
// ============================================================================

#[test]
fn test_einsum_two_input_matmul() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = einsum(&backend, &[&a, &b], "ij,jk->ik").unwrap();

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
    let backend = NativeBackend::new();
    // A(2×3) · B(3×4) · C(4×2) = D(2×2)
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = DenseTensor::from_data((1..=12).map(|x| x as f64).collect(), vec![3, 4]);
    let c = DenseTensor::from_data((1..=8).map(|x| x as f64).collect(), vec![4, 2]);

    let d = einsum(&backend, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert_eq!(d.shape(), &[2, 2]);

    // Verify against manual 2-step contraction
    use arnet_linalg::contract;
    let ab = contract(&backend, &a, &b, "ij,jk->ik").unwrap();
    let expected = contract(&backend, &ab, &c, "ik,kl->il").unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(d.get(&[i, j]), expected.get(&[i, j]));
        }
    }
}

#[test]
fn test_einsum_3_tensor_implicit_output() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = DenseTensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    // Implicit output: "ij,jk,kl" → free indices i,l → "ij,jk,kl->il"
    let d = einsum(&backend, &[&a, &b, &c], "ij,jk,kl").unwrap();
    let d_explicit = einsum(&backend, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert_eq!(d.shape(), d_explicit.shape());
    for i in 0..d.len() {
        assert_eq!(d.get(&[i / 2, i % 2]), d_explicit.get(&[i / 2, i % 2]));
    }
}

#[test]
fn test_einsum_4_tensor_chain() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = DenseTensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let d = DenseTensor::from_data(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]);

    let result = einsum(&backend, &[&a, &b, &c, &d], "ij,jk,kl,lm->im").unwrap();

    assert_eq!(result.shape(), &[2, 2]);

    // Verify against sequential contraction
    use arnet_linalg::contract;
    let ab = contract(&backend, &a, &b, "ij,jk->ik").unwrap();
    let abc = contract(&backend, &ab, &c, "ik,kl->il").unwrap();
    let expected = contract(&backend, &abc, &d, "il,lm->im").unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected.get(&[i, j]));
        }
    }
}

#[test]
fn test_einsum_3_tensor_trace_of_product() {
    let backend = NativeBackend::new();
    // tr(A · B · C) = "ij,jk,ki->"
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = DenseTensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    let result = einsum(&backend, &[&a, &b, &c], "ij,jk,ki->").unwrap();

    assert_eq!(result.shape(), &[1]);

    // A·B = [[19,22],[43,50]], A·B·C = [[19,22],[43,50]], tr = 19+50 = 69
    assert_eq!(result.get(&[0]), 69.0);
}

#[test]
fn test_einsum_2_tensor_hadamard() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);

    let c = einsum(&backend, &[&a, &b], "ij,ij->ij").unwrap();

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
    let backend = NativeBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // Notation expects 2 inputs but only 1 given
    let result = einsum(&backend, &[&a], "ij,jk->ik");
    assert!(result.is_err());
}

#[test]
fn test_einsum_rank_mismatch() {
    let backend = NativeBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // 3-index notation with rank-2 tensor
    let result = einsum(&backend, &[&a], "ijk->kji");
    assert!(result.is_err());
}

#[test]
fn test_einsum_diagonal_extraction_unsupported() {
    let backend = NativeBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // "ii->i" is diagonal extraction, not yet supported
    let result = einsum(&backend, &[&a], "ii->i");
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("diagonal extraction"));
}

#[test]
fn test_einsum_reduction_unsupported() {
    let backend = NativeBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // "ij->i" is a sum over j, not supported as single-tensor einsum
    let result = einsum(&backend, &[&a], "ij->i");
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("reduction"));
}

// ============================================================================
// Output index reordering (via einsum → contract)
// ============================================================================

#[test]
fn test_einsum_output_reorder_matmul() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    // "ij,jk->ki" should transpose the matmul result
    let normal = einsum(&backend, &[&a, &b], "ij,jk->ik").unwrap();
    let swapped = einsum(&backend, &[&a, &b], "ij,jk->ki").unwrap();

    assert_eq!(swapped.shape(), &[2, 2]);
    for i in 0..2 {
        for k in 0..2 {
            assert_eq!(swapped.get(&[k, i]), normal.get(&[i, k]));
        }
    }
}

#[test]
fn test_einsum_output_reorder_rectangular() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = DenseTensor::from_data((1..=12).map(|x| x as f64).collect(), vec![3, 4]);

    let normal = einsum(&backend, &[&a, &b], "ij,jk->ik").unwrap();
    let swapped = einsum(&backend, &[&a, &b], "ij,jk->ki").unwrap();

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
    let backend = NativeBackend::new();
    // Batched matmul: 2 batches of 2×2 matrices
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let b = DenseTensor::from_data(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2]);

    let normal = einsum(&backend, &[&a, &b], "bik,bkj->bij").unwrap();

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
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let b = DenseTensor::from_data(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2]);

    let normal = einsum(&backend, &[&a, &b], "bik,bkj->bij").unwrap();
    let swapped = einsum(&backend, &[&a, &b], "bik,bkj->bji").unwrap();

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
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let b = DenseTensor::from_data(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2]);

    let normal = einsum(&backend, &[&a, &b], "bik,bkj->bij").unwrap();
    let reordered = einsum(&backend, &[&a, &b], "bik,bkj->jbi").unwrap();

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
    let backend = NativeBackend::new();
    // "aij,ajk,ak->ai" with left-to-right generates intermediate "aij,ajk->aik" (batch on a)
    let t1 = DenseTensor::from_data((1..=12).map(|x| x as f64).collect(), vec![2, 2, 3]);
    let t2 = DenseTensor::from_data((1..=18).map(|x| x as f64).collect(), vec![2, 3, 3]);
    let t3 = DenseTensor::from_data((1..=6).map(|x| x as f64).collect(), vec![2, 3]);

    let result = einsum(&backend, &[&t1, &t2, &t3], "aij,ajk,ak->ai").unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    // Verify by manual contraction: sum_j sum_k t1[a,i,j] * t2[a,j,k] * t3[a,k]
    for a in 0..2 {
        for i in 0..2 {
            let mut expected = 0.0;
            for j in 0..3 {
                for k in 0..3 {
                    expected += t1.get(&[a, i, j]) * t2.get(&[a, j, k]) * t3.get(&[a, k]);
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
    let backend = NativeBackend::new();
    // "bi,bi->b": dot product per batch, output should be shape [batch], not [batch, 1]
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = einsum(&backend, &[&a, &b], "bi,bi->b").unwrap();

    assert_eq!(result.shape(), &[2]);
    // Batch 0: 1*5 + 2*6 = 17
    assert_eq!(result.get(&[0]), 17.0);
    // Batch 1: 3*7 + 4*8 = 53
    assert_eq!(result.get(&[1]), 53.0);
}

#[test]
fn test_einsum_batched_multi_contracted_different_order() {
    let backend = NativeBackend::new();
    // Contracted indices k,l appear in different order in LHS vs RHS
    // "bkli,bjlk->bij": LHS has [k,l], RHS has [l,k]
    let a = DenseTensor::from_data(
        (1..=24).map(|x| x as f64).collect(),
        vec![2, 2, 3, 2], // b=2, k=2, l=3, i=2
    );
    let b_tensor = DenseTensor::from_data(
        (1..=24).map(|x| x as f64).collect(),
        vec![2, 2, 3, 2], // b=2, j=2, l=3, k=2
    );

    let result = einsum(&backend, &[&a, &b_tensor], "bkli,bjlk->bij").unwrap();

    assert_eq!(result.shape(), &[2, 2, 2]);
    // Verify: result[b,i,j] = sum_{k,l} a[b,k,l,i] * b[b,j,l,k]
    for bi in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                let mut expected = 0.0;
                for k in 0..2 {
                    for l in 0..3 {
                        expected += a.get(&[bi, k, l, i]) * b_tensor.get(&[bi, j, l, k]);
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
