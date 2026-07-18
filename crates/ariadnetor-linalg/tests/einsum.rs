//! Tests for einsum: single-tensor, 2-tensor, and N-tensor operations

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{DenseHostOps, LinalgError, einsum_with_backend};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseTensor, DenseTensorData, MemoryOrder};

/// Create Dense from row-major data, converted to column-major for NativeBackend.
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> DenseTensor<T> {
    let rm = DenseTensorData::from_raw_parts(data, shape, MemoryOrder::RowMajor);
    let cm = ariadnetor_tensor::reorder_data(&rm, MemoryOrder::ColumnMajor);
    DenseTensor::from_data(cm)
}

/// Run `einsum` on a fresh host `NativeBackend` (the default host path).
/// `einsum` has no host-defaulting method form, so the tests funnel through
/// this thin wrapper to keep the call sites focused on notation, not backend
/// plumbing.
fn einsum<T: Scalar>(
    tensors: &[&DenseTensor<T>],
    notation: &str,
) -> Result<DenseTensor<T>, LinalgError> {
    einsum_with_backend(&NativeBackend::new(), tensors, notation)
}

/// Convert column-major Dense back to row-major so `.get()` returns correct values.
fn to_rm<T: Clone>(tensor: &DenseTensor<T>) -> DenseTensor<T> {
    let rm = ariadnetor_tensor::reorder_data(tensor.data(), MemoryOrder::RowMajor);
    DenseTensor::from_data(rm)
}

// ============================================================================
// Transpose / permutation (no repeated indices)
// ============================================================================

#[test]
fn test_einsum_transpose_2d() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let b = to_rm(&einsum(&[&a], "ij->ji").unwrap());

    assert_eq!(b.shape(), &[3, 2]);
    assert_eq!(b.get([0, 0]), 1.0);
    assert_eq!(b.get([0, 1]), 4.0);
    assert_eq!(b.get([1, 0]), 2.0);
    assert_eq!(b.get([1, 1]), 5.0);
    assert_eq!(b.get([2, 0]), 3.0);
    assert_eq!(b.get([2, 1]), 6.0);
}

#[test]
fn test_einsum_permutation_3d() {
    let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    let a = cm(data, vec![2, 3, 4]);

    let b = einsum(&[&a], "ijk->kji").unwrap();
    let a_rm = to_rm(&a);
    let b_rm = to_rm(&b);

    assert_eq!(b.shape(), &[4, 3, 2]);
    assert_eq!(b_rm.get([0, 0, 0]), a_rm.get([0, 0, 0]));
    assert_eq!(b_rm.get([3, 2, 1]), a_rm.get([1, 2, 3]));
    assert_eq!(b_rm.get([2, 1, 0]), a_rm.get([0, 1, 2]));
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
    let b0 = a_rm.get([0, 0, 0]) + a_rm.get([1, 0, 1]);
    let b1 = a_rm.get([0, 1, 0]) + a_rm.get([1, 1, 1]);
    let b2 = a_rm.get([0, 2, 0]) + a_rm.get([1, 2, 1]);
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
    let b00 = a_rm.get([0, 0, 0, 0]) + a_rm.get([1, 0, 0, 1]);
    assert_eq!(b.get([0, 0]), b00);

    let b21 = a_rm.get([0, 1, 2, 0]) + a_rm.get([1, 1, 2, 1]);
    assert_eq!(b.get([2, 1]), b21);
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
    assert_eq!(c.get([0, 0]), 19.0);
    assert_eq!(c.get([0, 1]), 22.0);
    assert_eq!(c.get([1, 0]), 43.0);
    assert_eq!(c.get([1, 1]), 50.0);
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
    let ab = a.contract(&b, "ij,jk->ik").unwrap();
    let expected = ab.contract(&c, "ik,kl->il").unwrap();
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
    let ab = a.contract(&b, "ij,jk->ik").unwrap();
    let abc = ab.contract(&c, "ik,kl->il").unwrap();
    let expected = abc.contract(&d, "il,lm->im").unwrap();
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

    // The multi-operand chain's final pairwise step is a full contraction,
    // which now yields a rank-0 scalar (shape []) via the unified `contract`.
    assert_eq!(result.shape(), &[] as &[usize]);

    // A·B = [[19,22],[43,50]], A·B·C = [[19,22],[43,50]], tr = 19+50 = 69
    assert_eq!(result.data_slice()[0], 69.0);
}

#[test]
fn test_einsum_2_tensor_hadamard() {
    let a = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![2.0_f64, 3.0, 4.0, 5.0], vec![2, 2]);

    let c = to_rm(&einsum(&[&a, &b], "ij,ij->ij").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get([0, 0]), 2.0);
    assert_eq!(c.get([0, 1]), 6.0);
    assert_eq!(c.get([1, 0]), 12.0);
    assert_eq!(c.get([1, 1]), 20.0);
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
            assert_eq!(swapped.get([k, i]), normal.get([i, k]));
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
            assert_eq!(swapped.get([k, i]), normal.get([i, k]));
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
    assert_eq!(normal.get([0, 0, 0]), 1.0);
    assert_eq!(normal.get([0, 0, 1]), 2.0);
    assert_eq!(normal.get([0, 1, 0]), 3.0);
    assert_eq!(normal.get([0, 1, 1]), 4.0);
    // Batch 1: [[5,6],[7,8]] × 2I = [[10,12],[14,16]]
    assert_eq!(normal.get([1, 0, 0]), 10.0);
    assert_eq!(normal.get([1, 0, 1]), 12.0);
    assert_eq!(normal.get([1, 1, 0]), 14.0);
    assert_eq!(normal.get([1, 1, 1]), 16.0);
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
                assert_eq!(swapped.get([batch, j, i]), normal.get([batch, i, j]));
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
                assert_eq!(reordered.get([j, batch, i]), normal.get([batch, i, j]));
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
                    expected += t1_rm.get([a, i, j]) * t2_rm.get([a, j, k]) * t3_rm.get([a, k]);
                }
            }
            assert!(
                (result.get([a, i]) - expected).abs() < 1e-10,
                "mismatch at [{a},{i}]: got {} expected {expected}",
                result.get([a, i])
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
                        expected += a_rm.get([bi, k, l, i]) * b_rm.get([bi, j, l, k]);
                    }
                }
                assert!(
                    (result.get([bi, i, j]) - expected).abs() < 1e-10,
                    "mismatch at [{bi},{i},{j}]: got {} expected {expected}",
                    result.get([bi, i, j])
                );
            }
        }
    }
}

/// Four-operand contraction whose requested output order `(v, y, c)`
/// differs from the first-appearance order `(c, v, y)` of the surviving
/// indices. The pairwise recursion must emit the requested order, not its
/// internal first-appearance labeling (which is only an implementation
/// detail of the interior steps). The middle step also carries an index
/// (`c`) in both operands and the output, so the per-slice batched path is
/// exercised alongside the plain GEMM path.
#[test]
fn multi_tensor_honors_requested_output_order() {
    // Row-major fills through the same `cm` helper the rest of this file
    // uses, with distinct non-symmetric values so axis mixups cannot cancel.
    let filled = |shape: Vec<usize>| {
        let total: usize = shape.iter().product();
        let data = (1..=total).map(|i| (i as f64 * 0.7).sin()).collect();
        cm(data, shape)
    };
    let omega = filled(vec![2, 2]); // (c, b)
    let w = filled(vec![2, 2, 2, 2]); // (w, k, b, v)
    let e = filled(vec![2, 4, 2]); // (w, x, c)
    let a = filled(vec![4, 2, 4]); // (x, k, y)

    let out = einsum(&[&omega, &w, &e, &a], "cb,wkbv,wxc,xky->vyc").expect("einsum");
    assert_eq!(out.shape(), &[2, 4, 2], "requested output order (v, y, c)");

    for v in 0..2 {
        for y in 0..4 {
            for c in 0..2 {
                let mut want = 0.0;
                for b in 0..2 {
                    for wi in 0..2 {
                        for k in 0..2 {
                            for x in 0..4 {
                                want += omega.get([c, b])
                                    * w.get([wi, k, b, v])
                                    * e.get([wi, x, c])
                                    * a.get([x, k, y]);
                            }
                        }
                    }
                }
                let got = out.get([v, y, c]);
                assert!(
                    (got - want).abs() < 1e-12,
                    "mismatch at (v={v}, y={y}, c={c}): got {got}, want {want}"
                );
            }
        }
    }
}

/// Batched contraction with TWO batch labels (`a`, `b`), asymmetric GEMM
/// dimensions (`m=2`, `n=4`, `k=3` all distinct), and an interleaved output
/// order (`baji` — batch labels swapped, free legs reversed). This exercises
/// the batch-product flatten and the per-slice offset arithmetic together:
/// a single-batch or square-dimension case would let a stride or transpose
/// mixup cancel silently.
#[test]
fn test_einsum_batched_two_batch_labels_asymmetric() {
    // a=2, b=2, i(m)=2, k=3, j(n)=4
    let lhs = cm((1..=24).map(|x| x as f64).collect(), vec![2, 2, 2, 3]); // abik
    let rhs = cm((1..=48).map(|x| x as f64).collect(), vec![2, 2, 3, 4]); // abkj

    let result = to_rm(&einsum(&[&lhs, &rhs], "abik,abkj->baji").unwrap());
    let lhs_rm = to_rm(&lhs);
    let rhs_rm = to_rm(&rhs);

    assert_eq!(result.shape(), &[2, 2, 4, 2]); // baji
    for a in 0..2 {
        for b in 0..2 {
            for i in 0..2 {
                for j in 0..4 {
                    let mut expected = 0.0;
                    for k in 0..3 {
                        expected += lhs_rm.get([a, b, i, k]) * rhs_rm.get([a, b, k, j]);
                    }
                    assert!(
                        (result.get([b, a, j, i]) - expected).abs() < 1e-10,
                        "mismatch at b={b},a={a},j={j},i={i}: got {} expected {expected}",
                        result.get([b, a, j, i])
                    );
                }
            }
        }
    }
}

/// A batched contraction whose contracted extent disagrees across operands
/// must be rejected up front with `InvalidArgument`, not sliced into the
/// wrong region. Locks the validation that replaced the per-slice
/// `contract_dense` extent check.
#[test]
fn test_einsum_batched_mismatched_contracted_extent_errors() {
    let lhs = cm((1..=12).map(|x| x as f64).collect(), vec![2, 2, 3]); // bik, k=3
    let rhs = cm((1..=16).map(|x| x as f64).collect(), vec![2, 4, 2]); // bkj, k=4

    let result = einsum(&[&lhs, &rhs], "bik,bkj->bij");
    assert!(
        matches!(result, Err(LinalgError::InvalidArgument(_))),
        "expected InvalidArgument on mismatched contracted extent, got {result:?}"
    );
}

/// A batched notation whose operand rank exceeds its index count must be
/// rejected with `InvalidArgument` before the permute / slice arithmetic runs
/// on the extra axis. Locks the up-front rank guard.
#[test]
fn test_einsum_batched_rank_exceeds_arity_errors() {
    // "bik" names 3 indices but the operand is rank 4.
    let lhs = cm((1..=60).map(|x| x as f64).collect(), vec![2, 2, 3, 5]);
    let rhs = cm((1..=12).map(|x| x as f64).collect(), vec![2, 3, 2]); // bkj

    let result = einsum(&[&lhs, &rhs], "bik,bkj->bij");
    assert!(
        matches!(result, Err(LinalgError::InvalidArgument(_))),
        "expected InvalidArgument on operand rank / notation arity mismatch, got {result:?}"
    );
}

/// A batched contraction over a zero-extent contracted axis is an empty sum:
/// the result is the zeros of the output shape, not a slicing error. Exercises
/// the degenerate short-circuit in the batched path.
#[test]
fn test_einsum_batched_zero_contracted_extent_is_zeros() {
    let lhs = cm(Vec::<f64>::new(), vec![2, 2, 0]); // bik, k=0
    let rhs = cm(Vec::<f64>::new(), vec![2, 0, 3]); // bkj, k=0

    let result = to_rm(&einsum(&[&lhs, &rhs], "bik,bkj->bij").unwrap());
    assert_eq!(result.shape(), &[2, 2, 3]);
    assert!(
        result.data_slice().iter().all(|&x| x == 0.0),
        "empty contracted sum must be all zeros"
    );
}
