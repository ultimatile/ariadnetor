use arnet_linalg::{TruncSvdParams, lq, qr, svd, trunc_svd};
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, MemoryOrder};

/// Create Dense from row-major data, converted to column-major for NativeBackend.
fn cm<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> Dense<T> {
    let rm = Dense::new(data, shape, MemoryOrder::RowMajor);
    arnet_tensor::reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)
}

/// Convert column-major Dense back to row-major so `.get()` returns correct values.
fn to_rm<T: Clone>(tensor: &Dense<T>) -> Dense<T> {
    arnet_tensor::reorder(tensor, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor)
}

// --- SVD tests ---

#[test]
fn test_svd_f64_2d() {
    let backend = NativeBackend::new();
    // A = [[1, 2], [3, 4]] shape [2, 2], nrow=1 → 2×2 matrix
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();
    let u = to_rm(&u);
    let vt = to_rm(&vt);
    let tensor = to_rm(&tensor);

    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);

    // Singular values should be positive and descending
    assert!(s.data()[0] > s.data()[1]);
    assert!(s.data()[1] >= 0.0);

    // Reconstruct: A ≈ U * diag(S) * Vt
    for i in 0..2 {
        for j in 0..2 {
            let mut val = 0.0;
            for k in 0..2 {
                val += u.get(&[i, k]) * s.data()[k] * vt.get(&[k, j]);
            }
            assert!(
                (val - tensor.get(&[i, j])).abs() < 1e-10,
                "Reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_svd_f64_rectangular() {
    let backend = NativeBackend::new();
    // shape [2, 3], nrow=1 → 2×3 matrix, k=2
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();
    let u = to_rm(&u);
    let vt = to_rm(&vt);
    let tensor = to_rm(&tensor);

    let (m, n, k) = (2, 3, 2);
    assert_eq!(u.shape(), &[m, k]);
    assert_eq!(s.shape(), &[k]);
    assert_eq!(vt.shape(), &[k, n]);

    // Reconstruct
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += u.get(&[i, l]) * s.data()[l] * vt.get(&[l, j]);
            }
            assert!(
                (val - tensor.get(&[i, j])).abs() < 1e-10,
                "Reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_svd_f64_higher_rank() {
    let backend = NativeBackend::new();
    // shape [2, 3, 4], nrow=2 → m=6, n=4, k=4
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let tensor = cm(data, vec![2, 3, 4]);

    let (u, s, vt) = svd(&backend, &tensor, 2).unwrap();
    let u = to_rm(&u);
    let vt = to_rm(&vt);

    let (m, n, k) = (6, 4, 4);
    assert_eq!(u.shape(), &[m, k]);
    assert_eq!(s.shape(), &[k]);
    assert_eq!(vt.shape(), &[k, n]);

    // Reconstruct and verify: U*S*Vt should match the original (RM-flattened) data
    let tensor_rm = to_rm(&tensor);
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += u.get(&[i, l]) * s.data()[l] * vt.get(&[l, j]);
            }
            let orig = tensor_rm.data()[i * n + j];
            assert!(
                (val - orig).abs() < 1e-9,
                "Reconstruction mismatch at ({i},{j}): {val} vs {orig}"
            );
        }
    }
}

#[test]
fn test_svd_f32() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);

    let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();
    let u = to_rm(&u);
    let vt = to_rm(&vt);
    let tensor = to_rm(&tensor);

    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);

    for i in 0..2 {
        for j in 0..2 {
            let mut val = 0.0f32;
            for k in 0..2 {
                val += u.get(&[i, k]) * s.data()[k] * vt.get(&[k, j]);
            }
            assert!(
                (val - tensor.get(&[i, j])).abs() < 1e-4,
                "Reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_svd_invalid_nrow() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // nrow=0 is invalid
    assert!(svd(&backend, &tensor, 0).is_err());
    // nrow=rank is invalid
    assert!(svd(&backend, &tensor, 2).is_err());
}

// --- Truncated SVD tests ---

#[test]
fn test_trunc_svd_chi_max() {
    let backend = NativeBackend::new();
    // 3×4 matrix with rank > 1
    let tensor = cm(
        vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![3, 4],
    );

    let params = TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    };
    let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

    // Truncated to chi=2
    assert_eq!(u.shape(), &[3, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 4]);

    // Singular values should be positive and descending
    assert!(s.data()[0] > s.data()[1]);
    assert!(s.data()[1] > 0.0);

    // Truncation error should be positive (we discarded one singular value)
    assert!(trunc_err > 0.0);

    // Verify truncation error equals the discarded singular value
    let (_, s_full, _, _) = trunc_svd(
        &backend,
        &tensor,
        1,
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    )
    .unwrap();
    let expected_err = s_full.data()[2];
    assert!(
        (trunc_err - expected_err).abs() < 1e-10,
        "trunc_err={trunc_err} vs expected={expected_err}"
    );
}

#[test]
fn test_trunc_svd_chi_max_zero_is_error() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let params = TruncSvdParams {
        chi_max: Some(0),
        target_trunc_err: None,
    };
    assert!(trunc_svd(&backend, &tensor, 1, &params).is_err());
}

#[test]
fn test_trunc_svd_chi_max_no_truncation() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // chi_max >= k=2 means no truncation
    let params = TruncSvdParams {
        chi_max: Some(5),
        target_trunc_err: None,
    };
    let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);
    assert_eq!(trunc_err, 0.0);
}

#[test]
fn test_trunc_svd_f32() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let params = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

    assert_eq!(u.shape(), &[2, 1]);
    assert_eq!(s.shape(), &[1]);
    assert_eq!(vt.shape(), &[1, 3]);
    assert!(trunc_err > 0.0);
}

// --- QR tests ---

#[test]
fn test_qr_f64_2d() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let (q, r) = qr(&backend, &tensor, 1).unwrap();
    let q = to_rm(&q);
    let r = to_rm(&r);
    let tensor = to_rm(&tensor);

    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    // Reconstruct: A ≈ Q * R
    for i in 0..2 {
        for j in 0..2 {
            let mut val = 0.0;
            for k in 0..2 {
                val += q.get(&[i, k]) * r.get(&[k, j]);
            }
            assert!(
                (val - tensor.get(&[i, j])).abs() < 1e-10,
                "QR reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_qr_f64_rectangular() {
    let backend = NativeBackend::new();
    // shape [3, 2], nrow=1 → 3×2 matrix, k=2
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

    let (q, r) = qr(&backend, &tensor, 1).unwrap();
    let q = to_rm(&q);
    let r = to_rm(&r);
    let tensor = to_rm(&tensor);

    let (m, n, k) = (3, 2, 2);
    assert_eq!(q.shape(), &[m, k]);
    assert_eq!(r.shape(), &[k, n]);

    // Reconstruct
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += q.get(&[i, l]) * r.get(&[l, j]);
            }
            assert!(
                (val - tensor.get(&[i, j])).abs() < 1e-10,
                "QR reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_qr_f64_orthogonality() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

    let (q, _r) = qr(&backend, &tensor, 1).unwrap();
    let q = to_rm(&q);

    let (m, k) = (3, 2);
    // Q^T * Q should be identity (k×k)
    for i in 0..k {
        for j in 0..k {
            let mut dot = 0.0;
            for l in 0..m {
                dot += q.get(&[l, i]) * q.get(&[l, j]);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-10,
                "Q orthogonality failed: Q^T*Q[{i},{j}] = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_qr_f64_higher_rank() {
    let backend = NativeBackend::new();
    // shape [2, 3, 4], nrow=2 → m=6, n=4, k=4
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let tensor = cm(data, vec![2, 3, 4]);

    let (q, r) = qr(&backend, &tensor, 2).unwrap();
    let q = to_rm(&q);
    let r = to_rm(&r);
    let tensor_rm = to_rm(&tensor);

    let (m, n, k) = (6, 4, 4);
    assert_eq!(q.shape(), &[m, k]);
    assert_eq!(r.shape(), &[k, n]);

    // Reconstruct and verify
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += q.get(&[i, l]) * r.get(&[l, j]);
            }
            let orig = tensor_rm.data()[i * n + j];
            assert!(
                (val - orig).abs() < 1e-9,
                "QR reconstruction mismatch at ({i},{j}): {val} vs {orig}"
            );
        }
    }
}

#[test]
fn test_qr_f32() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);

    let (q, r) = qr(&backend, &tensor, 1).unwrap();
    let q = to_rm(&q);
    let r = to_rm(&r);
    let tensor = to_rm(&tensor);

    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    for i in 0..2 {
        for j in 0..2 {
            let mut val = 0.0f32;
            for k in 0..2 {
                val += q.get(&[i, k]) * r.get(&[k, j]);
            }
            assert!(
                (val - tensor.get(&[i, j])).abs() < 1e-4,
                "QR reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_qr_invalid_nrow() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    assert!(qr(&backend, &tensor, 0).is_err());
    assert!(qr(&backend, &tensor, 2).is_err());
}

// --- LQ tests ---

#[test]
fn test_lq_f64_2d() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);

    let (l, q) = lq(&backend, &tensor, 1).unwrap();
    let l = to_rm(&l);
    let q = to_rm(&q);
    let tensor = to_rm(&tensor);

    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(q.shape(), &[2, 2]);

    // Reconstruct: A ≈ L * Q
    for i in 0..2 {
        for j in 0..2 {
            let mut val = 0.0;
            for k in 0..2 {
                val += l.get(&[i, k]) * q.get(&[k, j]);
            }
            assert!(
                (val - tensor.get(&[i, j])).abs() < 1e-10,
                "LQ reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_lq_f64_rectangular() {
    let backend = NativeBackend::new();
    // shape [2, 3], nrow=1 → 2×3 matrix, k=2
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let (l, q) = lq(&backend, &tensor, 1).unwrap();
    let l = to_rm(&l);
    let q = to_rm(&q);
    let tensor = to_rm(&tensor);

    let (m, n, k) = (2, 3, 2);
    assert_eq!(l.shape(), &[m, k]);
    assert_eq!(q.shape(), &[k, n]);

    // Reconstruct
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for ll in 0..k {
                val += l.get(&[i, ll]) * q.get(&[ll, j]);
            }
            assert!(
                (val - tensor.get(&[i, j])).abs() < 1e-10,
                "LQ reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_lq_f64_orthogonality() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let (_l, q) = lq(&backend, &tensor, 1).unwrap();
    let q = to_rm(&q);

    let (k, n) = (2, 3);
    // Q * Q^T should be identity (k×k)
    for i in 0..k {
        for j in 0..k {
            let mut dot = 0.0;
            for l in 0..n {
                dot += q.get(&[i, l]) * q.get(&[j, l]);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-10,
                "Q orthogonality failed: Q*Q^T[{i},{j}] = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_lq_f64_higher_rank() {
    let backend = NativeBackend::new();
    // shape [2, 3, 4], nrow=1 → m=2, n=12, k=2
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let tensor = cm(data, vec![2, 3, 4]);

    let (l, q) = lq(&backend, &tensor, 1).unwrap();
    let l = to_rm(&l);
    let q = to_rm(&q);
    let tensor_rm = to_rm(&tensor);

    let (m, n, k) = (2, 12, 2);
    assert_eq!(l.shape(), &[m, k]);
    assert_eq!(q.shape(), &[k, n]);

    // Reconstruct and verify
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for ll in 0..k {
                val += l.get(&[i, ll]) * q.get(&[ll, j]);
            }
            let orig = tensor_rm.data()[i * n + j];
            assert!(
                (val - orig).abs() < 1e-9,
                "LQ reconstruction mismatch at ({i},{j}): {val} vs {orig}"
            );
        }
    }
}

#[test]
fn test_lq_f32() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);

    let (l, q) = lq(&backend, &tensor, 1).unwrap();
    let l = to_rm(&l);
    let q = to_rm(&q);
    let tensor = to_rm(&tensor);

    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(q.shape(), &[2, 2]);

    for i in 0..2 {
        for j in 0..2 {
            let mut val = 0.0f32;
            for k in 0..2 {
                val += l.get(&[i, k]) * q.get(&[k, j]);
            }
            assert!(
                (val - tensor.get(&[i, j])).abs() < 1e-4,
                "LQ reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_lq_invalid_nrow() {
    let backend = NativeBackend::new();
    let tensor = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    assert!(lq(&backend, &tensor, 0).is_err());
    assert!(lq(&backend, &tensor, 2).is_err());
}

// --- Mutation testing: trunc_svd arithmetic and boundary conditions ---

#[test]
fn test_trunc_svd_trunc_err_equals_frobenius_of_discarded() {
    // Verify trunc_err = sqrt(sum of discarded sigma_i^2), not just last sv
    let backend = NativeBackend::new();
    // Construct a diagonal matrix with known singular values [5, 4, 3, 2, 1]
    // so SVD returns them exactly.
    let mut data = vec![0.0_f64; 5 * 5];
    let svs = [5.0, 4.0, 3.0, 2.0, 1.0];
    for (i, &sv) in svs.iter().enumerate() {
        data[i * 5 + i] = sv;
    }
    let tensor = cm(data, vec![5, 5]);

    // Keep 3, discard [2, 1]
    let params = TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: None,
    };
    let (_, s, _, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

    assert_eq!(s.len(), 3);
    // trunc_err = sqrt(2^2 + 1^2) = sqrt(5)
    let expected = (4.0 + 1.0f64).sqrt();
    assert!(
        (trunc_err - expected).abs() < 1e-10,
        "trunc_err={trunc_err}, expected={expected}"
    );
}

#[test]
fn test_trunc_svd_target_trunc_err_exact_boundary() {
    // Test exact boundary: target_trunc_err == norm of smallest sv
    // With target=1.0, discarding sv=1.0 gives norm=1.0, which is NOT > 1.0
    // so it should be discarded.
    let backend = NativeBackend::new();
    let mut data = vec![0.0_f64; 3 * 3];
    // Singular values: [10, 5, 1]
    data[0] = 10.0;
    data[4] = 5.0;
    data[8] = 1.0;
    let tensor = cm(data, vec![3, 3]);

    // target=1.0: discarding sv=1 gives norm²=1.0 == target²=1.0, NOT > target²
    // so sv=1 is discarded. chi_err becomes 2.
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(1.0),
    };
    let (_, s, _, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

    assert_eq!(s.len(), 2, "sv=1.0 should be discarded when target=1.0");
    assert!((trunc_err - 1.0).abs() < 1e-10);
}

#[test]
fn test_trunc_svd_target_trunc_err_strict_boundary() {
    // target_trunc_err slightly below sv norm: should NOT discard
    let backend = NativeBackend::new();
    let mut data = vec![0.0_f64; 3 * 3];
    data[0] = 10.0;
    data[4] = 5.0;
    data[8] = 1.0;
    let tensor = cm(data, vec![3, 3]);

    // target = 0.999: discarding sv=1 gives norm²=1.0 > target²=0.998001
    // so sv=1 is NOT discarded.
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(0.999),
    };
    let (_, s, _, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

    assert_eq!(
        s.len(),
        3,
        "sv=1.0 should NOT be discarded when target=0.999"
    );
    assert_eq!(trunc_err, 0.0);
}

#[test]
fn test_trunc_svd_target_err_accumulates_multiple_svs() {
    // Verify error accumulation works across multiple discarded svs
    let backend = NativeBackend::new();
    let mut data = vec![0.0_f64; 4 * 4];
    // Singular values: [10, 3, 2, 1]
    data[0] = 10.0;
    data[5] = 3.0;
    data[10] = 2.0;
    data[15] = 1.0;
    let tensor = cm(data, vec![4, 4]);

    // target = sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.236
    // Discarding sv=1: norm²=1 <= 5 → discard
    // Discarding sv=2: norm²=1+4=5 <= 5 → discard
    // Discarding sv=3: norm²=1+4+9=14 > 5 → stop
    // chi_err = 1 (index of sv=3 after discarding 1 and 2)
    let target = 5.0f64.sqrt();
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(target),
    };
    let (_, s, _, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

    assert_eq!(s.len(), 2, "should keep svs [10, 3], discard [2, 1]");
    let expected_err = (4.0 + 1.0f64).sqrt(); // sqrt(2^2 + 1^2)
    assert!(
        (trunc_err - expected_err).abs() < 1e-10,
        "trunc_err={trunc_err}, expected={expected_err}"
    );
}

#[test]
fn test_trunc_svd_target_err_keeps_at_least_one() {
    // Even with a huge target_trunc_err, at least one sv is kept
    let backend = NativeBackend::new();
    let mut data = vec![0.0_f64; 3 * 3];
    data[0] = 3.0;
    data[4] = 2.0;
    data[8] = 1.0;
    let tensor = cm(data, vec![3, 3]);

    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(1e10),
    };
    let (_, s, _, _) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

    assert!(!s.is_empty(), "must keep at least one singular value");
}

#[test]
fn test_trunc_svd_slicing_u_times_s_times_vt_approximation() {
    // Verify truncated U*S*Vt approximates original within trunc_err,
    // using a non-trivial non-diagonal matrix
    let backend = NativeBackend::new();
    #[rustfmt::skip]
    let data = vec![
        1.0_f64, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let tensor = cm(data, vec![4, 3]);

    for chi in 1..=2 {
        let params = TruncSvdParams {
            chi_max: Some(chi),
            target_trunc_err: None,
        };
        let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
        let u = to_rm(&u);
        let vt = to_rm(&vt);
        let tensor_rm = to_rm(&tensor);
        let (m, n) = (4, 3);

        // Compute ||A - U*S*Vt||_F
        let mut err_sq = 0.0;
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..chi {
                    val += u.get(&[i, l]) * s.data()[l] * vt.get(&[l, j]);
                }
                let diff = val - tensor_rm.data()[i * n + j];
                err_sq += diff * diff;
            }
        }
        let recon_err = err_sq.sqrt();

        assert!(
            (recon_err - trunc_err).abs() < 1e-9,
            "chi={chi}: recon_err={recon_err}, trunc_err={trunc_err}"
        );
    }
}

#[test]
fn test_trunc_svd_chi_max_and_target_err_stricter_wins() {
    // When both params are set, the one that produces smaller chi wins
    let backend = NativeBackend::new();
    let mut data = vec![0.0; 4 * 4];
    data[0] = 10.0;
    data[5] = 5.0;
    data[10] = 2.0;
    data[15] = 1.0;
    let tensor = cm(data, vec![4, 4]);

    // chi_max=3 but target_trunc_err allows discarding sv=1 and sv=2
    // target = sqrt(5) ≈ 2.236 → can discard [1, 2], chi_err=2
    // Result: min(3, 2) = 2
    let params = TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: Some(5.0f64.sqrt()),
    };
    let (_, s, _, _) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
    assert_eq!(s.len(), 2);

    // chi_max=1, target_trunc_err=0 (keeps all)
    // Result: min(1, 4) = 1
    let params = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: Some(0.0),
    };
    let (_, s, _, _) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
    assert_eq!(s.len(), 1);
}
