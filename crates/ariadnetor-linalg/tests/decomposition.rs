use arnet_native::NativeBackend;
use arnet_linalg::{lq, qr, svd, trunc_svd, TruncSvdParams};
use arnet_tensor::DenseTensor;

// --- SVD tests ---

#[test]
fn test_svd_f64_2d() {
    let backend = NativeBackend::new();
    // A = [[1, 2], [3, 4]] shape [2, 2], nrow=1 → 2×2 matrix
    let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();

    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);

    // Singular values should be positive and descending
    assert!(s.get(&[0]) > s.get(&[1]));
    assert!(s.get(&[1]) >= 0.0);

    // Reconstruct: A ≈ U * diag(S) * Vt
    for i in 0..2 {
        for j in 0..2 {
            let mut val = 0.0;
            for k in 0..2 {
                val += u.get(&[i, k]) * s.get(&[k]) * vt.get(&[k, j]);
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
    let tensor = DenseTensor::<f64>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );

    let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();

    let (m, n, k) = (2, 3, 2);
    assert_eq!(u.shape(), &[m, k]);
    assert_eq!(s.shape(), &[k]);
    assert_eq!(vt.shape(), &[k, n]);

    // Reconstruct
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += u.get(&[i, l]) * s.get(&[l]) * vt.get(&[l, j]);
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
    let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

    let (u, s, vt) = svd(&backend, &tensor, 2).unwrap();

    let (m, n, k) = (6, 4, 4);
    assert_eq!(u.shape(), &[m, k]);
    assert_eq!(s.shape(), &[k]);
    assert_eq!(vt.shape(), &[k, n]);

    // Reconstruct and verify against original flat data
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += u.get(&[i, l]) * s.get(&[l]) * vt.get(&[l, j]);
            }
            let orig = tensor.data()[i * n + j];
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
    let tensor = DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();

    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);

    for i in 0..2 {
        for j in 0..2 {
            let mut val = 0.0f32;
            for k in 0..2 {
                val += u.get(&[i, k]) * s.get(&[k]) * vt.get(&[k, j]);
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
    let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

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
    let tensor = DenseTensor::<f64>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
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
    assert!(s.get(&[0]) > s.get(&[1]));
    assert!(s.get(&[1]) > 0.0);

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
    let expected_err = s_full.get(&[2]);
    assert!(
        (trunc_err - expected_err).abs() < 1e-10,
        "trunc_err={trunc_err} vs expected={expected_err}"
    );
}

#[test]
fn test_trunc_svd_chi_max_zero_is_error() {
    let backend = NativeBackend::new();
    let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let params = TruncSvdParams {
        chi_max: Some(0),
        target_trunc_err: None,
    };
    assert!(trunc_svd(&backend, &tensor, 1, &params).is_err());
}

#[test]
fn test_trunc_svd_chi_max_no_truncation() {
    let backend = NativeBackend::new();
    let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

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
fn test_trunc_svd_target_trunc_err() {
    let backend = NativeBackend::new();
    // 4×4 matrix
    let data: Vec<f64> = (1..=16).map(|i| i as f64).collect();
    let tensor = DenseTensor::from_data(data, vec![4, 4]);

    // Full SVD first to know the singular values
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

    // Set threshold just above the smallest singular value
    let smallest_sv = s_full.get(&[s_full.len() - 1]);
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(smallest_sv + 1e-10),
    };
    let (_u, s, _vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

    // Should have discarded the smallest singular value
    assert!(s.len() < s_full.len());
    // Truncation error should be approximately equal to the discarded singular value
    assert!(trunc_err <= smallest_sv + 1e-10);
}

#[test]
fn test_trunc_svd_both_params() {
    let backend = NativeBackend::new();
    // 4×4 matrix
    let data: Vec<f64> = (1..=16).map(|i| i as f64).collect();
    let tensor = DenseTensor::from_data(data, vec![4, 4]);

    // Full SVD to get singular values
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
    let k_full = s_full.len();

    // chi_max is the binding constraint: target_trunc_err=0 forces keeping all,
    // but chi_max limits to 2
    let params = TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: Some(0.0),
    };
    let (_, s, _, _) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
    assert_eq!(s.len(), 2);

    // target_trunc_err is the binding constraint: chi_max allows all,
    // but large target_trunc_err allows aggressive truncation
    let params = TruncSvdParams {
        chi_max: Some(k_full),
        target_trunc_err: Some(1e10),
    };
    let (_, s, _, _) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
    // Large allowed error → aggressive truncation → minimum 1 value kept
    assert_eq!(s.len(), 1);

    // Neither constraint truncates: chi_max=k_full, target_trunc_err=0
    let params = TruncSvdParams {
        chi_max: Some(k_full),
        target_trunc_err: Some(0.0),
    };
    let (_, s, _, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
    assert_eq!(s.len(), k_full);
    assert_eq!(trunc_err, 0.0);
}

#[test]
fn test_trunc_svd_f32() {
    let backend = NativeBackend::new();
    let tensor = DenseTensor::<f32>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );

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

#[test]
fn test_trunc_svd_reconstruction() {
    let backend = NativeBackend::new();
    // Verify that truncated reconstruction is a valid low-rank approximation
    let tensor = DenseTensor::<f64>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
    );

    let params = TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    };
    let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

    let (m, n, chi) = (3, 4, 2);

    // Reconstruct: A_approx = U * diag(S) * Vt
    let mut recon_err_sq = 0.0;
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..chi {
                val += u.get(&[i, l]) * s.get(&[l]) * vt.get(&[l, j]);
            }
            let diff = val - tensor.data()[i * n + j];
            recon_err_sq += diff * diff;
        }
    }
    let recon_err = recon_err_sq.sqrt();

    // Reconstruction error should equal the truncation error
    // (Eckart-Young theorem: ||A - A_k||_F = sqrt(sum of discarded σ²))
    assert!(
        (recon_err - trunc_err).abs() < 1e-10,
        "recon_err={recon_err} vs trunc_err={trunc_err}"
    );
}

// --- QR tests ---

#[test]
fn test_qr_f64_2d() {
    let backend = NativeBackend::new();
    let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let (q, r) = qr(&backend, &tensor, 1).unwrap();

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
    let tensor = DenseTensor::<f64>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3, 2],
    );

    let (q, r) = qr(&backend, &tensor, 1).unwrap();

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
    let tensor = DenseTensor::<f64>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3, 2],
    );

    let (q, _r) = qr(&backend, &tensor, 1).unwrap();

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
    let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

    let (q, r) = qr(&backend, &tensor, 2).unwrap();

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
            let orig = tensor.data()[i * n + j];
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
    let tensor = DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let (q, r) = qr(&backend, &tensor, 1).unwrap();

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
    let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    assert!(qr(&backend, &tensor, 0).is_err());
    assert!(qr(&backend, &tensor, 2).is_err());
}

// --- LQ tests ---

#[test]
fn test_lq_f64_2d() {
    let backend = NativeBackend::new();
    let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let (l, q) = lq(&backend, &tensor, 1).unwrap();

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
    let tensor = DenseTensor::<f64>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );

    let (l, q) = lq(&backend, &tensor, 1).unwrap();

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
    let tensor = DenseTensor::<f64>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );

    let (_l, q) = lq(&backend, &tensor, 1).unwrap();

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
    let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

    let (l, q) = lq(&backend, &tensor, 1).unwrap();

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
            let orig = tensor.data()[i * n + j];
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
    let tensor = DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let (l, q) = lq(&backend, &tensor, 1).unwrap();

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
    let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    assert!(lq(&backend, &tensor, 0).is_err());
    assert!(lq(&backend, &tensor, 2).is_err());
}
