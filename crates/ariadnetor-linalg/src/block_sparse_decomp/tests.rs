use arnet_native::NativeBackend;
use arnet_tensor::block_sparse::{BlockCoord, BlockSparse, Direction, QNIndex};
use arnet_tensor::sector::{U1Sector, Z2Sector};

use super::*;

fn backend() -> NativeBackend {
    NativeBackend
}

// -- Test tensor constructors ------------------------------------------------

/// Rank-2 U1, identity flux, blocks (0,0): 2×2 and (1,1): 3×3.
fn sample_u1_rank2() -> BlockSparse<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut bs = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));
    let d = bs.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    d.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let d = bs.block_data_mut(&BlockCoord(vec![1, 1])).unwrap();
    d.copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    bs
}

/// Rank-3 U1, identity flux. Fused left sector 1 merges tuples (0,1) and (1,0).
fn sample_u1_rank3() -> BlockSparse<f64, U1Sector> {
    let leg0 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let leg1 = QNIndex::new(vec![(U1Sector(0), 3), (U1Sector(1), 2)], Direction::Out);
    let leg2 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut bs = BlockSparse::<f64, U1Sector>::zeros(vec![leg0, leg1, leg2], U1Sector(0));
    for (i, v) in bs
        .block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *v = (i + 1) as f64;
    }
    for (i, v) in bs
        .block_data_mut(&BlockCoord(vec![0, 1, 1]))
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *v = (i + 13) as f64;
    }
    for (i, v) in bs
        .block_data_mut(&BlockCoord(vec![1, 0, 1]))
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *v = (i + 25) as f64;
    }
    bs
}

/// Rank-2 U1, flux=1. Single allowed block (1,0): 3×4.
fn sample_u1_nonzero_flux() -> BlockSparse<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 4)], Direction::In);
    let mut bs = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(1));
    for (i, v) in bs
        .block_data_mut(&BlockCoord(vec![1, 0]))
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *v = (i + 1) as f64;
    }
    bs
}

// -- Dense helpers for verification ------------------------------------------

fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn diag_scale_vt(s: &[f64], vt: &[f64], k: usize, n: usize) -> Vec<f64> {
    let mut result = vt.to_vec();
    for i in 0..k {
        for j in 0..n {
            result[i * n + j] *= s[i];
        }
    }
    result
}

fn assert_approx_eq(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "index {i}: {x} vs {y} (diff={})",
            (x - y).abs()
        );
    }
}

/// Per-sector SVD reconstruction check: U * diag(S) * Vt ≈ original.
fn verify_svd_reconstruction<S: Sector + PartialEq>(
    tensor: &BlockSparse<f64, S>,
    u: &BlockSparse<f64, S>,
    sv: &BlockSingularValues<f64, S>,
    vt: &BlockSparse<f64, S>,
    nrow: usize,
) {
    let groups = compute_fused_sector_groups(tensor, nrow);
    for group in &groups {
        let original = assemble_sector_matrix(tensor, group);
        let s_data: &[f64] = sv
            .values
            .iter()
            .find(|(s, _)| *s == group.sector)
            .map(|(_, v)| v.as_slice())
            .unwrap();
        let k_s = s_data.len();
        let u_groups = compute_fused_sector_groups(u, nrow);
        let u_g = u_groups.iter().find(|g| g.sector == group.sector).unwrap();
        let u_mat = assemble_sector_matrix(u, u_g);
        let vt_groups = compute_fused_sector_groups(vt, 1);
        let vt_g = vt_groups.iter().find(|g| g.sector == group.sector).unwrap();
        let vt_mat = assemble_sector_matrix(vt, vt_g);
        let sv_vt = diag_scale_vt(s_data, &vt_mat, k_s, vt_g.n);
        let reconstructed = matmul(&u_mat, u_g.m, k_s, &sv_vt, vt_g.n);
        assert_approx_eq(&reconstructed, &original, 1e-10);
    }
}

/// Per-sector two-factor reconstruction: left * right ≈ original.
fn verify_two_factor_reconstruction<S: Sector + PartialEq>(
    tensor: &BlockSparse<f64, S>,
    left: &BlockSparse<f64, S>,
    right: &BlockSparse<f64, S>,
    nrow: usize,
) {
    let groups = compute_fused_sector_groups(tensor, nrow);
    for group in &groups {
        let original = assemble_sector_matrix(tensor, group);
        let k = group.m.min(group.n);
        let l_groups = compute_fused_sector_groups(left, nrow);
        let l_g = l_groups.iter().find(|g| g.sector == group.sector).unwrap();
        let l_mat = assemble_sector_matrix(left, l_g);
        let r_groups = compute_fused_sector_groups(right, 1);
        let r_g = r_groups.iter().find(|g| g.sector == group.sector).unwrap();
        let r_mat = assemble_sector_matrix(right, r_g);
        let reconstructed = matmul(&l_mat, l_g.m, k, &r_mat, r_g.n);
        assert_approx_eq(&reconstructed, &original, 1e-10);
    }
}

// -- Validation tests --------------------------------------------------------

#[test]
fn nrow_zero_rejected() {
    let bs = sample_u1_rank2();
    let err = svd_block_sparse(&backend(), &bs, 0)
        .err()
        .expect("expected error");
    assert!(err.to_string().contains("nrow"));
}

#[test]
fn nrow_ge_rank_rejected() {
    let err = svd_block_sparse(&backend(), &sample_u1_rank2(), 2)
        .err()
        .expect("expected error");
    assert!(err.to_string().contains("nrow"));
}

#[test]
fn trunc_svd_chi_max_zero_rejected() {
    let params = TruncSvdParams {
        chi_max: Some(0),
        target_trunc_err: None,
    };
    let err = trunc_svd_block_sparse(&backend(), &sample_u1_rank2(), 1, &params)
        .err()
        .expect("expected error");
    assert!(err.to_string().contains("chi_max"));
}

// -- SVD tests ---------------------------------------------------------------

#[test]
fn svd_rank2_reconstruction() {
    let bs = sample_u1_rank2();
    let (u, sv, vt) = svd_block_sparse(&backend(), &bs, 1).unwrap();
    // Structure checks
    assert_eq!(u.rank(), 2);
    assert_eq!(*u.flux(), U1Sector(0));
    assert_eq!(vt.rank(), 2);
    assert_eq!(*vt.flux(), U1Sector(0));
    assert_eq!(sv.values.len(), 2);
    assert_eq!(sv.values[0].1.len(), 2); // sector 0: min(2,2)
    assert_eq!(sv.values[1].1.len(), 3); // sector 1: min(3,3)
    for (_, vals) in &sv.values {
        for w in vals.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }
    verify_svd_reconstruction(&bs, &u, &sv, &vt, 1);
}

#[test]
fn svd_rank3_fused_sectors() {
    let bs = sample_u1_rank3();
    let (u, sv, vt) = svd_block_sparse(&backend(), &bs, 2).unwrap();
    assert_eq!(u.rank(), 3);
    assert_eq!(*u.flux(), U1Sector(0));
    assert_eq!(vt.rank(), 2);
    assert_eq!(sv.values.len(), 2);
    assert_eq!(sv.values[0].1.len(), 2); // sector 0: m=6, n=2 → k=2
    assert_eq!(sv.values[1].1.len(), 3); // sector 1: m=7, n=3 → k=3
    verify_svd_reconstruction(&bs, &u, &sv, &vt, 2);
}

#[test]
fn svd_nonzero_flux() {
    let bs = sample_u1_nonzero_flux();
    let (u, sv, vt) = svd_block_sparse(&backend(), &bs, 1).unwrap();
    assert_eq!(*u.flux(), U1Sector(0));
    assert_eq!(*vt.flux(), U1Sector(1));
    assert_eq!(sv.values.len(), 1);
    assert_eq!(sv.values[0].0, U1Sector(1));
    assert_eq!(sv.values[0].1.len(), 3); // 3×4 → k=3
    verify_svd_reconstruction(&bs, &u, &sv, &vt, 1);
}

// -- Truncated SVD tests -----------------------------------------------------

#[test]
fn trunc_svd_chi_max() {
    let bs = sample_u1_rank2();
    let params = TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: None,
    };
    let (u, sv, vt, trunc_err) = trunc_svd_block_sparse(&backend(), &bs, 1, &params).unwrap();
    let total_kept: usize = sv.values.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(total_kept, 3);
    assert!(trunc_err > 0.0);
    assert_eq!(*u.flux(), U1Sector(0));
    assert_eq!(*vt.flux(), U1Sector(0));
}

#[test]
fn trunc_svd_no_truncation() {
    let params = TruncSvdParams {
        chi_max: Some(100),
        target_trunc_err: None,
    };
    let (_, sv, _, trunc_err) =
        trunc_svd_block_sparse(&backend(), &sample_u1_rank2(), 1, &params).unwrap();
    let total_kept: usize = sv.values.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(total_kept, 5);
    assert!(trunc_err.abs() < 1e-15);
}

#[test]
fn trunc_svd_target_err() {
    let bs = sample_u1_rank2();
    let params_full = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let (_, sv_full, _, _) = trunc_svd_block_sparse(&backend(), &bs, 1, &params_full).unwrap();
    let mut all_sv: Vec<f64> = sv_full
        .values
        .iter()
        .flat_map(|(_, v)| v.iter().copied())
        .collect();
    all_sv.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let smallest = *all_sv.last().unwrap();

    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(smallest + 0.01),
    };
    let (_, sv_trunc, _, trunc_err) = trunc_svd_block_sparse(&backend(), &bs, 1, &params).unwrap();
    let total_kept: usize = sv_trunc.values.iter().map(|(_, v)| v.len()).sum();
    assert!(
        total_kept < 5,
        "expected truncation, got {total_kept} values"
    );
    assert!(trunc_err <= smallest + 0.01 + 1e-10);
}

// -- QR tests ----------------------------------------------------------------

#[test]
fn qr_rank2_reconstruction() {
    let bs = sample_u1_rank2();
    let (q, r) = qr_block_sparse(&backend(), &bs, 1).unwrap();
    assert_eq!(q.rank(), 2);
    assert_eq!(*q.flux(), U1Sector(0));
    assert_eq!(r.rank(), 2);
    assert_eq!(*r.flux(), U1Sector(0));
    verify_two_factor_reconstruction(&bs, &q, &r, 1);
}

#[test]
fn qr_orthogonality() {
    let (q, _) = qr_block_sparse(&backend(), &sample_u1_rank2(), 1).unwrap();
    let q_groups = compute_fused_sector_groups(&q, 1);
    for g in &q_groups {
        let q_mat = assemble_sector_matrix(&q, g);
        let (m, k) = (g.m, g.n);
        let mut qtq = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                for p in 0..m {
                    qtq[i * k + j] += q_mat[p * k + i] * q_mat[p * k + j];
                }
            }
        }
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[i * k + j] - expected).abs() < 1e-10);
            }
        }
    }
}

// -- LQ tests ----------------------------------------------------------------

#[test]
fn lq_rank2_reconstruction() {
    let bs = sample_u1_rank2();
    let (l, q) = lq_block_sparse(&backend(), &bs, 1).unwrap();
    assert_eq!(l.rank(), 2);
    assert_eq!(*l.flux(), U1Sector(0));
    assert_eq!(q.rank(), 2);
    assert_eq!(*q.flux(), U1Sector(0));
    verify_two_factor_reconstruction(&bs, &l, &q, 1);
}

#[test]
fn lq_orthogonality() {
    let (_, q) = lq_block_sparse(&backend(), &sample_u1_rank2(), 1).unwrap();
    let q_groups = compute_fused_sector_groups(&q, 1);
    for g in &q_groups {
        let q_mat = assemble_sector_matrix(&q, g);
        let (k, n) = (g.m, g.n);
        let mut qqt = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                for p in 0..n {
                    qqt[i * k + j] += q_mat[i * n + p] * q_mat[j * n + p];
                }
            }
        }
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qqt[i * k + j] - expected).abs() < 1e-10);
            }
        }
    }
}

// -- Z2 symmetry test --------------------------------------------------------

#[test]
fn svd_z2_reconstruction() {
    let row = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)],
        Direction::Out,
    );
    let col = QNIndex::new(
        vec![(Z2Sector::new(0), 4), (Z2Sector::new(1), 5)],
        Direction::In,
    );
    let mut bs = BlockSparse::<f64, Z2Sector>::zeros(vec![row, col], Z2Sector::new(0));
    for (i, v) in bs
        .block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *v = (i + 1) as f64;
    }
    for (i, v) in bs
        .block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *v = (i + 9) as f64;
    }
    let (u, sv, vt) = svd_block_sparse(&backend(), &bs, 1).unwrap();
    assert_eq!(*u.flux(), Z2Sector::new(0));
    assert_eq!(*vt.flux(), Z2Sector::new(0));
    assert_eq!(sv.values.len(), 2);
    verify_svd_reconstruction(&bs, &u, &sv, &vt, 1);
}

// -- Empty tensor test -------------------------------------------------------

#[test]
fn svd_empty_tensor() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let bs = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(1));
    let (u, sv, vt) = svd_block_sparse(&backend(), &bs, 1).unwrap();
    assert_eq!(sv.values.len(), 0);
    assert_eq!(u.num_blocks(), 0);
    assert_eq!(vt.num_blocks(), 0);
}

#[test]
fn trunc_svd_empty_tensor_with_target_err() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let bs = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(1));
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(0.1),
    };
    let (u, sv, vt, trunc_err) = trunc_svd_block_sparse(&backend(), &bs, 1, &params).unwrap();
    assert_eq!(sv.values.len(), 0);
    assert_eq!(u.num_blocks(), 0);
    assert_eq!(vt.num_blocks(), 0);
    assert!(trunc_err.abs() < 1e-15);
}
