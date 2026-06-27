use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_native::NativeBackend;
use arnet_tensor::test_fixtures::{legs, out_in_legs, square_legs};
use arnet_tensor::{BlockCoord, BlockSparseTensorData, Direction};
use arnet_tensor::{U1Sector, Z2Sector};

use super::*;

fn backend() -> NativeBackend {
    NativeBackend::new()
}

fn order() -> MemoryOrder {
    backend().preferred_order()
}
// -- Test tensor constructors ------------------------------------------------

/// Rank-2 U1, identity flux, blocks (0,0): 2×2 and (1,1): 3×3.
fn sample_u1_rank2() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 3)]),
        U1Sector(0),
        order(),
    );
    let d = bs.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    d.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let d = bs.block_data_mut(&BlockCoord(vec![1, 1])).unwrap();
    d.copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    bs
}

/// Rank-3 U1, identity flux. Fused left sector 1 merges tuples (0,1) and (1,0).
fn sample_u1_rank3() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 3), (U1Sector(1), 2)], Direction::Out),
            (vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In),
        ]),
        U1Sector(0),
        order(),
    );
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
fn sample_u1_nonzero_flux() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        out_in_legs(
            vec![(U1Sector(0), 2), (U1Sector(1), 3)],
            vec![(U1Sector(0), 4)],
        ),
        U1Sector(1),
        order(),
    );
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

fn mat_idx(row: usize, col: usize, rows: usize, cols: usize, order: MemoryOrder) -> usize {
    match order {
        MemoryOrder::RowMajor => row * cols + col,
        MemoryOrder::ColumnMajor => col * rows + row,
    }
}

fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize, order: MemoryOrder) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[mat_idx(i, p, m, k, order)] * b[mat_idx(p, j, k, n, order)];
            }
            c[mat_idx(i, j, m, n, order)] = sum;
        }
    }
    c
}

fn diag_scale_vt(s: &[f64], vt: &[f64], k: usize, n: usize, order: MemoryOrder) -> Vec<f64> {
    let mut result = vt.to_vec();
    for (i, &si) in s.iter().enumerate().take(k) {
        for j in 0..n {
            let idx = mat_idx(i, j, k, n, order);
            result[idx] *= si;
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
    tensor: &BlockSparseTensorData<f64, S>,
    u: &BlockSparseTensorData<f64, S>,
    sv: &BlockScalars<f64, S>,
    vt: &BlockSparseTensorData<f64, S>,
    nrow: usize,
    order: MemoryOrder,
) {
    let groups = compute_fused_sector_groups(tensor, nrow);
    for group in &groups {
        let original = assemble_sector_matrix(tensor, group, order);
        let s_data: &[f64] = sv
            .values
            .iter()
            .find(|(s, _)| *s == group.sector)
            .map(|(_, v)| v.as_slice())
            .unwrap();
        let k_s = s_data.len();
        let u_groups = compute_fused_sector_groups(u, nrow);
        let u_g = u_groups.iter().find(|g| g.sector == group.sector).unwrap();
        let u_mat = assemble_sector_matrix(u, u_g, order);
        let vt_groups = compute_fused_sector_groups(vt, 1);
        let vt_g = vt_groups.iter().find(|g| g.sector == group.sector).unwrap();
        let vt_mat = assemble_sector_matrix(vt, vt_g, order);
        let sv_vt = diag_scale_vt(s_data, &vt_mat, k_s, vt_g.n, order);
        let reconstructed = matmul(&u_mat, u_g.m, k_s, &sv_vt, vt_g.n, order);
        assert_approx_eq(&reconstructed, &original, 1e-10);
    }
}

/// Per-sector two-factor reconstruction: left * right ≈ original.
fn verify_two_factor_reconstruction<S: Sector + PartialEq>(
    tensor: &BlockSparseTensorData<f64, S>,
    left: &BlockSparseTensorData<f64, S>,
    right: &BlockSparseTensorData<f64, S>,
    nrow: usize,
    order: MemoryOrder,
) {
    let groups = compute_fused_sector_groups(tensor, nrow);
    for group in &groups {
        let original = assemble_sector_matrix(tensor, group, order);
        let k = group.m.min(group.n);
        let l_groups = compute_fused_sector_groups(left, nrow);
        let l_g = l_groups.iter().find(|g| g.sector == group.sector).unwrap();
        let l_mat = assemble_sector_matrix(left, l_g, order);
        let r_groups = compute_fused_sector_groups(right, 1);
        let r_g = r_groups.iter().find(|g| g.sector == group.sector).unwrap();
        let r_mat = assemble_sector_matrix(right, r_g, order);
        let reconstructed = matmul(&l_mat, l_g.m, k, &r_mat, r_g.n, order);
        assert_approx_eq(&reconstructed, &original, 1e-10);
    }
}

/// Rank-2 with known SVs: sector 0 → identity [1,1], sector 1 → diag(3,2).
/// Global SVs sorted: [3, 2, 1, 1].
fn sample_known_svs() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 2)]),
        U1Sector(0),
        order(),
    );
    bs.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    bs.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[3.0, 0.0, 0.0, 2.0]);
    bs
}

/// Rank-4 tensor with nrow=2 producing multi-tuple fused sectors.
///
/// Fused sector U1(1) has left tuples [(0,1),(1,0)] and right tuples [(0,1),(1,0)],
/// giving non-trivial cumulative offsets [0, 1] on both sides.
fn sample_rank4_multi_tuple() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In),
        ]),
        U1Sector(0),
        order(),
    );
    // Distinct values so offset errors are detectable
    bs.block_data_mut(&BlockCoord(vec![0, 0, 0, 0])).unwrap()[0] = 5.0;
    bs.block_data_mut(&BlockCoord(vec![0, 1, 0, 1])).unwrap()[0] = 1.0;
    bs.block_data_mut(&BlockCoord(vec![0, 1, 1, 0])).unwrap()[0] = 2.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 0, 1])).unwrap()[0] = 3.0;
    bs.block_data_mut(&BlockCoord(vec![1, 0, 1, 0])).unwrap()[0] = 4.0;
    bs.block_data_mut(&BlockCoord(vec![1, 1, 1, 1])).unwrap()[0] = 6.0;
    bs
}

// -- Validation tests --------------------------------------------------------
#[test]
fn nrow_zero_rejected() {
    let bs = sample_u1_rank2();
    let err = svd_block_sparse_with_policy_dense(&backend(), &bs, 0, ExecPolicy::Sequential)
        .err()
        .expect("expected error");
    assert!(err.to_string().contains("nrow"));
}

#[test]
fn nrow_ge_rank_rejected() {
    let err = svd_block_sparse_with_policy_dense(
        &backend(),
        &sample_u1_rank2(),
        2,
        ExecPolicy::Sequential,
    )
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
    let err = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &sample_u1_rank2(),
        1,
        &params,
        ExecPolicy::Sequential,
    )
    .err()
    .expect("expected error");
    assert!(err.to_string().contains("chi_max"));
}

// -- SVD tests ---------------------------------------------------------------
#[test]
fn svd_rank2_reconstruction() {
    let bs = sample_u1_rank2();
    let (u, sv, vt) =
        svd_block_sparse_with_policy_dense(&backend(), &bs, 1, ExecPolicy::Sequential).unwrap();
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
    verify_svd_reconstruction(&bs, &u, &sv, &vt, 1, order());
}

#[test]
fn svd_rank3_fused_sectors() {
    let bs = sample_u1_rank3();
    let (u, sv, vt) =
        svd_block_sparse_with_policy_dense(&backend(), &bs, 2, ExecPolicy::Sequential).unwrap();
    assert_eq!(u.rank(), 3);
    assert_eq!(*u.flux(), U1Sector(0));
    assert_eq!(vt.rank(), 2);
    assert_eq!(sv.values.len(), 2);
    assert_eq!(sv.values[0].1.len(), 2); // sector 0: m=6, n=2 → k=2
    assert_eq!(sv.values[1].1.len(), 3); // sector 1: m=7, n=3 → k=3
    verify_svd_reconstruction(&bs, &u, &sv, &vt, 2, order());
}

#[test]
fn svd_nonzero_flux() {
    let bs = sample_u1_nonzero_flux();
    let (u, sv, vt) =
        svd_block_sparse_with_policy_dense(&backend(), &bs, 1, ExecPolicy::Sequential).unwrap();
    assert_eq!(*u.flux(), U1Sector(0));
    assert_eq!(*vt.flux(), U1Sector(1));
    assert_eq!(sv.values.len(), 1);
    assert_eq!(sv.values[0].0, U1Sector(1));
    assert_eq!(sv.values[0].1.len(), 3); // 3×4 → k=3
    verify_svd_reconstruction(&bs, &u, &sv, &vt, 1, order());
}

// -- Truncated SVD tests -----------------------------------------------------
#[test]
fn trunc_svd_chi_max() {
    let bs = sample_u1_rank2();
    let params = TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: None,
    };
    let (u, sv, vt, trunc_err) = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &bs,
        1,
        &params,
        ExecPolicy::Sequential,
    )
    .unwrap();
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
    let (_, sv, _, trunc_err) = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &sample_u1_rank2(),
        1,
        &params,
        ExecPolicy::Sequential,
    )
    .unwrap();
    let total_kept: usize = sv.values.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(total_kept, 5);
    assert!(trunc_err.abs() < 1e-15);
}

// -- QR tests ----------------------------------------------------------------
#[test]
fn qr_rank2_reconstruction() {
    let bs = sample_u1_rank2();
    let (q, r) =
        qr_block_sparse_with_policy_dense(&backend(), &bs, 1, ExecPolicy::Sequential).unwrap();
    assert_eq!(q.rank(), 2);
    assert_eq!(*q.flux(), U1Sector(0));
    assert_eq!(r.rank(), 2);
    assert_eq!(*r.flux(), U1Sector(0));
    verify_two_factor_reconstruction(&bs, &q, &r, 1, order());
}

#[test]
fn qr_orthogonality() {
    let ord = order();
    let (q, _) = qr_block_sparse_with_policy_dense(
        &backend(),
        &sample_u1_rank2(),
        1,
        ExecPolicy::Sequential,
    )
    .unwrap();
    let q_groups = compute_fused_sector_groups(&q, 1);
    for g in &q_groups {
        let q_mat = assemble_sector_matrix(&q, g, ord);
        let (m, k) = (g.m, g.n);
        // Q^T * Q should be identity (k × k)
        let mut qtq = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                for p in 0..m {
                    qtq[i * k + j] +=
                        q_mat[mat_idx(p, i, m, k, ord)] * q_mat[mat_idx(p, j, m, k, ord)];
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
    let (l, q) =
        lq_block_sparse_with_policy_dense(&backend(), &bs, 1, ExecPolicy::Sequential).unwrap();
    assert_eq!(l.rank(), 2);
    assert_eq!(*l.flux(), U1Sector(0));
    assert_eq!(q.rank(), 2);
    assert_eq!(*q.flux(), U1Sector(0));
    verify_two_factor_reconstruction(&bs, &l, &q, 1, order());
}

#[test]
fn lq_orthogonality() {
    let ord = order();
    let (_, q) = lq_block_sparse_with_policy_dense(
        &backend(),
        &sample_u1_rank2(),
        1,
        ExecPolicy::Sequential,
    )
    .unwrap();
    let q_groups = compute_fused_sector_groups(&q, 1);
    for g in &q_groups {
        let q_mat = assemble_sector_matrix(&q, g, ord);
        let (k, n) = (g.m, g.n);
        // Q * Q^T should be identity (k × k)
        let mut qqt = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                for p in 0..n {
                    qqt[i * k + j] +=
                        q_mat[mat_idx(i, p, k, n, ord)] * q_mat[mat_idx(j, p, k, n, ord)];
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

// -- Exact truncation error tests --------------------------------------------
#[test]
fn trunc_svd_exact_truncation_error() {
    let bs = sample_known_svs();
    // Verify SVs are exactly [3, 2, 1, 1]
    let (_, sv_full, _) =
        svd_block_sparse_with_policy_dense(&backend(), &bs, 1, ExecPolicy::Sequential).unwrap();
    let mut all_sv: Vec<f64> = sv_full
        .values
        .iter()
        .flat_map(|(_, v)| v.iter().copied())
        .collect();
    all_sv.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(all_sv.len(), 4);
    assert!((all_sv[0] - 3.0).abs() < 1e-10);
    assert!((all_sv[3] - 1.0).abs() < 1e-10);

    // chi_max=2: keep [3,2], discard [1,1]
    let params = TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    };
    let (_, _, _, trunc_err) = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &bs,
        1,
        &params,
        ExecPolicy::Sequential,
    )
    .unwrap();
    // trunc_err = sqrt(1² + 1²) = sqrt(2)
    let expected_err = 2.0_f64.sqrt();
    assert!(
        (trunc_err - expected_err).abs() < 1e-12,
        "trunc_err={trunc_err}, expected={expected_err}"
    );
}

#[test]
fn trunc_svd_target_err_exact_count() {
    // SVs = [3, 2, 1, 1]. With target_err=1.1:
    // target_sq = 1.21. Can discard one SV (1²=1 ≤ 1.21), not two (1+1=2 > 1.21).
    let bs = sample_known_svs();
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(1.1),
    };
    let (_, sv, _, trunc_err) = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &bs,
        1,
        &params,
        ExecPolicy::Sequential,
    )
    .unwrap();
    let total_kept: usize = sv.values.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(total_kept, 3, "should discard exactly 1 SV");
    assert!(
        (trunc_err - 1.0).abs() < 1e-12,
        "trunc_err={trunc_err}, expected=1.0"
    );
}

#[test]
fn trunc_svd_target_err_boundary() {
    // SVs = [3, 2, 1, 1]. With target_err=1.0 (target_sq=1.0):
    // Smallest SV² = 1.0. Since 0+1 = 1 is NOT > 1.0 (strict >), it IS discarded.
    let bs = sample_known_svs();
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(1.0),
    };
    let (_, sv, _, _) = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &bs,
        1,
        &params,
        ExecPolicy::Sequential,
    )
    .unwrap();
    let total_kept: usize = sv.values.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(
        total_kept, 3,
        "boundary: sv²==target_sq should still discard"
    );
}

#[test]
fn trunc_svd_zero_sector_bond_structure() {
    // SVs = [3, 2, 1, 1]. chi_max=1 keeps only SV=3 (sector 1).
    // Sector 0 gets k=0, should be excluded from bond.
    let bs = sample_known_svs();
    let params = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let (u, sv, vt, _) = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &bs,
        1,
        &params,
        ExecPolicy::Sequential,
    )
    .unwrap();

    // Only sector 1 survives in SVD output
    assert_eq!(sv.values.len(), 1);
    assert_eq!(sv.values[0].0, U1Sector(1));
    assert_eq!(sv.values[0].1.len(), 1);

    // Bond indices should not contain zero-dimension blocks
    let u_bond = u.indices().last().unwrap();
    for bi in 0..u_bond.num_blocks() {
        assert!(
            u_bond.block_dim(bi) > 0,
            "bond block {bi} has zero dimension"
        );
    }
    let vt_bond = &vt.indices()[0];
    for bi in 0..vt_bond.num_blocks() {
        assert!(
            vt_bond.block_dim(bi) > 0,
            "bond block {bi} has zero dimension"
        );
    }
}

// -- Multi-tuple fused sector tests ------------------------------------------
#[test]
fn svd_rank4_multi_tuple_reconstruction() {
    let bs = sample_rank4_multi_tuple();
    let (u, sv, vt) =
        svd_block_sparse_with_policy_dense(&backend(), &bs, 2, ExecPolicy::Sequential).unwrap();
    assert_eq!(u.rank(), 3);
    assert_eq!(vt.rank(), 3);
    verify_svd_reconstruction(&bs, &u, &sv, &vt, 2, order());
}

#[test]
fn qr_rank4_multi_tuple_reconstruction() {
    let bs = sample_rank4_multi_tuple();
    let (q, r) =
        qr_block_sparse_with_policy_dense(&backend(), &bs, 2, ExecPolicy::Sequential).unwrap();
    assert_eq!(q.rank(), 3);
    assert_eq!(r.rank(), 3);
    verify_two_factor_reconstruction(&bs, &q, &r, 2, order());
}

#[test]
fn lq_rank4_multi_tuple_reconstruction() {
    let bs = sample_rank4_multi_tuple();
    let (l, q) =
        lq_block_sparse_with_policy_dense(&backend(), &bs, 2, ExecPolicy::Sequential).unwrap();
    assert_eq!(l.rank(), 3);
    assert_eq!(q.rank(), 3);
    verify_two_factor_reconstruction(&bs, &l, &q, 2, order());
}

#[test]
fn trunc_svd_rank4_multi_tuple_reconstruction() {
    let bs = sample_rank4_multi_tuple();
    let params = TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    };
    let (u, sv, vt, trunc_err) = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &bs,
        2,
        &params,
        ExecPolicy::Sequential,
    )
    .unwrap();
    let total_kept: usize = sv.values.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(total_kept, 2);
    assert!(trunc_err > 0.0);
    // Bond should not have zero-dimension blocks
    let u_bond = u.indices().last().unwrap();
    for bi in 0..u_bond.num_blocks() {
        assert!(u_bond.block_dim(bi) > 0);
    }
    assert_eq!(u.rank(), 3);
    assert_eq!(vt.rank(), 3);
}
// -- Z2 symmetry test --------------------------------------------------------

#[test]
fn svd_z2_reconstruction() {
    let mut bs = BlockSparseTensorData::<f64, Z2Sector>::zeros(
        out_in_legs(
            vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)],
            vec![(Z2Sector::new(0), 4), (Z2Sector::new(1), 5)],
        ),
        Z2Sector::new(0),
        order(),
    );
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
    let (u, sv, vt) =
        svd_block_sparse_with_policy_dense(&backend(), &bs, 1, ExecPolicy::Sequential).unwrap();
    assert_eq!(*u.flux(), Z2Sector::new(0));
    assert_eq!(*vt.flux(), Z2Sector::new(0));
    assert_eq!(sv.values.len(), 2);
    verify_svd_reconstruction(&bs, &u, &sv, &vt, 1, order());
}
// -- Empty tensor test -------------------------------------------------------

#[test]
fn svd_empty_tensor() {
    let bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 2)], Direction::Out),
            (vec![(U1Sector(0), 3)], Direction::Out),
        ]),
        U1Sector(1),
        order(),
    );
    let (u, sv, vt) =
        svd_block_sparse_with_policy_dense(&backend(), &bs, 1, ExecPolicy::Sequential).unwrap();
    assert_eq!(sv.values.len(), 0);
    assert_eq!(u.num_blocks(), 0);
    assert_eq!(vt.num_blocks(), 0);
}

#[test]
fn trunc_svd_empty_tensor_with_target_err() {
    let bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 2)], Direction::Out),
            (vec![(U1Sector(0), 3)], Direction::Out),
        ]),
        U1Sector(1),
        order(),
    );
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(0.1),
    };
    let (u, sv, vt, trunc_err) = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &bs,
        1,
        &params,
        ExecPolicy::Sequential,
    )
    .unwrap();
    assert_eq!(sv.values.len(), 0);
    assert_eq!(u.num_blocks(), 0);
    assert_eq!(vt.num_blocks(), 0);
    assert!(trunc_err.abs() < 1e-15);
}

mod mutant;
mod policy_forwarding;
