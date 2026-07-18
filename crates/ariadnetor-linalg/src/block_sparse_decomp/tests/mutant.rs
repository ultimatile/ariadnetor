use super::*;

#[test]
fn trunc_svd_error_and_target_err_arithmetic() {
    let bs = sample_known_svs(); // SVs = [3, 2, 1, 1]
    // chi_max=1 discards [2,1,1] → trunc_err = sqrt(4+1+1) = sqrt(6)
    let p1 = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let (_, _, _, err) =
        trunc_svd_block_sparse_with_policy_dense(&backend(), &bs, 1, &p1, ExecPolicy::Sequential)
            .unwrap();
    assert!((err - 6.0f64.sqrt()).abs() < 1e-10, "trunc_err={err}");
    // target_err=0.5 → target_sq=0.25; smallest sv²=1 > 0.25 → all kept
    let p2 = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(0.5),
    };
    let (_, sv, _, err2) =
        trunc_svd_block_sparse_with_policy_dense(&backend(), &bs, 1, &p2, ExecPolicy::Sequential)
            .unwrap();
    let kept: usize = sv.values.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(kept, 4);
    assert!(err2.abs() < 1e-12);
}

#[test]
fn trunc_svd_error_stays_finite_at_overflow_scale() {
    // Two equal singular values 1e200 across sectors (well-conditioned, so the
    // per-sector SVDs resolve them reliably). Discarding one leaves a
    // truncation error of 1e200; a naive sum of squares squares it to 1e400
    // and saturates to inf.
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        square_legs(vec![(U1Sector(0), 1), (U1Sector(1), 1)]),
        U1Sector(0),
        order(),
    );
    bs.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[1e200]);
    bs.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[1e200]);

    let params = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let (_, sv, _, err) = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &bs,
        1,
        &params,
        ExecPolicy::Sequential,
    )
    .unwrap();

    let kept: usize = sv.values.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(kept, 1);
    assert!(err.is_finite(), "trunc_err={err} must stay finite");
    assert!((err - 1e200).abs() < 1e190);
}

// ---------------------------------------------------------------------------
// RowMajor: direct tests for fused sector and truncation RM paths
// ---------------------------------------------------------------------------

/// Rank-4 tensor where fused sector U1(1) has multi-tuple left AND right,
/// each with dims > 1, producing non-zero row_off (2) and col_off (2).
fn sample_rm_rank4() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In),
        ]),
        U1Sector(0),
        MemoryOrder::RowMajor,
    );

    // Sector U1(1) blocks with distinct sequential values:
    // (0,1,0,1): m_i=2, n_j=2, 4 elements
    bs.block_data_mut(&BlockCoord(vec![0, 1, 0, 1]))
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    // (0,1,1,0): m_i=2, n_j=3, 6 elements
    bs.block_data_mut(&BlockCoord(vec![0, 1, 1, 0]))
        .unwrap()
        .copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    // (1,0,0,1): m_i=3, n_j=2, 6 elements
    bs.block_data_mut(&BlockCoord(vec![1, 0, 0, 1]))
        .unwrap()
        .copy_from_slice(&[11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    // (1,0,1,0): m_i=3, n_j=3, 9 elements
    bs.block_data_mut(&BlockCoord(vec![1, 0, 1, 0]))
        .unwrap()
        .copy_from_slice(&[17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]);
    bs
}

/// assemble_sector_matrix RM: hand-verified 5×5 block matrix.
///
/// Sector U1(1) has left_offsets=[0,2], right_offsets=[0,2] with
/// block dims (2,3) × (2,3), exercising all arithmetic in lines 171-173.
#[test]
fn assemble_sector_matrix_row_major() {
    let bs = sample_rm_rank4();
    let groups = compute_fused_sector_groups(&bs, 2);
    let group = groups.iter().find(|g| g.sector == U1Sector(1)).unwrap();

    assert_eq!(group.m, 5);
    assert_eq!(group.n, 5);
    assert_eq!(group.left_offsets, vec![0, 2]);
    assert_eq!(group.right_offsets, vec![0, 2]);

    let rm = assemble_sector_matrix(&bs, group, MemoryOrder::RowMajor);

    // 5×5 RM block matrix [[A(2×2), B(2×3)], [C(3×2), D(3×3)]]
    #[rustfmt::skip]
    let expected = vec![
         1.0,  2.0,  5.0,  6.0,  7.0,
         3.0,  4.0,  8.0,  9.0, 10.0,
        11.0, 12.0, 17.0, 18.0, 19.0,
        13.0, 14.0, 20.0, 21.0, 22.0,
        15.0, 16.0, 23.0, 24.0, 25.0,
    ];
    assert_eq!(rm, expected);
}

/// build_left_tensor RM: verify block data from a known 5×2 RM matrix.
///
/// Exercises lines 234-236 (RM branch of build_left_tensor).
#[test]
fn build_left_tensor_row_major() {
    let bs = sample_rm_rank4();
    let groups = compute_fused_sector_groups(&bs, 2);

    // k=2 for sector U1(1) only
    let k_per_sector: Vec<usize> = groups
        .iter()
        .map(|g| if g.sector == U1Sector(1) { 2 } else { 0 })
        .collect();

    // 5×2 RM matrix: [1..10]
    let left_mat: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let left_matrices: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            if g.sector == U1Sector(1) {
                left_mat.clone()
            } else {
                vec![]
            }
        })
        .collect();

    let result = build_left_tensor(
        &groups,
        &left_matrices,
        &k_per_sector,
        bs.indices(),
        2,
        MemoryOrder::RowMajor,
    );

    // Left tuple [0,1] (m_i=2, row_off=0): rows 0-1 → [1,2, 3,4]
    assert_eq!(
        result.block_data(&BlockCoord(vec![0, 1, 0])).unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );
    // Left tuple [1,0] (m_i=3, row_off=2): rows 2-4 → [5,6, 7,8, 9,10]
    assert_eq!(
        result.block_data(&BlockCoord(vec![1, 0, 0])).unwrap(),
        &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    );
}

/// build_right_tensor RM: verify block data from a known 2×5 RM matrix.
///
/// Exercises lines 298-300 (RM branch of build_right_tensor).
#[test]
fn build_right_tensor_row_major() {
    let bs = sample_rm_rank4();
    let groups = compute_fused_sector_groups(&bs, 2);

    let k_per_sector: Vec<usize> = groups
        .iter()
        .map(|g| if g.sector == U1Sector(1) { 2 } else { 0 })
        .collect();

    // 2×5 RM matrix: [1..10]
    let right_mat: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let right_matrices: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            if g.sector == U1Sector(1) {
                right_mat.clone()
            } else {
                vec![]
            }
        })
        .collect();

    let result = build_right_tensor(
        &groups,
        &right_matrices,
        &k_per_sector,
        bs.indices(),
        2,
        U1Sector(0),
        MemoryOrder::RowMajor,
    );

    // Right tuple [0,1] (n_j=2, col_off=0):
    //   r=0: mat[0..2]=[1,2], r=1: mat[5..7]=[6,7]
    assert_eq!(
        result.block_data(&BlockCoord(vec![0, 0, 1])).unwrap(),
        &[1.0, 2.0, 6.0, 7.0]
    );
    // Right tuple [1,0] (n_j=3, col_off=2):
    //   r=0: mat[2..5]=[3,4,5], r=1: mat[7..10]=[8,9,10]
    assert_eq!(
        result.block_data(&BlockCoord(vec![0, 1, 0])).unwrap(),
        &[3.0, 4.0, 5.0, 8.0, 9.0, 10.0]
    );
}

/// truncate_cols RM: keep first 2 cols of 3×4 RM matrix.
#[test]
fn truncate_cols_row_major() {
    // 3×4 RM matrix
    #[rustfmt::skip]
    let data: Vec<f64> = vec![
        1.0,  2.0,  3.0,  4.0,   // row 0
        5.0,  6.0,  7.0,  8.0,   // row 1
        9.0, 10.0, 11.0, 12.0,   // row 2
    ];
    let result = truncate_cols(&data, 3, 4, 2, MemoryOrder::RowMajor);
    // Keep first 2 cols: [[1,2],[5,6],[9,10]]
    assert_eq!(result, vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
}

/// truncate_rows RM: keep first 2 rows of 4×3 RM matrix.
#[test]
fn truncate_rows_row_major() {
    // 4×3 RM matrix
    #[rustfmt::skip]
    let data: Vec<f64> = vec![
        1.0,  2.0,  3.0,   // row 0
        4.0,  5.0,  6.0,   // row 1
        7.0,  8.0,  9.0,   // row 2
        10.0, 11.0, 12.0,  // row 3
    ];
    let result = truncate_rows(&data, 4, 3, 2, MemoryOrder::RowMajor);
    // Keep first 2 rows: [[1,2,3],[4,5,6]]
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
