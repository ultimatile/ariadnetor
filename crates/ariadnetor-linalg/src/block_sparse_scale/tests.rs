use arnet_native::NativeBackend;
use arnet_tensor::U1Sector;
use arnet_tensor::{BlockCoord, BlockSparse, Direction, QNIndex};

use crate::block_sparse_contract::{BlockSparseContractResult, contract_block_sparse};
use crate::block_sparse_decomp::BlockSingularValues;
use crate::{TruncSvdParams, svd_block_sparse, trunc_svd_block_sparse};

use super::diagonal_scale_block_sparse;

fn backend() -> NativeBackend {
    NativeBackend::new()
}

/// Rank-2 U1, flux=0, blocks (0,0):2×2 and (1,1):3×3.
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

/// Contract U (rank-2) and Vt/S·Vt (rank-2) to reconstruct a rank-2 tensor.
/// U has bond as last axis (In), Vt has bond as first axis (Out).
fn contract_uv(
    u: &BlockSparse<f64, U1Sector>,
    vt: &BlockSparse<f64, U1Sector>,
) -> BlockSparse<f64, U1Sector> {
    // Contract over bond axis: U's last axis with Vt's first axis.
    let result = contract_block_sparse(&backend(), u, vt, &[u.rank() - 1], &[0]).unwrap();
    match result {
        BlockSparseContractResult::Tensor(t) => t,
        BlockSparseContractResult::Scalar(_) => panic!("expected tensor, got scalar"),
    }
}

/// Assert two BlockSparse tensors are approximately equal.
fn assert_bs_approx(a: &BlockSparse<f64, U1Sector>, b: &BlockSparse<f64, U1Sector>, tol: f64) {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.num_blocks(), b.num_blocks());
    for meta in a.block_metas() {
        let da = a.block_data(&meta.coord).unwrap();
        let db = b
            .block_data(&meta.coord)
            .unwrap_or_else(|| panic!("block {:?} missing in second tensor", meta.coord));
        assert_eq!(da.len(), db.len());
        for (i, (&x, &y)) in da.iter().zip(db.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "block {:?}[{i}]: {x} vs {y} (diff={})",
                meta.coord,
                (x - y).abs()
            );
        }
    }
}

// =========================================================================
// diagonal_scale_block_sparse
// =========================================================================

#[test]
fn scale_vt_axis0() {
    // SVD → scale Vt at axis 0 (bond) by S → verify U·(S·Vt) ≈ original.
    let bs = sample_u1_rank2();
    let (u, sv, vt, _) = trunc_svd_block_sparse(
        &backend(),
        &bs,
        1,
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    )
    .unwrap();

    let svt = diagonal_scale_block_sparse(&backend(), &vt, &sv, 0).unwrap();
    let recon = contract_uv(&u, &svt);
    assert_bs_approx(&recon, &bs, 1e-10);
}

#[test]
fn scale_u_axis_last() {
    // SVD → scale U at last axis (bond) by S → verify (U·S)·Vt ≈ original.
    let bs = sample_u1_rank2();
    let (u, sv, vt, _) = trunc_svd_block_sparse(
        &backend(),
        &bs,
        1,
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    )
    .unwrap();

    let us = diagonal_scale_block_sparse(&backend(), &u, &sv, u.rank() - 1).unwrap();
    let recon = contract_uv(&us, &vt);
    assert_bs_approx(&recon, &bs, 1e-10);
}

#[test]
fn scale_sqrt_via_map() {
    // map(sqrt) on weights → scale both U and Vt → (U·√S)·(√S·Vt) ≈ original.
    let bs = sample_u1_rank2();
    let (u, sv, vt, _) = trunc_svd_block_sparse(
        &backend(),
        &bs,
        1,
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    )
    .unwrap();

    let sqrt_sv = sv.map(|v| v.sqrt());
    let u_scaled = diagonal_scale_block_sparse(&backend(), &u, &sqrt_sv, u.rank() - 1).unwrap();
    let vt_scaled = diagonal_scale_block_sparse(&backend(), &vt, &sqrt_sv, 0).unwrap();
    let recon = contract_uv(&u_scaled, &vt_scaled);
    assert_bs_approx(&recon, &bs, 1e-10);
}

#[test]
fn scale_identity_weights() {
    // All weights 1.0 → tensor unchanged.
    let bs = sample_u1_rank2();
    let (_, sv, vt) = svd_block_sparse(&backend(), &bs, 1).unwrap();

    let ones = sv.map(|_| 1.0_f64);
    let vt_scaled = diagonal_scale_block_sparse(&backend(), &vt, &ones, 0).unwrap();

    for meta in vt.block_metas() {
        let orig = vt.block_data(&meta.coord).unwrap();
        let scaled = vt_scaled.block_data(&meta.coord).unwrap();
        for (i, (&a, &b)) in orig.iter().zip(scaled.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-14,
                "block {:?}[{i}]: {a} vs {b}",
                meta.coord
            );
        }
    }
}

#[test]
fn scale_axis_out_of_range() {
    let bs = sample_u1_rank2();
    let weights = BlockSingularValues {
        values: vec![(U1Sector(0), vec![1.0])],
    };
    let result = diagonal_scale_block_sparse(&backend(), &bs, &weights, 5);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        format!("{err}").contains("out of range"),
        "unexpected error: {err}"
    );
}

#[test]
fn scale_trunc_svd_reconstruction() {
    // Truncated SVD with chi_max=2, then S·Vt, verify reconstruction.
    let bs = sample_u1_rank2();
    let (u, sv, vt, _err) = trunc_svd_block_sparse(
        &backend(),
        &bs,
        1,
        &TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
    )
    .unwrap();

    let svt = diagonal_scale_block_sparse(&backend(), &vt, &sv, 0).unwrap();
    assert_eq!(svt.rank(), vt.rank());
    assert_eq!(svt.shape(), vt.shape());

    // Reconstruction should be a valid low-rank approximation.
    let recon = contract_uv(&u, &svt);
    // Not exact (truncated), just verify it's finite and correct shape.
    assert_eq!(recon.shape(), bs.shape());
    for meta in recon.block_metas() {
        let data = recon.block_data(&meta.coord).unwrap();
        for &v in data {
            assert!(v.is_finite(), "non-finite value in reconstruction");
        }
    }
}

/// Rank-3 element-level scaling test to catch inner_stride arithmetic mutations.
///
/// Uses a single-block (sector 0) rank-3 tensor with distinct dimensions
/// (2×3×4) so that `block_shape[axis+1..]`, `block_shape[..axis]`, and
/// `(idx / inner_stride) % d_axis` all yield observably different results
/// under mutations like `axis + 1` → `axis - 1` or `axis * 1`.
#[test]
fn scale_rank3_middle_axis_element_values() {
    // Single-sector rank-3 tensor: shape (2, 3, 4), 24 elements.
    let idx0 = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let idx1 = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let idx2 = QNIndex::new(vec![(U1Sector(0), 4)], Direction::In);
    let mut bs = BlockSparse::<f64, U1Sector>::zeros(vec![idx0, idx1, idx2], U1Sector(0));
    let data = bs.block_data_mut(&BlockCoord(vec![0, 0, 0])).unwrap();
    // Fill with 1..=24 so every element is distinct.
    for (i, d) in data.iter_mut().enumerate() {
        *d = (i + 1) as f64;
    }

    // Weights for sector 0 at axis=1: 3 elements → [2.0, 3.0, 5.0].
    let weights = BlockSingularValues {
        values: vec![(U1Sector(0), vec![2.0, 3.0, 5.0])],
    };

    let scaled = diagonal_scale_block_sparse(&backend(), &bs, &weights, 1).unwrap();
    let out = scaled.block_data(&BlockCoord(vec![0, 0, 0])).unwrap();

    // NativeBackend uses ColumnMajor: shape (2, 3, 4), axis=1.
    // inner_stride = block_shape[..axis].product() = block_shape[..1].product() = 2.
    // Element at flat index `idx` has axis-1 coordinate = (idx / 2) % 3.
    // Weight applied = weights[(idx / 2) % 3].
    let w = [2.0, 3.0, 5.0];
    for (idx, &val) in out.iter().enumerate() {
        let i_axis = (idx / 2) % 3;
        let expected = ((idx + 1) as f64) * w[i_axis];
        assert!(
            (val - expected).abs() < 1e-12,
            "idx={idx}: got {val}, expected {expected} (i_axis={i_axis}, w={})",
            w[i_axis]
        );
    }
}

// =========================================================================
// BlockSingularValues::map
// =========================================================================

#[test]
fn bsv_map_basic() {
    let bsv = BlockSingularValues {
        values: vec![
            (U1Sector(0), vec![4.0_f64, 1.0]),
            (U1Sector(1), vec![9.0, 16.0, 25.0]),
        ],
    };
    let sqrt_bsv = bsv.map(|v| v.sqrt());
    assert_eq!(sqrt_bsv.values.len(), 2);
    assert_eq!(sqrt_bsv.values[0].0, U1Sector(0));
    assert!((sqrt_bsv.values[0].1[0] - 2.0).abs() < 1e-14);
    assert!((sqrt_bsv.values[0].1[1] - 1.0).abs() < 1e-14);
    assert_eq!(sqrt_bsv.values[1].0, U1Sector(1));
    assert!((sqrt_bsv.values[1].1[0] - 3.0).abs() < 1e-14);
    assert!((sqrt_bsv.values[1].1[1] - 4.0).abs() < 1e-14);
    assert!((sqrt_bsv.values[1].1[2] - 5.0).abs() < 1e-14);
}
