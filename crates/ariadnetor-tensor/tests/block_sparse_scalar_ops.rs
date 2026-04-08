use arnet_tensor::block_sparse::{BlockCoord, BlockSparse, Direction, QNIndex};
use arnet_tensor::sector::{U1Sector, Z2Sector};
use num_complex::Complex;

/// Helper: rank-2 U1 tensor with flux=0, blocks (0,0):2×2 and (1,1):3×3.
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

/// Helper: rank-2 U1 complex tensor with flux=0.
fn sample_u1_complex() -> BlockSparse<Complex<f64>, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let mut bs = BlockSparse::<Complex<f64>, U1Sector>::zeros(vec![row, col], U1Sector(0));
    let d = bs.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    d.copy_from_slice(&[
        Complex::new(1.0, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0),
        Complex::new(-2.0, 0.5),
    ]);
    bs
}

// =========================================================================
// conj
// =========================================================================

#[test]
fn conj_real_f64() {
    let bs = sample_u1_rank2();
    let c = bs.conj();
    // Real conjugate is identity.
    for meta in c.block_metas() {
        let orig = bs.block_data(&meta.coord).unwrap();
        let conj = c.block_data(&meta.coord).unwrap();
        assert_eq!(orig, conj);
    }
}

#[test]
fn conj_complex_f64() {
    let bs = sample_u1_complex();
    let c = bs.conj();
    let orig = bs.block_data(&BlockCoord(vec![0, 0])).unwrap();
    let conj = c.block_data(&BlockCoord(vec![0, 0])).unwrap();
    for (o, c) in orig.iter().zip(conj.iter()) {
        assert_eq!(c.re, o.re);
        assert_eq!(c.im, -o.im);
    }
}

#[test]
fn conj_preserves_structure() {
    let bs = sample_u1_rank2();
    let c = bs.conj();
    assert_eq!(bs.shape(), c.shape());
    assert_eq!(bs.rank(), c.rank());
    assert_eq!(bs.num_blocks(), c.num_blocks());
    assert_eq!(bs.stored_len(), c.stored_len());
    assert_eq!(bs.flux(), c.flux());
    for (bm, cm) in bs.block_metas().iter().zip(c.block_metas().iter()) {
        assert_eq!(bm.coord, cm.coord);
        assert_eq!(bm.offset, cm.offset);
        assert_eq!(bm.size, cm.size);
    }
}

// =========================================================================
// norm
// =========================================================================

#[test]
fn norm_u1_rank2() {
    let bs = sample_u1_rank2();
    // Elements: 1..=13, norm = sqrt(1² + 2² + ... + 13²) = sqrt(819)
    let expected = (1..=13).map(|x| (x * x) as f64).sum::<f64>().sqrt();
    let norm = bs.norm();
    assert!((norm - expected).abs() < 1e-12);
}

#[test]
fn norm_zero_tensor() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let bs = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));
    assert_eq!(bs.norm(), 0.0);
}

#[test]
fn norm_complex() {
    let bs = sample_u1_complex();
    // |1+2i|²=5, |3-i|²=10, |4i|²=16, |-2+0.5i|²=4.25 → sum=35.25
    let expected = 35.25_f64.sqrt();
    let norm = bs.norm();
    assert!((norm - expected).abs() < 1e-12);
}

#[test]
fn norm_frobenius_alias() {
    let bs = sample_u1_rank2();
    assert_eq!(bs.norm(), bs.norm_frobenius());
}

#[test]
fn norm_z2_sector() {
    let row = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 1)],
        Direction::Out,
    );
    let col = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 1)],
        Direction::In,
    );
    let mut bs = BlockSparse::<f64, Z2Sector>::zeros(vec![row, col], Z2Sector::new(0));
    let d = bs.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    d.copy_from_slice(&[3.0, 0.0, 0.0, 4.0]);
    // (1,1) block: 1×1
    let d = bs.block_data_mut(&BlockCoord(vec![1, 1])).unwrap();
    d.copy_from_slice(&[5.0]);
    // norm = sqrt(9 + 16 + 25) = sqrt(50)
    let expected = 50.0_f64.sqrt();
    assert!((bs.norm() - expected).abs() < 1e-12);
}

// =========================================================================
// normalize / normalized
// =========================================================================

#[test]
fn normalize_f64() {
    let mut bs = sample_u1_rank2();
    let orig_norm = bs.normalize();
    assert!(orig_norm > 0.0);
    assert!((bs.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn normalized_out_of_place() {
    let bs = sample_u1_rank2();
    let orig_norm_expected = bs.norm();
    let (normed, orig_norm) = bs.normalized();
    // Original unchanged.
    assert!((bs.norm() - orig_norm_expected).abs() < 1e-14);
    // Normalized tensor has unit norm.
    assert!((normed.norm() - 1.0).abs() < 1e-12);
    assert!((orig_norm - orig_norm_expected).abs() < 1e-14);
}

#[test]
fn normalize_complex() {
    let mut bs = sample_u1_complex();
    bs.normalize();
    assert!((bs.norm() - 1.0).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "Cannot normalize zero tensor")]
fn normalize_zero_panics() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let mut bs = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));
    bs.normalize();
}
