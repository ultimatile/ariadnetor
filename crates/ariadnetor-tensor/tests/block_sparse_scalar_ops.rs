//! Scalar-dependent operations (conj, dagger, norm, normalize) for
//! `BlockSparseTensorData<T, S>`.

use arnet_tensor::{
    BlockCoord, BlockSparseTensorData, Direction, MemoryOrder, QNIndex, U1Sector, Z2Sector,
};
use num_complex::Complex;

/// Helper: rank-2 U1 real tensor with flux=0, blocks (0,0):2×2 and (1,1):3×3.
fn sample_u1_rank2() -> BlockSparseTensorData<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut td = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
    );
    {
        let d = td.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
        d.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    }
    {
        let d = td.block_data_mut(&BlockCoord(vec![1, 1])).unwrap();
        d.copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    }
    td
}

/// Helper: rank-2 U1 complex tensor with flux=0.
fn sample_u1_complex() -> BlockSparseTensorData<Complex<f64>, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let mut td = BlockSparseTensorData::<Complex<f64>, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
    );
    {
        let d = td.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
        d.copy_from_slice(&[
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(0.0, 4.0),
            Complex::new(-2.0, 0.5),
        ]);
    }
    td
}

// =========================================================================
// conj
// =========================================================================

#[test]
fn conj_real_f64() {
    let td = sample_u1_rank2();
    let c = td.conj();
    // Real conjugate is identity.
    for meta in c.block_metas() {
        let orig = td.block_data(&meta.coord).unwrap();
        let conj = c.block_data(&meta.coord).unwrap();
        assert_eq!(orig, conj);
    }
}

#[test]
fn conj_complex_f64() {
    let td = sample_u1_complex();
    let c = td.conj();
    let orig = td.block_data(&BlockCoord(vec![0, 0])).unwrap();
    let conj = c.block_data(&BlockCoord(vec![0, 0])).unwrap();
    for (o, c) in orig.iter().zip(conj.iter()) {
        assert_eq!(c.re, o.re);
        assert_eq!(c.im, -o.im);
    }
}

#[test]
fn conj_preserves_structure() {
    let td = sample_u1_rank2();
    let c = td.conj();
    assert_eq!(td.shape(), c.shape());
    assert_eq!(td.rank(), c.rank());
    assert_eq!(td.num_blocks(), c.num_blocks());
    assert_eq!(td.storage().stored_len(), c.storage().stored_len());
    assert_eq!(td.flux(), c.flux());
    for (bm, cm) in td.block_metas().iter().zip(c.block_metas().iter()) {
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
    let td = sample_u1_rank2();
    // Elements: 1..=13, norm = sqrt(1² + 2² + ... + 13²) = sqrt(819)
    let expected = (1..=13).map(|x| (x * x) as f64).sum::<f64>().sqrt();
    let norm = td.norm();
    assert!((norm - expected).abs() < 1e-12);
}

#[test]
fn norm_zero_tensor() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let td = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
    );
    assert_eq!(td.norm(), 0.0);
}

#[test]
fn norm_complex() {
    let td = sample_u1_complex();
    // |1+2i|²=5, |3-i|²=10, |4i|²=16, |-2+0.5i|²=4.25 → sum=35.25
    let expected = 35.25_f64.sqrt();
    let norm = td.norm();
    assert!((norm - expected).abs() < 1e-12);
}

#[test]
fn norm_frobenius_alias() {
    let td = sample_u1_rank2();
    assert_eq!(td.norm(), td.storage().norm_frobenius());
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
    let mut td = BlockSparseTensorData::<f64, Z2Sector>::zeros(
        vec![row, col],
        Z2Sector::new(0),
        MemoryOrder::RowMajor,
    );
    {
        let d = td.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
        d.copy_from_slice(&[3.0, 0.0, 0.0, 4.0]);
    }
    {
        let d = td.block_data_mut(&BlockCoord(vec![1, 1])).unwrap();
        d.copy_from_slice(&[5.0]);
    }
    // norm = sqrt(9 + 16 + 25) = sqrt(50)
    let expected = 50.0_f64.sqrt();
    assert!((td.norm() - expected).abs() < 1e-12);
}

// =========================================================================
// dagger
// =========================================================================

#[test]
fn dagger_involution_real() {
    let td = sample_u1_rank2();
    let dd = td.dagger().dagger();
    assert_eq!(td.shape(), dd.shape());
    assert_eq!(td.flux(), dd.flux());
    for (bi, di) in td.indices().iter().zip(dd.indices().iter()) {
        assert_eq!(bi.direction(), di.direction());
        assert_eq!(bi.blocks(), di.blocks());
    }
    for meta in td.block_metas() {
        let a = td.block_data(&meta.coord).unwrap();
        let b = dd.block_data(&meta.coord).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn dagger_involution_complex() {
    let td = sample_u1_complex();
    let dd = td.dagger().dagger();
    for meta in td.block_metas() {
        let a = td.block_data(&meta.coord).unwrap();
        let b = dd.block_data(&meta.coord).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.re - y.re).abs() < 1e-14);
            assert!((x.im - y.im).abs() < 1e-14);
        }
    }
}

#[test]
fn dagger_flips_directions() {
    let td = sample_u1_rank2();
    let dag = td.dagger();
    for (orig, flipped) in td.indices().iter().zip(dag.indices().iter()) {
        assert_ne!(orig.direction(), flipped.direction());
        assert_eq!(orig.blocks(), flipped.blocks());
    }
}

#[test]
fn dagger_duals_flux() {
    // Non-identity flux: rank-2 tensor with flux=U1(1)
    let row = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
    let mut td = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(1),
        MemoryOrder::RowMajor,
    );
    // Allowed block: (1, 0) since Out.apply(1) + In.apply(0) = 1 + 0 = 1 = flux
    td.block_data_mut(&BlockCoord(vec![1, 0])).unwrap()[0] = 7.0;

    let dag = td.dagger();
    assert_eq!(*dag.flux(), U1Sector(-1));
    let d = dag.block_data(&BlockCoord(vec![1, 0])).unwrap();
    assert_eq!(d[0], 7.0);
}

#[test]
fn dagger_conjugates_complex_elements() {
    let td = sample_u1_complex();
    let dag = td.dagger();
    let orig = td.block_data(&BlockCoord(vec![0, 0])).unwrap();
    let dagd = dag.block_data(&BlockCoord(vec![0, 0])).unwrap();
    for (o, d) in orig.iter().zip(dagd.iter()) {
        assert_eq!(d.re, o.re);
        assert_eq!(d.im, -o.im);
    }
}

#[test]
fn dagger_preserves_block_count() {
    let td = sample_u1_rank2();
    let dag = td.dagger();
    assert_eq!(td.num_blocks(), dag.num_blocks());
    assert_eq!(td.shape(), dag.shape());
}

// =========================================================================
// normalize / normalized — exposed on the storage half.
// =========================================================================

#[test]
fn normalize_f64() {
    let mut td = sample_u1_rank2();
    let orig_norm = td.storage_mut().normalize();
    assert!(orig_norm > 0.0);
    assert!((td.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn normalized_out_of_place() {
    let td = sample_u1_rank2();
    let orig_norm_expected = td.norm();
    let (normed_storage, orig_norm) = td.storage().normalized();
    // Original unchanged.
    assert!((td.norm() - orig_norm_expected).abs() < 1e-14);
    // Normalized storage has unit norm.
    assert!((normed_storage.norm() - 1.0).abs() < 1e-12);
    assert!((orig_norm - orig_norm_expected).abs() < 1e-14);
}

#[test]
fn normalize_complex() {
    let mut td = sample_u1_complex();
    td.storage_mut().normalize();
    assert!((td.norm() - 1.0).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "Cannot normalize zero tensor")]
fn normalize_zero_panics() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let mut td = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
    );
    td.storage_mut().normalize();
}
