//! Umbrella-surface test for the Host-defaulting method traits.
//!
//! Verifies the User-API path end to end: both traits import from `arnet`
//! and their methods resolve on tensors built through the umbrella types,
//! with no backend handle at the call site.

use arnet::{
    BlockCoord, BlockSparseHostOps, BlockSparseTensor, DenseHostOps, DenseTensor, Direction,
    QNIndex, U1Sector,
};

#[test]
fn dense_methods_resolve_through_umbrella() {
    let t = DenseTensor::<f64>::from_raw_parts(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let (u, s, vt) = t.svd(1).expect("svd via method");
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 3]);
    let tt = t.transpose(&[1, 0]).expect("transpose via method");
    assert_eq!(tt.shape(), &[3, 2]);
}

#[test]
fn block_sparse_methods_resolve_through_umbrella() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 2)], Direction::In);
    let mut t = BlockSparseTensor::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));
    t.block_data_mut(&BlockCoord(vec![0, 0]))
        .expect("block (0,0) exists")
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let out = t.permute(&[1, 0]).expect("permute via method");
    assert_eq!(out.shape(), t.shape());
}
