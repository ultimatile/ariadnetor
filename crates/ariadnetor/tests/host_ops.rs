//! Umbrella-surface test for the Host-defaulting method traits.
//!
//! Verifies the User-API path end to end: both traits import from `arnet`
//! and their methods resolve on tensors built through the umbrella types,
//! with no backend handle at the call site.

use arnet::{
    BlockCoord, BlockSparseHostOps, BlockSparseSvdResult, BlockSparseTensor, DenseHostOps,
    DenseTensor, Direction, QNIndex, U1Sector,
};

#[test]
fn dense_methods_resolve_through_umbrella() {
    // Built through the umbrella's own safe surface (zeros + set), the
    // only construction path an end user has: the raw flat-buffer
    // constructor is not on the umbrella API. A full-rank 2x3 matrix is
    // all the SVD / permute shape assertions below need.
    let mut t = DenseTensor::<f64>::zeros(vec![2, 3]);
    for (i, v) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().enumerate() {
        t.set([i / 3, i % 3], v);
    }
    let (u, s, vt) = t.svd(1).expect("svd via method");
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 3]);
    let tt = t.permute(&[1, 0]).expect("permute via method");
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

    // `BlockSparseHostOps::svd` returns the `(U, S, Vt)` alias; binding it with
    // an explicit `arnet::`-qualified type is the load-bearing assertion here —
    // it fails to compile unless the result alias is re-exported. This alias is
    // the part newly re-exported by this change (the dense aliases already
    // were). The shape check below is only a smoke test that the call ran.
    let (u, _s, _vt): BlockSparseSvdResult<f64, U1Sector> =
        t.svd(1).expect("block-sparse svd via method");
    assert_eq!(u.shape()[0], t.shape()[0]);
}
