//! Authority-routing tests for the block-sparse explicit-backend paths.
//!
//! Mirrors the dense suite: a twin is called with a backend `a2` distinct from
//! the one the input tensor carries (`a1`), and the tensor's backend must drive
//! no kernel call. A dedicated negative test exercises the release-active
//! layout-order check by fabricating a tensor whose layout disagrees with the
//! supplied backend's preferred order.

use std::sync::Arc;

use arnet_core::backend::MemoryOrder;
use arnet_tensor::{
    BlockCoord, BlockSparseTensor, BlockSparseTensorData, Direction, QNIndex, U1Sector,
};

use crate::test_util::RecordingBackend;
use crate::*;

fn total_recorded(b: &RecordingBackend) -> usize {
    b.svd_policies.lock().unwrap().len()
        + b.qr_policies.lock().unwrap().len()
        + b.lq_policies.lock().unwrap().len()
        + b.gemm_policies.lock().unwrap().len()
}

fn pair() -> (Arc<RecordingBackend>, Arc<RecordingBackend>) {
    (
        Arc::new(RecordingBackend::new()),
        Arc::new(RecordingBackend::new()),
    )
}

/// Rank-2 U1 data, flux 0: Out(0:2, 1:3), In(0:2, 1:3), laid out in `order`.
fn rank2_data(order: MemoryOrder) -> BlockSparseTensorData<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0), order);
    bs.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    bs.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    bs
}

/// Rank-2 tensor pinned to backend `a` (built in `a`'s preferred order).
fn rank2(a: &Arc<RecordingBackend>) -> BlockSparseTensor<f64, U1Sector, RecordingBackend> {
    let order = a.preferred_order();
    BlockSparseTensor::with_backend(rank2_data(order), a.clone())
}

/// Compare two block-sparse tensors' joined data block by block.
fn bsp_eq(
    a: &BlockSparseTensor<f64, U1Sector, RecordingBackend>,
    b: &BlockSparseTensor<f64, U1Sector, RecordingBackend>,
) {
    let (da, db) = (a.data(), b.data());
    assert_eq!(da.shape(), db.shape(), "shape mismatch");
    assert_eq!(da.num_blocks(), db.num_blocks(), "block count mismatch");
    for meta in da.block_metas() {
        let xa = da.block_data(&meta.coord).unwrap();
        let xb = db.block_data(&meta.coord).unwrap();
        assert_eq!(xa.len(), xb.len());
        for (x, y) in xa.iter().zip(xb) {
            assert!((x - y).abs() < 1e-10, "value mismatch: {x} vs {y}");
        }
    }
}

#[test]
fn svd_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = rank2(&a1);
    let (u, _s, vt) = svd_block_sparse_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(
        total_recorded(&a1),
        0,
        "tensor backend must not be consulted"
    );
    assert!(
        total_recorded(&a2) > 0,
        "passed backend must run the kernel"
    );
    assert!(Arc::ptr_eq(u.backend_arc(), &a2));
    assert!(Arc::ptr_eq(vt.backend_arc(), &a2));
    let (lu, _ls, lvt) = svd_block_sparse(&t, 1).unwrap();
    bsp_eq(&u, &lu);
    bsp_eq(&vt, &lvt);
}

#[test]
fn trunc_svd_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = rank2(&a1);
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let (u, _s, vt, _err) = trunc_svd_block_sparse_with_backend(&a2, &t, 1, &params).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(u.backend_arc(), &a2));
    assert!(Arc::ptr_eq(vt.backend_arc(), &a2));
    let (lu, _ls, lvt, _lerr) = trunc_svd_block_sparse(&t, 1, &params).unwrap();
    bsp_eq(&u, &lu);
    bsp_eq(&vt, &lvt);
}

#[test]
fn qr_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = rank2(&a1);
    let (q, r) = qr_block_sparse_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(q.backend_arc(), &a2));
    assert!(Arc::ptr_eq(r.backend_arc(), &a2));
    let (lq, lr) = qr_block_sparse(&t, 1).unwrap();
    bsp_eq(&q, &lq);
    bsp_eq(&r, &lr);
}

#[test]
fn lq_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = rank2(&a1);
    let (l, q) = lq_block_sparse_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(l.backend_arc(), &a2));
    assert!(Arc::ptr_eq(q.backend_arc(), &a2));
    let (ll, lq) = lq_block_sparse(&t, 1).unwrap();
    bsp_eq(&l, &ll);
    bsp_eq(&q, &lq);
}

#[test]
fn contract_routes_to_passed_backend() {
    let (a1, a2) = pair();
    // t1's In leg (axis 1) contracts with t2's Out leg (axis 0): matching
    // sectors, opposite direction.
    let t1 = rank2(&a1);
    let t2 = rank2(&a1);
    let out = contract_block_sparse_with_backend(&a2, &t1, &t2, &[1], &[0]).unwrap();
    assert_eq!(
        total_recorded(&a1),
        0,
        "neither operand's backend is consulted"
    );
    assert!(total_recorded(&a2) > 0);
    let lout = contract_block_sparse(&t1, &t2, &[1], &[0]).unwrap();
    match (out, lout) {
        (BlockSparseContractResult::Tensor(t), BlockSparseContractResult::Tensor(lt)) => {
            assert!(Arc::ptr_eq(t.backend_arc(), &a2));
            bsp_eq(&t, &lt);
        }
        _ => panic!("expected a tensor result"),
    }
}

#[test]
fn permute_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = rank2(&a1);
    let out = permute_block_sparse_with_backend(&a2, &t, &[1, 0]).unwrap();
    // Allocation-only kernel: pointer identity is the available authority proof.
    assert_eq!(total_recorded(&a1), 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    bsp_eq(&out, &permute_block_sparse(&t, &[1, 0]).unwrap());
}

#[test]
fn fuse_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = rank2(&a1);
    let out = fuse_legs_block_sparse_with_backend(&a2, &t, 0, 2, Direction::Out).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    bsp_eq(
        &out,
        &fuse_legs_block_sparse(&t, 0, 2, Direction::Out).unwrap(),
    );
}

#[test]
fn diagonal_scale_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = rank2(&a1);
    // Derive a valid weight set (singular values) and a tensor with the bond
    // on axis 0 from an SVD of `t`.
    let (_u, sv, vt) = svd_block_sparse(&t, 1).unwrap();
    let out = diagonal_scale_block_sparse_with_backend(&a2, &vt, &sv, 0).unwrap();
    // Allocation-only kernel: pointer identity is the available authority proof.
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    bsp_eq(&out, &diagonal_scale_block_sparse(&vt, &sv, 0).unwrap());
}

#[test]
fn mismatched_layout_order_is_rejected() {
    // Fabricate a tensor whose layout (row-major) disagrees with the supplied
    // backend's preferred order (column-major). The explicit-backend release
    // check must reject it rather than silently misinterpret the buffer.
    let backend = Arc::new(RecordingBackend::new());
    assert_eq!(backend.preferred_order(), MemoryOrder::ColumnMajor);
    let t = BlockSparseTensor::with_backend(rank2_data(MemoryOrder::RowMajor), backend.clone());
    let err = svd_block_sparse_with_backend(&backend, &t, 1).unwrap_err();
    assert!(
        matches!(err, LinalgError::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}
