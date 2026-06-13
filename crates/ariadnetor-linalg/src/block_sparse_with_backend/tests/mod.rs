//! Authority-routing tests for the block-sparse explicit-backend paths.
//!
//! Each kernel-dispatching twin is exercised with a [`RecordingBackend`] and
//! the central invariant proved here is that the operation routes its kernel
//! to the call-site-supplied backend — the recorder must register the call,
//! catching a regression where a twin ignores its `backend` argument and falls
//! back to a hardcoded `Host`. Results are checked against an independent
//! `NativeBackend` run for numerical agreement.
//!
//! Tensors no longer carry a backend, so the former "the tensor's own backend
//! is never consulted" half of the invariant is structurally unviolable and is
//! dropped. For the allocation-only ops (`permute`, `fuse`, `diagonal_scale`)
//! the backend drives no observable kernel, so only numerical correctness is
//! checked. A dedicated negative test exercises the release-active layout-order
//! check by fabricating a tensor whose layout disagrees with the supplied
//! backend's preferred order.

use arnet_core::backend::MemoryOrder;
use arnet_native::NativeBackend;
use arnet_tensor::{
    BlockCoord, BlockSparseTensor, BlockSparseTensorData, Direction, QNIndex, U1Sector,
};

use crate::test_util::RecordingBackend;
use crate::*;

/// Number of kernel descriptors a recording backend has seen across the
/// op families block-sparse paths dispatch into. Zero means the backend drove
/// no computation.
fn total_recorded(b: &RecordingBackend) -> usize {
    b.svd_policies.lock().unwrap().len()
        + b.qr_policies.lock().unwrap().len()
        + b.lq_policies.lock().unwrap().len()
        + b.gemm_policies.lock().unwrap().len()
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

/// Rank-2 tensor laid out in the recording backend's preferred order.
fn rank2() -> BlockSparseTensor<f64, U1Sector> {
    BlockSparseTensor::from_data(rank2_data(RecordingBackend::new().preferred_order()))
}

/// Compare two block-sparse tensors' joined data block by block.
fn bsp_eq(a: &BlockSparseTensor<f64, U1Sector>, b: &BlockSparseTensor<f64, U1Sector>) {
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
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2();
    let (u, _s, vt) = svd_block_sparse_with_backend(&rec, &t, 1).unwrap();
    assert!(
        total_recorded(&rec) > 0,
        "passed backend must run the kernel"
    );
    let (hu, _hs, hvt) = svd_block_sparse_with_backend(&host, &t, 1).unwrap();
    bsp_eq(&u, &hu);
    bsp_eq(&vt, &hvt);
}

#[test]
fn trunc_svd_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2();
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let (u, _s, vt, _err) = trunc_svd_block_sparse_with_backend(&rec, &t, 1, &params).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hu, _hs, hvt, _herr) = trunc_svd_block_sparse_with_backend(&host, &t, 1, &params).unwrap();
    bsp_eq(&u, &hu);
    bsp_eq(&vt, &hvt);
}

#[test]
fn qr_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2();
    let (q, r) = qr_block_sparse_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hq, hr) = qr_block_sparse_with_backend(&host, &t, 1).unwrap();
    bsp_eq(&q, &hq);
    bsp_eq(&r, &hr);
}

#[test]
fn lq_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2();
    let (l, q) = lq_block_sparse_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hl, hq) = lq_block_sparse_with_backend(&host, &t, 1).unwrap();
    bsp_eq(&l, &hl);
    bsp_eq(&q, &hq);
}

#[test]
fn contract_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    // t1's In leg (axis 1) contracts with t2's Out leg (axis 0): matching
    // sectors, opposite direction.
    let t1 = rank2();
    let t2 = rank2();
    let out = contract_block_sparse_with_backend(&rec, &t1, &t2, &[1], &[0]).unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = contract_block_sparse_with_backend(&host, &t1, &t2, &[1], &[0]).unwrap();
    match (out, hout) {
        (BlockSparseContractResult::Tensor(t), BlockSparseContractResult::Tensor(ht)) => {
            bsp_eq(&t, &ht);
        }
        _ => panic!("expected a tensor result"),
    }
}

// --- Allocation-only ops: the backend drives no observable kernel, so only
// numerical correctness is checked against an independent Host run. ---

#[test]
fn permute_matches_host() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2();
    let out = permute_block_sparse_with_backend(&rec, &t, &[1, 0]).unwrap();
    bsp_eq(
        &out,
        &permute_block_sparse_with_backend(&host, &t, &[1, 0]).unwrap(),
    );
}

#[test]
fn fuse_matches_host() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2();
    let out = fuse_legs_block_sparse_with_backend(&rec, &t, 0, 2, Direction::Out).unwrap();
    bsp_eq(
        &out,
        &fuse_legs_block_sparse_with_backend(&host, &t, 0, 2, Direction::Out).unwrap(),
    );
}

#[test]
fn diagonal_scale_matches_host() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2();
    // Derive a valid weight set (singular values) and a tensor with the bond
    // on axis 0 from an SVD of `t`.
    let (_u, sv, vt) = svd_block_sparse_with_backend(&host, &t, 1).unwrap();
    let out = diagonal_scale_block_sparse_with_backend(&rec, &vt, &sv, 0).unwrap();
    bsp_eq(
        &out,
        &diagonal_scale_block_sparse_with_backend(&host, &vt, &sv, 0).unwrap(),
    );
}

#[test]
fn mismatched_layout_order_is_rejected() {
    // Fabricate a tensor whose layout (row-major) disagrees with the supplied
    // backend's preferred order (column-major). `from_data` does not check the
    // order, so the tensor holds row-major data; the twin's internal layout
    // check must reject it rather than silently misinterpret the buffer.
    let backend = RecordingBackend::new();
    assert_eq!(backend.preferred_order(), MemoryOrder::ColumnMajor);
    let t = BlockSparseTensor::from_data(rank2_data(MemoryOrder::RowMajor));
    let err = svd_block_sparse_with_backend(&backend, &t, 1).unwrap_err();
    assert!(
        matches!(err, LinalgError::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}
