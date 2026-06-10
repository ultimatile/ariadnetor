//! Tests for the Host-defaulting method surface.
//!
//! Authority: each method must dispatch with `NativeBackend::shared()` and
//! never with the receiver's own backend, so every receiver is built with a
//! separate `Arc<NativeBackend>` instance and every result is checked for
//! pointer identity against the shared singleton (and non-identity against
//! the receiver's instance). Kernel routing of the underlying twins is
//! already proven by the `with_backend` suites; the sugar only needs to
//! prove which handle it passes.
//!
//! Equivalence: each method must match its `*_with_backend` twin called
//! with the shared handle. Both paths use the same backend instance, hence
//! the same execution policy, so data-movement ops compare exactly;
//! FP-reduction (faer-backed) ops compare with tolerance. `eig` / `eigvals`
//! results are compared order-insensitively (the backend contract promises
//! no eigenvalue ordering), with eigenvectors verified by residual.

use std::sync::Arc;

use arnet_core::Complex;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_tensor::{
    BlockCoord, BlockSparseTensor, BlockSparseTensorData, DenseTensor, Direction, NativeBackend,
    QNIndex, U1Sector,
};

use crate::*;

fn shared() -> Arc<NativeBackend> {
    NativeBackend::shared()
}

/// A backend instance distinct from the shared singleton, for receivers.
fn own() -> Arc<NativeBackend> {
    Arc::new(NativeBackend::new())
}

/// Assert `t` carries the shared singleton, not the receiver's instance.
fn assert_shared_authority<St, L>(
    t: &arnet_tensor::Tensor<St, L, NativeBackend>,
    receiver_backend: &Arc<NativeBackend>,
) where
    St: arnet_tensor::Storage + arnet_tensor::StorageFor<L>,
    L: arnet_tensor::TensorLayout,
{
    assert!(
        Arc::ptr_eq(t.backend_arc(), &shared()),
        "result must carry the shared Host singleton"
    );
    assert!(
        !Arc::ptr_eq(t.backend_arc(), receiver_backend),
        "result must not carry the receiver's own backend"
    );
}

fn tensor(data: Vec<f64>, shape: Vec<usize>, b: &Arc<NativeBackend>) -> DenseTensor<f64> {
    DenseTensor::from_raw_parts(data, shape, b.clone())
}

fn sym2(b: &Arc<NativeBackend>) -> DenseTensor<f64> {
    // Symmetric, so eigh / expm_hermitian have a real spectrum.
    tensor(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2], b)
}

fn mat23(b: &Arc<NativeBackend>) -> DenseTensor<f64> {
    tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], b)
}

fn mat22(b: &Arc<NativeBackend>) -> DenseTensor<f64> {
    tensor(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2], b)
}

fn approx_eq(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-10, "value mismatch: {x} vs {y}");
    }
}

#[test]
fn svd_uses_shared_host() {
    let b = own();
    let t = mat23(&b);
    let (u, s, vt) = t.svd(1).unwrap();
    assert_shared_authority(&u, &b);
    assert_shared_authority(&s, &b);
    assert_shared_authority(&vt, &b);
    let (ru, rs, rvt) = svd_with_backend(&shared(), &t, 1).unwrap();
    approx_eq(u.data().data(), ru.data().data());
    approx_eq(s.data().data(), rs.data().data());
    approx_eq(vt.data().data(), rvt.data().data());
}

#[test]
fn trunc_svd_uses_shared_host() {
    let b = own();
    let t = mat23(&b);
    let params = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let (u, s, vt, err) = t.trunc_svd(1, &params).unwrap();
    assert_shared_authority(&u, &b);
    assert_shared_authority(&s, &b);
    assert_shared_authority(&vt, &b);
    let (ru, rs, rvt, rerr) = trunc_svd_with_backend(&shared(), &t, 1, &params).unwrap();
    approx_eq(u.data().data(), ru.data().data());
    approx_eq(s.data().data(), rs.data().data());
    approx_eq(vt.data().data(), rvt.data().data());
    assert!((err - rerr).abs() < 1e-10);
}

#[test]
fn qr_uses_shared_host() {
    let b = own();
    let t = mat23(&b);
    let (q, r) = t.qr(1).unwrap();
    assert_shared_authority(&q, &b);
    assert_shared_authority(&r, &b);
    let (rq, rr) = qr_with_backend(&shared(), &t, 1).unwrap();
    approx_eq(q.data().data(), rq.data().data());
    approx_eq(r.data().data(), rr.data().data());
}

#[test]
fn lq_uses_shared_host() {
    let b = own();
    let t = mat23(&b);
    let (l, q) = t.lq(1).unwrap();
    assert_shared_authority(&l, &b);
    assert_shared_authority(&q, &b);
    let (rl, rq) = lq_with_backend(&shared(), &t, 1).unwrap();
    approx_eq(l.data().data(), rl.data().data());
    approx_eq(q.data().data(), rq.data().data());
}

#[test]
fn eigh_uses_shared_host() {
    let b = own();
    let t = sym2(&b);
    let (w, v) = t.eigh(1).unwrap();
    assert_shared_authority(&w, &b);
    assert_shared_authority(&v, &b);
    let (rw, rv) = eigh_with_backend(&shared(), &t, 1).unwrap();
    approx_eq(w.data().data(), rw.data().data());
    approx_eq(v.data().data(), rv.data().data());
}

#[test]
fn eigvalsh_uses_shared_host() {
    let b = own();
    let t = sym2(&b);
    let w = t.eigvalsh(1).unwrap();
    assert_shared_authority(&w, &b);
    let rw = eigvalsh_with_backend(&shared(), &t, 1).unwrap();
    approx_eq(w.data().data(), rw.data().data());
}

/// Eigenvalues as (re, im) pairs sorted for order-insensitive comparison.
fn sorted_eigvals(t: &DenseTensor<Complex<f64>>) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = t.data().data().iter().map(|z| (z.re, z.im)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn approx_eq_pairs(a: &[(f64, f64)], b: &[(f64, f64)]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!(
            (x.0 - y.0).abs() < 1e-10 && (x.1 - y.1).abs() < 1e-10,
            "eigenvalue mismatch: {x:?} vs {y:?}"
        );
    }
}

#[test]
fn eig_uses_shared_host() {
    let b = own();
    let t = mat22(&b);
    let (w, v) = t.eig(1).unwrap();
    assert_shared_authority(&w, &b);
    assert_shared_authority(&v, &b);
    // Order-insensitive eigenvalue comparison against the explicit path.
    let (rw, _rv) = eig_with_backend(&shared(), &t, 1).unwrap();
    approx_eq_pairs(&sorted_eigvals(&w), &sorted_eigvals(&rw));
    // Eigenvectors verified by residual A v - w v = 0 per pair (column-major
    // 2x2: A[i][k] = a[k * 2 + i], v[:, j] = vd[j * 2..][..2]).
    let a = t.data().data();
    let wd = w.data().data();
    let vd = v.data().data();
    for j in 0..2 {
        for i in 0..2 {
            let av = a[i] * vd[2 * j] + a[2 + i] * vd[2 * j + 1];
            let res = av - wd[j] * vd[2 * j + i];
            assert!(res.norm() < 1e-10, "residual too large: {res}");
        }
    }
}

#[test]
fn eigvals_uses_shared_host() {
    let b = own();
    let t = mat22(&b);
    let w = t.eigvals(1).unwrap();
    assert_shared_authority(&w, &b);
    let rw = eigvals_with_backend(&shared(), &t, 1).unwrap();
    approx_eq_pairs(&sorted_eigvals(&w), &sorted_eigvals(&rw));
}

#[test]
fn contract_uses_shared_host() {
    let b = own();
    let lhs = mat22(&b);
    let rhs = mat22(&b);
    let out = lhs.contract(&rhs, "ab,bc->ac").unwrap();
    assert_shared_authority(&out, &b);
    let r = contract_with_backend(&shared(), &lhs, &rhs, "ab,bc->ac").unwrap();
    approx_eq(out.data().data(), r.data().data());
}

#[test]
fn transpose_uses_shared_host() {
    let b = own();
    let t = mat23(&b);
    let out = t.transpose(&[1, 0]).unwrap();
    assert_shared_authority(&out, &b);
    let r = transpose_with_backend(&shared(), &t, &[1, 0]).unwrap();
    assert_eq!(out.data().data(), r.data().data());
}

#[test]
fn trace_uses_shared_host() {
    let b = own();
    let t = mat22(&b);
    let out = t.trace(&[(0, 1)]).unwrap();
    assert_shared_authority(&out, &b);
    let r = trace_with_backend(&shared(), &t, &[(0, 1)]).unwrap();
    assert_eq!(out.data().data(), r.data().data());
}

#[test]
fn diag_uses_shared_host() {
    let b = own();
    let t = tensor(vec![1.0, 2.0, 3.0], vec![3], &b);
    let out = t.diag().unwrap();
    assert_shared_authority(&out, &b);
    let r = diag_with_backend(&shared(), &t).unwrap();
    assert_eq!(out.data().data(), r.data().data());
}

#[test]
fn diagonal_scale_uses_shared_host() {
    let b = own();
    let t = mat23(&b);
    let weights = [10.0, 20.0];
    let out = t.diagonal_scale(&weights, 0).unwrap();
    assert_shared_authority(&out, &b);
    let r = diagonal_scale_with_backend(&shared(), &t, &weights, 0).unwrap();
    assert_eq!(out.data().data(), r.data().data());
}

#[test]
fn solve_uses_shared_host() {
    let b = own();
    let a = mat22(&b);
    let rhs = mat22(&b);
    let out = a.solve(&rhs, 1).unwrap();
    assert_shared_authority(&out, &b);
    let r = solve_with_backend(&shared(), &a, &rhs, 1).unwrap();
    approx_eq(out.data().data(), r.data().data());
}

#[test]
fn inverse_uses_shared_host() {
    let b = own();
    let t = mat22(&b);
    let out = t.inverse(1).unwrap();
    assert_shared_authority(&out, &b);
    let r = inverse_with_backend(&shared(), &t, 1).unwrap();
    approx_eq(out.data().data(), r.data().data());
}

#[test]
fn expm_uses_shared_host() {
    let b = own();
    let t = sym2(&b);
    let out = t.expm(1).unwrap();
    assert_shared_authority(&out, &b);
    let r = expm_with_backend(&shared(), &t, 1).unwrap();
    approx_eq(out.data().data(), r.data().data());
}

#[test]
fn expm_hermitian_uses_shared_host() {
    let b = own();
    let t = sym2(&b);
    let out = t.expm_hermitian(1).unwrap();
    assert_shared_authority(&out, &b);
    let r = expm_hermitian_with_backend(&shared(), &t, 1).unwrap();
    approx_eq(out.data().data(), r.data().data());
}

#[test]
fn expm_antihermitian_uses_shared_host() {
    let b = own();
    // expm_antihermitian requires a complex element type; a real
    // anti-symmetric matrix embedded in the complex field is anti-Hermitian.
    let z = |re: f64| Complex::new(re, 0.0);
    let t: DenseTensor<Complex<f64>> =
        DenseTensor::from_raw_parts(vec![z(0.0), z(1.0), z(-1.0), z(0.0)], vec![2, 2], b.clone());
    let out = t.expm_antihermitian(1).unwrap();
    assert_shared_authority(&out, &b);
    let r = expm_antihermitian_with_backend(&shared(), &t, 1).unwrap();
    for (x, y) in out.data().data().iter().zip(r.data().data()) {
        assert!((x - y).norm() < 1e-10, "value mismatch: {x} vs {y}");
    }
}

// --- Block-sparse surface ---------------------------------------------------

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

/// Rank-2 tensor pinned to backend `b` (built in `b`'s preferred order).
fn rank2(b: &Arc<NativeBackend>) -> BlockSparseTensor<f64, U1Sector> {
    BlockSparseTensor::with_backend(rank2_data(b.preferred_order()), b.clone())
}

/// Compare two block-sparse tensors' joined data block by block, exactly.
fn bsp_eq(a: &BlockSparseTensor<f64, U1Sector>, b: &BlockSparseTensor<f64, U1Sector>) {
    let (da, db) = (a.data(), b.data());
    assert_eq!(da.shape(), db.shape(), "shape mismatch");
    assert_eq!(da.num_blocks(), db.num_blocks(), "block count mismatch");
    for meta in da.block_metas() {
        let xa = da.block_data(&meta.coord).unwrap();
        let xb = db.block_data(&meta.coord).unwrap();
        assert_eq!(xa, xb, "block {:?} mismatch", meta.coord);
    }
}

/// Like [`bsp_eq`] but with tolerance, for FP-reduction results.
fn bsp_approx_eq(a: &BlockSparseTensor<f64, U1Sector>, b: &BlockSparseTensor<f64, U1Sector>) {
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
fn bsp_svd_uses_shared_host() {
    let b = own();
    let t = rank2(&b);
    let (u, _s, vt) = t.svd(1).unwrap();
    assert_shared_authority(&u, &b);
    assert_shared_authority(&vt, &b);
    let (ru, _rs, rvt) = svd_block_sparse_with_backend(&shared(), &t, 1).unwrap();
    bsp_approx_eq(&u, &ru);
    bsp_approx_eq(&vt, &rvt);
}

#[test]
fn bsp_trunc_svd_uses_shared_host() {
    let b = own();
    let t = rank2(&b);
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let (u, _s, vt, _err) = t.trunc_svd(1, &params).unwrap();
    assert_shared_authority(&u, &b);
    assert_shared_authority(&vt, &b);
    let (ru, _rs, rvt, _rerr) =
        trunc_svd_block_sparse_with_backend(&shared(), &t, 1, &params).unwrap();
    bsp_approx_eq(&u, &ru);
    bsp_approx_eq(&vt, &rvt);
}

#[test]
fn bsp_qr_uses_shared_host() {
    let b = own();
    let t = rank2(&b);
    let (q, r) = t.qr(1).unwrap();
    assert_shared_authority(&q, &b);
    assert_shared_authority(&r, &b);
    let (rq, rr) = qr_block_sparse_with_backend(&shared(), &t, 1).unwrap();
    bsp_approx_eq(&q, &rq);
    bsp_approx_eq(&r, &rr);
}

#[test]
fn bsp_lq_uses_shared_host() {
    let b = own();
    let t = rank2(&b);
    let (l, q) = t.lq(1).unwrap();
    assert_shared_authority(&l, &b);
    assert_shared_authority(&q, &b);
    let (rl, rq) = lq_block_sparse_with_backend(&shared(), &t, 1).unwrap();
    bsp_approx_eq(&l, &rl);
    bsp_approx_eq(&q, &rq);
}

#[test]
fn bsp_contract_uses_shared_host() {
    let b = own();
    // t1's In leg (axis 1) contracts with t2's Out leg (axis 0): matching
    // sectors, opposite direction.
    let t1 = rank2(&b);
    let t2 = rank2(&b);
    let out = t1.contract(&t2, &[1], &[0]).unwrap();
    let r = contract_block_sparse_with_backend(&shared(), &t1, &t2, &[1], &[0]).unwrap();
    match (out, r) {
        (BlockSparseContractResult::Tensor(t), BlockSparseContractResult::Tensor(rt)) => {
            assert_shared_authority(&t, &b);
            bsp_approx_eq(&t, &rt);
        }
        _ => panic!("expected a tensor result"),
    }
}

#[test]
fn bsp_permute_uses_shared_host() {
    let b = own();
    let t = rank2(&b);
    let out = t.permute(&[1, 0]).unwrap();
    assert_shared_authority(&out, &b);
    bsp_eq(
        &out,
        &permute_block_sparse_with_backend(&shared(), &t, &[1, 0]).unwrap(),
    );
}

#[test]
fn bsp_fuse_legs_uses_shared_host() {
    let b = own();
    let t = rank2(&b);
    let out = t.fuse_legs(0, 2, Direction::Out).unwrap();
    assert_shared_authority(&out, &b);
    bsp_eq(
        &out,
        &fuse_legs_block_sparse_with_backend(&shared(), &t, 0, 2, Direction::Out).unwrap(),
    );
}

#[test]
fn bsp_diagonal_scale_uses_shared_host() {
    let b = own();
    let t = rank2(&b);
    // Derive a valid weight set (singular values) and a tensor with the bond
    // on axis 0 from a legacy SVD of `t` — the legacy path pins `vt` to the
    // receiver's own backend `b`, which is what the authority check needs.
    let (_u, sv, vt) = svd_block_sparse(&t, 1).unwrap();
    let out = vt.diagonal_scale(&sv, 0).unwrap();
    assert_shared_authority(&out, &b);
    bsp_eq(
        &out,
        &diagonal_scale_block_sparse_with_backend(&shared(), &vt, &sv, 0).unwrap(),
    );
}

#[test]
fn bsp_mismatched_layout_order_is_rejected() {
    // Fabricate a tensor whose layout (row-major) disagrees with the shared
    // Host's preferred order (column-major). The method path must inherit the
    // explicit-backend release-active check and reject it.
    let b = own();
    assert_eq!(shared().preferred_order(), MemoryOrder::ColumnMajor);
    let t = BlockSparseTensor::with_backend(rank2_data(MemoryOrder::RowMajor), b);
    let err = t.svd(1).unwrap_err();
    assert!(
        matches!(err, LinalgError::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}
