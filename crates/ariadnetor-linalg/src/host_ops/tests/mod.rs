//! Tests for the Host-defaulting method surface.
//!
//! Equivalence: each method must match its `*_with_backend` twin run on the
//! Host substrate. Tensors no longer carry a backend, so the former
//! "result carries the shared singleton, not the receiver's instance"
//! pointer-identity invariant is structurally unviolable and is dropped;
//! what remains to prove is that the sugar delegates to the right twin with
//! the right arguments. Each method's output is therefore compared against an
//! independent `NativeBackend::new()` run of its twin — both dispatch on the
//! same Host substrate and execution policy, so data-movement ops compare
//! exactly while FP-reduction (faer-backed) ops compare with tolerance.
//! `eig` / `eigvals` results are compared order-insensitively (the backend
//! contract promises no eigenvalue ordering), with eigenvectors verified by
//! residual.

use ariadnetor_core::Complex;
use ariadnetor_core::backend::MemoryOrder;
use ariadnetor_tensor::test_fixtures::square_legs;
use ariadnetor_tensor::{
    BlockCoord, BlockSparseTensor, BlockSparseTensorData, DenseTensor, Direction, NativeBackend,
    U1Sector,
};
use ariadnetor_tensor::{ComputeBackendTensorExt, Host};

use crate::*;

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> DenseTensor<f64> {
    Host::shared().dense(data, shape)
}

fn sym2() -> DenseTensor<f64> {
    // Symmetric, so eigh / expm_hermitian have a real spectrum.
    tensor(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2])
}

fn mat23() -> DenseTensor<f64> {
    tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
}

fn mat22() -> DenseTensor<f64> {
    tensor(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2])
}

fn approx_eq(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-10, "value mismatch: {x} vs {y}");
    }
}

#[test]
fn svd_matches_twin() {
    let host = NativeBackend::new();
    let t = mat23();
    let (u, s, vt) = t.svd(1).unwrap();
    let (ru, rs, rvt) = svd(&host, &t, 1).unwrap();
    approx_eq(u.data().data(), ru.data().data());
    approx_eq(s.data().data(), rs.data().data());
    approx_eq(vt.data().data(), rvt.data().data());
}

#[test]
fn trunc_svd_matches_twin() {
    let host = NativeBackend::new();
    let t = mat23();
    let params = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let (u, s, vt, err) = t.trunc_svd(1, &params).unwrap();
    let (ru, rs, rvt, rerr) = trunc_svd(&host, &t, 1, &params).unwrap();
    approx_eq(u.data().data(), ru.data().data());
    approx_eq(s.data().data(), rs.data().data());
    approx_eq(vt.data().data(), rvt.data().data());
    assert!((err - rerr).abs() < 1e-10);
}

#[test]
fn qr_matches_twin() {
    let host = NativeBackend::new();
    let t = mat23();
    let (q, r) = t.qr(1).unwrap();
    let (rq, rr) = qr(&host, &t, 1).unwrap();
    approx_eq(q.data().data(), rq.data().data());
    approx_eq(r.data().data(), rr.data().data());
}

#[test]
fn lq_matches_twin() {
    let host = NativeBackend::new();
    let t = mat23();
    let (l, q) = t.lq(1).unwrap();
    let (rl, rq) = lq(&host, &t, 1).unwrap();
    approx_eq(l.data().data(), rl.data().data());
    approx_eq(q.data().data(), rq.data().data());
}

#[test]
fn eigh_matches_twin() {
    let host = NativeBackend::new();
    let t = sym2();
    let (w, v) = t.eigh(1).unwrap();
    let (rw, rv) = eigh_with_backend(&host, &t, 1).unwrap();
    approx_eq(w.data().data(), rw.data().data());
    approx_eq(v.data().data(), rv.data().data());
}

#[test]
fn eigvalsh_matches_twin() {
    let host = NativeBackend::new();
    let t = sym2();
    let w = t.eigvalsh(1).unwrap();
    let rw = eigvalsh_with_backend(&host, &t, 1).unwrap();
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
fn eig_matches_twin() {
    let host = NativeBackend::new();
    let t = mat22();
    let (w, v) = t.eig(1).unwrap();
    // Order-insensitive eigenvalue comparison against the explicit path.
    let (rw, _rv) = eig_with_backend(&host, &t, 1).unwrap();
    approx_eq_pairs(&sorted_eigvals(&w), &sorted_eigvals(&rw));
    // Eigenvectors verified by residual A v - w v = 0 per pair (column-major
    // 2x2: A[i][k] = a[k * 2 + i], v[:, j] = vd[j * 2..][..2]). A zero vector
    // would satisfy the residual vacuously, so each column must be nonzero.
    let a = t.data().data();
    let wd = w.data().data();
    let vd = v.data().data();
    for j in 0..2 {
        let norm = (vd[2 * j].norm_sqr() + vd[2 * j + 1].norm_sqr()).sqrt();
        assert!(norm > 1e-6, "eigenvector {j} must be nonzero, norm {norm}");
        for i in 0..2 {
            let av = a[i] * vd[2 * j] + a[2 + i] * vd[2 * j + 1];
            let res = av - wd[j] * vd[2 * j + i];
            assert!(res.norm() < 1e-10, "residual too large: {res}");
        }
    }
}

#[test]
fn eigvals_matches_twin() {
    let host = NativeBackend::new();
    let t = mat22();
    let w = t.eigvals(1).unwrap();
    let rw = eigvals_with_backend(&host, &t, 1).unwrap();
    approx_eq_pairs(&sorted_eigvals(&w), &sorted_eigvals(&rw));
}

#[test]
fn contract_matches_twin() {
    let host = NativeBackend::new();
    let lhs = mat22();
    let rhs = mat22();
    let out = lhs.contract(&rhs, "ab,bc->ac").unwrap();
    let r = contract(&host, &lhs, &rhs, "ab,bc->ac").unwrap();
    approx_eq(out.data().data(), r.data().data());
}

#[test]
fn permute_matches_twin() {
    let host = NativeBackend::new();
    let t = mat23();
    let out = t.permute(&[1, 0]).unwrap();
    let r = permute_with_backend(&host, &t, &[1, 0]).unwrap();
    assert_eq!(out.data().data(), r.data().data());
}

#[test]
fn trace_matches_twin() {
    let host = NativeBackend::new();
    let t = mat22();
    let out = t.trace(&[(0, 1)]).unwrap();
    let r = trace_with_backend(&host, &t, &[(0, 1)]).unwrap();
    assert_eq!(out.data().data(), r.data().data());
}

#[test]
fn diag_matches_twin() {
    let host = NativeBackend::new();
    let t = tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let out = t.diag().unwrap();
    let r = diag_with_backend(&host, &t).unwrap();
    assert_eq!(out.data().data(), r.data().data());
}

#[test]
fn diagonal_scale_matches_twin() {
    let host = NativeBackend::new();
    let t = mat23();
    let weights = [10.0, 20.0];
    let out = t.diagonal_scale(&weights, 0).unwrap();
    let r = diagonal_scale(&host, &t, &weights, 0).unwrap();
    assert_eq!(out.data().data(), r.data().data());
}

#[test]
fn solve_matches_twin() {
    let host = NativeBackend::new();
    let a = mat22();
    let rhs = mat22();
    let out = a.solve(&rhs, 1).unwrap();
    let r = solve_with_backend(&host, &a, &rhs, 1).unwrap();
    approx_eq(out.data().data(), r.data().data());
}

#[test]
fn inverse_matches_twin() {
    let host = NativeBackend::new();
    let t = mat22();
    let out = t.inverse(1).unwrap();
    let r = inverse_with_backend(&host, &t, 1).unwrap();
    approx_eq(out.data().data(), r.data().data());
}

#[test]
fn expm_matches_twin() {
    let host = NativeBackend::new();
    // Non-symmetric input: the Hermitian sibling would compute a different
    // (triangle-trusting) result here, so cross-wiring fails the equivalence.
    let t = mat22();
    let out = t.expm(1).unwrap();
    let r = expm_with_backend(&host, &t, 1).unwrap();
    approx_eq(out.data().data(), r.data().data());
}

#[test]
fn expm_hermitian_matches_twin() {
    let host = NativeBackend::new();
    let t = sym2();
    let out = t.expm_hermitian(1).unwrap();
    let r = expm_hermitian_with_backend(&host, &t, 1).unwrap();
    approx_eq(out.data().data(), r.data().data());
}

#[test]
fn expm_hermitian_is_not_cross_wired_to_expm() {
    // On a Hermitian input the general and Hermitian exponentials coincide,
    // so the equivalence test above cannot detect a delegation swap. A
    // non-Hermitian input discriminates: the Hermitian path trusts
    // hermiticity (eigh-based), so the method's outcome must match the
    // Hermitian twin's — and must differ from the general exponential.
    let host = NativeBackend::new();
    let t = mat22();
    let method = t.expm_hermitian(1);
    let twin = expm_hermitian_with_backend(&host, &t, 1);
    match (method, twin) {
        (Ok(m), Ok(tw)) => {
            approx_eq(m.data().data(), tw.data().data());
            let general = expm_with_backend(&host, &t, 1).unwrap();
            let coincide = m
                .data()
                .data()
                .iter()
                .zip(general.data().data())
                .all(|(x, y)| (x - y).abs() < 1e-10);
            assert!(
                !coincide,
                "hermitian path unexpectedly equals the general exponential \
                 on a non-Hermitian input; the fixture cannot discriminate"
            );
        }
        // Both rejecting identically also proves the method follows the
        // Hermitian twin (the general twin succeeds on this input).
        (Err(_), Err(_)) => {}
        (m, tw) => panic!("method/twin disagree on non-Hermitian input: {m:?} vs {tw:?}"),
    }
}

#[test]
fn expm_antihermitian_rejects_real_input() {
    // The anti-Hermitian path rejects real element types; its siblings
    // accept them, so this discriminates the delegation target.
    let t = mat22();
    let err = t.expm_antihermitian(1).unwrap_err();
    assert!(
        matches!(err, LinalgError::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

#[test]
fn expm_antihermitian_matches_twin() {
    let host = NativeBackend::new();
    // expm_antihermitian requires a complex element type; a real
    // anti-symmetric matrix embedded in the complex field is anti-Hermitian.
    let z = |re: f64| Complex::new(re, 0.0);
    let t: DenseTensor<Complex<f64>> =
        Host::shared().dense(vec![z(0.0), z(1.0), z(-1.0), z(0.0)], vec![2, 2]);
    let out = t.expm_antihermitian(1).unwrap();
    let r = expm_antihermitian_with_backend(&host, &t, 1).unwrap();
    for (x, y) in out.data().data().iter().zip(r.data().data()) {
        assert!((x - y).norm() < 1e-10, "value mismatch: {x} vs {y}");
    }
}

// --- Block-sparse surface ---------------------------------------------------

/// Rank-2 U1 data, flux 0: Out(0:2, 1:3), In(0:2, 1:3), laid out in `order`.
fn rank2_data(order: MemoryOrder) -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 3)]),
        U1Sector(0),
        order,
    );
    bs.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    bs.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    bs
}

/// Rank-2 tensor laid out in the shared Host's preferred order.
fn rank2() -> BlockSparseTensor<f64, U1Sector> {
    BlockSparseTensor::from_data(rank2_data(NativeBackend::new().preferred_order()))
}

/// Compare two block-sparse tensors' structural metadata: shape, block
/// count, flux, memory order, and indices (sectors / dims / directions;
/// `QNIndex` has no `PartialEq`, so indices compare via `Debug`).
fn bsp_meta_eq(a: &BlockSparseTensor<f64, U1Sector>, b: &BlockSparseTensor<f64, U1Sector>) {
    let (da, db) = (a.data(), b.data());
    assert_eq!(da.shape(), db.shape(), "shape mismatch");
    assert_eq!(da.num_blocks(), db.num_blocks(), "block count mismatch");
    assert_eq!(da.flux(), db.flux(), "flux mismatch");
    assert_eq!(da.order(), db.order(), "order mismatch");
    assert_eq!(
        format!("{:?}", da.indices()),
        format!("{:?}", db.indices()),
        "indices mismatch"
    );
}

/// Compare two block-sparse tensors' metadata and data block by block,
/// exactly.
fn bsp_eq(a: &BlockSparseTensor<f64, U1Sector>, b: &BlockSparseTensor<f64, U1Sector>) {
    bsp_meta_eq(a, b);
    let (da, db) = (a.data(), b.data());
    for meta in da.block_metas() {
        let xa = da.block_data(&meta.coord).unwrap();
        let xb = db.block_data(&meta.coord).unwrap();
        assert_eq!(xa, xb, "block {:?} mismatch", meta.coord);
    }
}

/// Like [`bsp_eq`] but with tolerance, for FP-reduction results.
fn bsp_approx_eq(a: &BlockSparseTensor<f64, U1Sector>, b: &BlockSparseTensor<f64, U1Sector>) {
    bsp_meta_eq(a, b);
    let (da, db) = (a.data(), b.data());
    for meta in da.block_metas() {
        let xa = da.block_data(&meta.coord).unwrap();
        let xb = db.block_data(&meta.coord).unwrap();
        assert_eq!(xa.len(), xb.len());
        for (x, y) in xa.iter().zip(xb) {
            assert!((x - y).abs() < 1e-10, "value mismatch: {x} vs {y}");
        }
    }
}

/// Compare two singular-value sets sector by sector, with tolerance.
fn sv_approx_eq(a: &BlockScalars<f64, U1Sector>, b: &BlockScalars<f64, U1Sector>) {
    assert_eq!(a.values.len(), b.values.len(), "sector count mismatch");
    for ((sa, va), (sb, vb)) in a.values.iter().zip(&b.values) {
        assert_eq!(sa, sb, "sector mismatch");
        assert_eq!(va.len(), vb.len(), "singular-value count mismatch");
        for (x, y) in va.iter().zip(vb) {
            assert!((x - y).abs() < 1e-10, "singular value mismatch: {x} vs {y}");
        }
    }
}

#[test]
fn bsp_svd_matches_twin() {
    let host = NativeBackend::new();
    let t = rank2();
    let (u, s, vt) = t.svd(1).unwrap();
    let (ru, rs, rvt) = svd(&host, &t, 1).unwrap();
    bsp_approx_eq(&u, &ru);
    sv_approx_eq(&s, &rs);
    bsp_approx_eq(&vt, &rvt);
}

#[test]
fn bsp_trunc_svd_matches_twin() {
    let host = NativeBackend::new();
    let t = rank2();
    let params = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let (u, s, vt, err) = t.trunc_svd(1, &params).unwrap();
    let (ru, rs, rvt, rerr) = trunc_svd(&host, &t, 1, &params).unwrap();
    bsp_approx_eq(&u, &ru);
    sv_approx_eq(&s, &rs);
    bsp_approx_eq(&vt, &rvt);
    assert!((err - rerr).abs() < 1e-10, "truncation error mismatch");
}

#[test]
fn bsp_qr_matches_twin() {
    let host = NativeBackend::new();
    let t = rank2();
    let (q, r) = t.qr(1).unwrap();
    let (rq, rr) = qr(&host, &t, 1).unwrap();
    bsp_approx_eq(&q, &rq);
    bsp_approx_eq(&r, &rr);
}

#[test]
fn bsp_lq_matches_twin() {
    let host = NativeBackend::new();
    let t = rank2();
    let (l, q) = t.lq(1).unwrap();
    let (rl, rq) = lq(&host, &t, 1).unwrap();
    bsp_approx_eq(&l, &rl);
    bsp_approx_eq(&q, &rq);
}

#[test]
fn bsp_contract_matches_twin() {
    let host = NativeBackend::new();
    // t1's In leg (axis 1) contracts with t2's Out leg (axis 0): matching
    // sectors, opposite direction.
    let t1 = rank2();
    let t2 = rank2();
    // t1's In leg (axis 1) ↔ t2's Out leg (axis 0): a_{ab} b_{bc} -> ab_{ac}.
    let out = t1.contract(&t2, "ab,bc->ac").unwrap();
    let r = tensordot(&host, &t1, &t2, &[1], &[0]).unwrap();
    bsp_approx_eq(&out, &r);
}

#[test]
fn bsp_permute_matches_twin() {
    let host = NativeBackend::new();
    let t = rank2();
    let out = t.permute(&[1, 0]).unwrap();
    bsp_eq(
        &out,
        &permute_block_sparse_with_backend(&host, &t, &[1, 0]).unwrap(),
    );
}

#[test]
fn bsp_fuse_legs_matches_twin() {
    let host = NativeBackend::new();
    let t = rank2();
    let out = t.fuse_legs(0, 2, Direction::Out).unwrap();
    bsp_eq(
        &out,
        &fuse_legs_block_sparse_with_backend(&host, &t, 0, 2, Direction::Out).unwrap(),
    );
}

#[test]
fn bsp_diagonal_scale_matches_twin() {
    let host = NativeBackend::new();
    let t = rank2();
    // Derive a valid weight set (singular values) and a tensor with the bond
    // on axis 0 from an SVD of `t`.
    let (_u, sv, vt) = t.svd(1).unwrap();
    let out = vt.diagonal_scale(&sv, 0).unwrap();
    bsp_eq(&out, &diagonal_scale(&host, &vt, &sv, 0).unwrap());
}

#[test]
fn bsp_mismatched_layout_order_is_rejected() {
    // Fabricate a tensor whose layout (row-major) disagrees with the shared
    // Host's preferred order (column-major). The method path must inherit the
    // explicit-backend release-active check and reject it.
    assert_eq!(
        NativeBackend::new().preferred_order(),
        MemoryOrder::ColumnMajor
    );
    let t = BlockSparseTensor::from_data(rank2_data(MemoryOrder::RowMajor));
    let err = t.svd(1).unwrap_err();
    assert!(
        matches!(err, LinalgError::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}
