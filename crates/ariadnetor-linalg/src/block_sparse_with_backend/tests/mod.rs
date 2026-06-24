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

use arnet_core::Complex;
use arnet_core::backend::{ExecPolicy, MemoryOrder};
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
    let (u, _s, vt) = svd(&rec, &t, 1).unwrap();
    assert!(
        total_recorded(&rec) > 0,
        "passed backend must run the kernel"
    );
    let (hu, _hs, hvt) = svd(&host, &t, 1).unwrap();
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
    let (u, _s, vt, _err) = trunc_svd(&rec, &t, 1, &params).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hu, _hs, hvt, _herr) = trunc_svd(&host, &t, 1, &params).unwrap();
    bsp_eq(&u, &hu);
    bsp_eq(&vt, &hvt);
}

#[test]
fn qr_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2();
    let (q, r) = qr(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hq, hr) = qr(&host, &t, 1).unwrap();
    bsp_eq(&q, &hq);
    bsp_eq(&r, &hr);
}

#[test]
fn lq_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2();
    let (l, q) = lq(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hl, hq) = lq(&host, &t, 1).unwrap();
    bsp_eq(&l, &hl);
    bsp_eq(&q, &hq);
}

/// Rank-2 symmetric (Hermitian) U1 tensor, identity flux, in the recording
/// backend's order. Symmetric data is order-agnostic in flat form.
fn hermitian_rank2() -> BlockSparseTensor<f64, U1Sector> {
    let order = RecordingBackend::new().preferred_order();
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0), order);
    bs.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[2.0, 1.0, 1.0, 3.0]);
    bs.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[5.0, 1.0, 0.0, 1.0, 6.0, 2.0, 0.0, 2.0, 7.0]);
    BlockSparseTensor::from_data(bs)
}

fn eigenvalues_eq(a: &BlockScalars<f64, U1Sector>, b: &BlockScalars<f64, U1Sector>) {
    assert_eq!(a.values.len(), b.values.len(), "sector count mismatch");
    for ((sa, va), (sb, vb)) in a.values.iter().zip(&b.values) {
        assert_eq!(sa, sb, "sector mismatch");
        assert_eq!(va.len(), vb.len(), "eigenvalue count mismatch");
        for (x, y) in va.iter().zip(vb) {
            assert!((x - y).abs() < 1e-10, "eigenvalue mismatch: {x} vs {y}");
        }
    }
}

#[test]
fn eigh_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = hermitian_rank2();
    let (w, v) = eigh_block_sparse_with_backend(&rec, &t, 1).unwrap();
    let policies = rec.eigh_policies.lock().unwrap().clone();
    assert!(
        !policies.is_empty(),
        "passed backend must run the eigh kernel"
    );
    // The auto path pins per-sector Sequential.
    assert!(policies.iter().all(|p| matches!(p, ExecPolicy::Sequential)));
    let (hw, hv) = eigh_block_sparse_with_backend(&host, &t, 1).unwrap();
    eigenvalues_eq(&w, &hw);
    bsp_eq(&v, &hv);
}

#[test]
fn eigvalsh_matches_eigh() {
    let host = NativeBackend::new();
    let t = hermitian_rank2();
    let (w, _v) = eigh_block_sparse_with_backend(&host, &t, 1).unwrap();
    let w_only = eigvalsh_block_sparse_with_backend(&host, &t, 1).unwrap();
    eigenvalues_eq(&w, &w_only);
}

#[test]
fn eigh_host_method_matches_with_backend() {
    let host = NativeBackend::new();
    let t = hermitian_rank2();
    let (w_backend, v_backend) = eigh_block_sparse_with_backend(&host, &t, 1).unwrap();
    let (w_method, v_method) = t.eigh(1).unwrap();
    eigenvalues_eq(&w_method, &w_backend);
    bsp_eq(&v_method, &v_backend);
}

fn eig_eigenvalues_eq(
    a: &BlockScalars<Complex<f64>, U1Sector>,
    b: &BlockScalars<Complex<f64>, U1Sector>,
) {
    assert_eq!(a.values.len(), b.values.len(), "sector count mismatch");
    for ((sa, va), (sb, vb)) in a.values.iter().zip(&b.values) {
        assert_eq!(sa, sb, "sector mismatch");
        assert_eq!(va.len(), vb.len(), "eigenvalue count mismatch");
        for (x, y) in va.iter().zip(vb) {
            assert!((x - y).norm() < 1e-10, "eigenvalue mismatch: {x} vs {y}");
        }
    }
}

/// Block-by-block equality for complex eigenvector tensors.
fn bsp_eq_complex(
    a: &BlockSparseTensor<Complex<f64>, U1Sector>,
    b: &BlockSparseTensor<Complex<f64>, U1Sector>,
) {
    let (da, db) = (a.data(), b.data());
    assert_eq!(da.shape(), db.shape(), "shape mismatch");
    assert_eq!(da.num_blocks(), db.num_blocks(), "block count mismatch");
    for meta in da.block_metas() {
        let xa = da.block_data(&meta.coord).unwrap();
        let xb = db.block_data(&meta.coord).unwrap();
        assert_eq!(xa.len(), xb.len());
        for (x, y) in xa.iter().zip(xb) {
            assert!((x - y).norm() < 1e-10, "value mismatch: {x} vs {y}");
        }
    }
}

#[test]
fn eig_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    // `rank2()` is non-symmetric, so this genuinely exercises the general path.
    let t = rank2();
    let (w, v) = eig_block_sparse_with_backend(&rec, &t, 1).unwrap();
    let policies = rec.eig_policies.lock().unwrap().clone();
    assert!(
        !policies.is_empty(),
        "passed backend must run the eig kernel"
    );
    // The auto path pins per-sector Sequential.
    assert!(policies.iter().all(|p| matches!(p, ExecPolicy::Sequential)));
    let (hw, hv) = eig_block_sparse_with_backend(&host, &t, 1).unwrap();
    eig_eigenvalues_eq(&w, &hw);
    bsp_eq_complex(&v, &hv);
}

#[test]
fn eigvals_matches_eig() {
    let host = NativeBackend::new();
    let t = rank2();
    let (w, _v) = eig_block_sparse_with_backend(&host, &t, 1).unwrap();
    let w_only = eigvals_block_sparse_with_backend(&host, &t, 1).unwrap();
    eig_eigenvalues_eq(&w, &w_only);
}

#[test]
fn eig_host_method_matches_with_backend() {
    let host = NativeBackend::new();
    let t = rank2();
    let (w_backend, v_backend) = eig_block_sparse_with_backend(&host, &t, 1).unwrap();
    let (w_method, v_method) = t.eig(1).unwrap();
    eig_eigenvalues_eq(&w_method, &w_backend);
    bsp_eq_complex(&v_method, &v_backend);
}

/// Complex rank-2 tensor in the recording backend's order. The data is not
/// constructed to be anti-Hermitian — the `expm_antihermitian` routing /
/// delegation tests only require the two paths to agree, and kernel-level
/// numerical correctness is covered by the `block_sparse_expm` unit tests.
fn rank2_c() -> BlockSparseTensor<Complex<f64>, U1Sector> {
    let order = RecordingBackend::new().preferred_order();
    let c = |re: f64, im: f64| Complex::new(re, im);
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);
    let mut bs =
        BlockSparseTensorData::<Complex<f64>, U1Sector>::zeros(vec![row, col], U1Sector(0), order);
    bs.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[c(0.0, 1.0), c(1.0, 1.0), c(-1.0, 1.0), c(0.0, 2.0)]);
    bs.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[c(0.0, 3.0)]);
    BlockSparseTensor::from_data(bs)
}

#[test]
fn expm_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2();
    let r = expm_block_sparse_with_backend(&rec, &t, 1).unwrap();
    assert!(
        total_recorded(&rec) > 0,
        "passed backend must run the expm kernel"
    );
    let hr = expm_block_sparse_with_backend(&host, &t, 1).unwrap();
    bsp_eq(&r, &hr);
}

#[test]
fn expm_host_method_matches_with_backend() {
    let host = NativeBackend::new();
    let t = rank2();
    let backend = expm_block_sparse_with_backend(&host, &t, 1).unwrap();
    let method = t.expm(1).unwrap();
    bsp_eq(&method, &backend);
}

#[test]
fn expm_hermitian_host_method_matches_with_backend() {
    let host = NativeBackend::new();
    let t = hermitian_rank2();
    let backend = expm_hermitian_block_sparse_with_backend(&host, &t, 1).unwrap();
    let method = t.expm_hermitian(1).unwrap();
    bsp_eq(&method, &backend);
}

#[test]
fn expm_antihermitian_host_method_matches_with_backend() {
    let host = NativeBackend::new();
    let t = rank2_c();
    let backend = expm_antihermitian_block_sparse_with_backend(&host, &t, 1).unwrap();
    let method = t.expm_antihermitian(1).unwrap();
    bsp_eq_complex(&method, &backend);
}

#[test]
fn expm_hermitian_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = hermitian_rank2();
    let r = expm_hermitian_block_sparse_with_backend(&rec, &t, 1).unwrap();
    assert!(
        total_recorded(&rec) > 0,
        "passed backend must run the expm_hermitian kernel"
    );
    let hr = expm_hermitian_block_sparse_with_backend(&host, &t, 1).unwrap();
    bsp_eq(&r, &hr);
}

#[test]
fn expm_antihermitian_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = rank2_c();
    let r = expm_antihermitian_block_sparse_with_backend(&rec, &t, 1).unwrap();
    assert!(
        total_recorded(&rec) > 0,
        "passed backend must run the expm_antihermitian kernel"
    );
    let hr = expm_antihermitian_block_sparse_with_backend(&host, &t, 1).unwrap();
    bsp_eq_complex(&r, &hr);
}

/// Whether two real block-sparse tensors coincide block-by-block within `1e-10`.
fn bsp_coincides(
    a: &BlockSparseTensor<f64, U1Sector>,
    b: &BlockSparseTensor<f64, U1Sector>,
) -> bool {
    let (da, db) = (a.data(), b.data());
    da.block_metas().iter().all(|meta| {
        let (xa, xb) = (
            da.block_data(&meta.coord).unwrap(),
            db.block_data(&meta.coord).unwrap(),
        );
        xa.iter().zip(xb).all(|(x, y)| (x - y).abs() < 1e-10)
    })
}

#[test]
fn expm_hermitian_is_not_cross_wired_to_expm() {
    // On a Hermitian input the general and Hermitian exponentials coincide, so
    // `expm_hermitian_host_method_matches_with_backend` (Hermitian fixture)
    // cannot detect a host-method delegation swap. A non-Hermitian input
    // discriminates: the Hermitian host method must match its Hermitian twin
    // (eigh-based, trusting hermiticity) and differ from the general exponential.
    let host = NativeBackend::new();
    let t = rank2(); // non-symmetric, so the Hermitian and general paths diverge
    let method = t.expm_hermitian(1).unwrap();
    let twin = expm_hermitian_block_sparse_with_backend(&host, &t, 1).unwrap();
    bsp_eq(&method, &twin);
    let general = expm_block_sparse_with_backend(&host, &t, 1).unwrap();
    assert!(
        !bsp_coincides(&method, &general),
        "hermitian path unexpectedly equals the general exponential on a \
         non-Hermitian input; the fixture cannot discriminate a delegation swap"
    );
}

#[test]
fn expm_antihermitian_host_rejects_real_input() {
    // The anti-Hermitian host method rejects real element types; its siblings
    // (`expm` / `expm_hermitian`) accept them, so this discriminates that the
    // host method delegates to the anti-Hermitian twin and not a sibling.
    let t = rank2();
    let err = t.expm_antihermitian(1).unwrap_err();
    assert!(
        matches!(err, LinalgError::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

#[test]
fn contract_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    // t1's In leg (axis 1) contracts with t2's Out leg (axis 0): matching
    // sectors, opposite direction.
    let t1 = rank2();
    let t2 = rank2();
    let out = tensordot(&rec, &t1, &t2, &[1], &[0]).unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = tensordot(&host, &t1, &t2, &[1], &[0]).unwrap();
    bsp_eq(&out, &hout);
}

#[test]
fn contract_rejects_notation_arity_mismatch() {
    // A notation naming fewer axes than the operand rank must error, not
    // silently treat the undeclared axes as free (which would yield an outer
    // product). rank2() is rank 2; "a,b->ab" declares one axis per operand.
    let host = NativeBackend::new();
    let t = rank2();
    let err = contract(&host, &t, &t, "a,b->ab").unwrap_err();
    assert!(
        matches!(err, LinalgError::InvalidArgument(_)),
        "expected InvalidArgument for arity mismatch, got {err:?}"
    );
}

#[test]
fn contract_free_output_reorder_matches_tensordot_then_permute() {
    // The dispatched `contract` reorders the natural tensordot output legs into
    // the notation's requested order. In production this path is only reached
    // once the MPS / DMRG bodies merge (#302); validate it here against the
    // explicit composition it is defined to equal — a natural-order tensordot
    // followed by `permute_block_sparse`.
    let host = NativeBackend::new();
    let t1 = rank2();
    let t2 = rank2();

    // Contract t1's In leg (axis 1) against t2's Out leg (axis 0). The natural
    // output order is [t1_free(0), t2_free(1)] = "ac"; the notation requests the
    // swapped order "ca", so the block-sparse reorder pass must fire.
    let reordered = contract(&host, &t1, &t2, "ab,bc->ca").unwrap();

    let natural = tensordot(&host, &t1, &t2, &[1], &[0]).unwrap();
    let permuted = permute_block_sparse_with_backend(&host, &natural, &[1, 0]).unwrap();
    bsp_eq(&reordered, &permuted);

    // Guard against a silent no-op: the reordered result must differ from the
    // natural one (the diagonal blocks are transposed, not identical).
    let (nat, reo) = (natural.data(), reordered.data());
    let differs = nat.block_metas().iter().any(|m| {
        let a = nat.block_data(&m.coord).unwrap();
        match reo.block_data(&m.coord) {
            Some(b) => a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-9),
            None => true,
        }
    });
    assert!(
        differs,
        "reorder produced the natural order; the permute pass did not fire"
    );
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
    let (_u, sv, vt) = svd(&host, &t, 1).unwrap();
    let out = diagonal_scale_block_sparse_with_backend(&rec, &vt, &sv, 0).unwrap();
    bsp_eq(
        &out,
        &diagonal_scale_block_sparse_with_backend(&host, &vt, &sv, 0).unwrap(),
    );
}

// ---------------------------------------------------------------------------
// Layout-keyed dispatch: block-sparse policy routing through `expert::*`.
//
// `expert::svd` / `qr` / `lq` / `trunc_svd` are the first public entries that
// pin an `ExecPolicy` on a block-sparse decomposition — the auto-policy
// crate-root forms keep block-sparse on `Sequential`. These tests prove the
// dispatch wrapper carries the caller's policy through to every per-sector
// descriptor (the `rank2` fixture has two sectors), and that the auto form
// pins `Sequential`. The recorder registers one descriptor per sector.
// ---------------------------------------------------------------------------

fn assert_all_eq(got: &[ExecPolicy], want: ExecPolicy, op: &str) {
    assert!(!got.is_empty(), "{op}: expected per-sector calls, got none");
    for (i, p) in got.iter().enumerate() {
        assert_eq!(*p, want, "{op}: sector #{i} forwarded {p:?}, want {want:?}");
    }
}

#[test]
fn expert_svd_bsp_forwards_explicit_policy() {
    let rec = RecordingBackend::new();
    let _ = expert::svd(&rec, &rank2(), 1, ExecPolicy::Parallel(0)).unwrap();
    assert_all_eq(&rec.svd_recorded(), ExecPolicy::Parallel(0), "expert::svd");
}

#[test]
fn expert_qr_bsp_forwards_explicit_policy() {
    let rec = RecordingBackend::new();
    let _ = expert::qr(&rec, &rank2(), 1, ExecPolicy::Parallel(0)).unwrap();
    assert_all_eq(&rec.qr_recorded(), ExecPolicy::Parallel(0), "expert::qr");
}

#[test]
fn expert_lq_bsp_forwards_explicit_policy() {
    let rec = RecordingBackend::new();
    let _ = expert::lq(&rec, &rank2(), 1, ExecPolicy::Parallel(0)).unwrap();
    assert_all_eq(&rec.lq_recorded(), ExecPolicy::Parallel(0), "expert::lq");
}

#[test]
fn expert_trunc_svd_bsp_forwards_explicit_policy() {
    let rec = RecordingBackend::new();
    let params = TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: None,
    };
    let _ = expert::trunc_svd(&rec, &rank2(), 1, &params, ExecPolicy::Parallel(0)).unwrap();
    assert_all_eq(
        &rec.svd_recorded(),
        ExecPolicy::Parallel(0),
        "expert::trunc_svd",
    );
}

#[test]
fn auto_svd_bsp_pins_sequential() {
    let rec = RecordingBackend::new();
    let _ = svd(&rec, &rank2(), 1).unwrap();
    assert_all_eq(&rec.svd_recorded(), ExecPolicy::Sequential, "svd (auto)");
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
    let err = svd(&backend, &t, 1).unwrap_err();
    assert!(
        matches!(err, LinalgError::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}
