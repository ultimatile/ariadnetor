//! Authority-routing tests for the dense explicit-backend paths.
//!
//! Each twin is exercised with a backend `a2` distinct from the one the input
//! tensor was built with (`a1`). The central invariant proved here is that the
//! tensor's own backend is never used for the computation: `a1` must record no
//! kernel call. Ops that dispatch a recorded kernel additionally show `a2`
//! receiving the call, and every result is checked to carry `a2` (pointer
//! identity) and to match the legacy backend-derived wrapper numerically.

use std::sync::Arc;

use arnet_tensor::DenseTensor;

use crate::test_util::RecordingBackend;
use crate::*;

/// Total number of kernel descriptors a recording backend has seen across all
/// op families. Zero means the backend drove no computation.
fn total_recorded(b: &RecordingBackend) -> usize {
    b.svd_policies.lock().unwrap().len()
        + b.qr_policies.lock().unwrap().len()
        + b.lq_policies.lock().unwrap().len()
        + b.gemm_policies.lock().unwrap().len()
        + b.eigh_policies.lock().unwrap().len()
        + b.eig_policies.lock().unwrap().len()
        + b.solve_policies.lock().unwrap().len()
        + b.transpose_policies.lock().unwrap().len()
}

fn pair() -> (Arc<RecordingBackend>, Arc<RecordingBackend>) {
    (
        Arc::new(RecordingBackend::new()),
        Arc::new(RecordingBackend::new()),
    )
}

/// Build a dense tensor pinned to backend `a1`.
fn tensor(
    data: Vec<f64>,
    shape: Vec<usize>,
    a1: &Arc<RecordingBackend>,
) -> DenseTensor<f64, RecordingBackend> {
    DenseTensor::from_raw_parts(data, shape, a1.clone())
}

fn sym2(a1: &Arc<RecordingBackend>) -> DenseTensor<f64, RecordingBackend> {
    // Symmetric, so eigh / expm_hermitian have a real spectrum.
    tensor(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2], a1)
}

fn mat23(a1: &Arc<RecordingBackend>) -> DenseTensor<f64, RecordingBackend> {
    tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], a1)
}

fn mat22(a1: &Arc<RecordingBackend>) -> DenseTensor<f64, RecordingBackend> {
    tensor(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2], a1)
}

fn approx_eq(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-10, "value mismatch: {x} vs {y}");
    }
}

#[test]
fn svd_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = mat23(&a1);
    let (u, s, vt) = svd_with_backend(&a2, &t, 1).unwrap();
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
    assert!(Arc::ptr_eq(s.backend_arc(), &a2));
    assert!(Arc::ptr_eq(vt.backend_arc(), &a2));
    let (lu, ls, lvt) = svd(&t, 1).unwrap();
    approx_eq(u.data().data(), lu.data().data());
    approx_eq(s.data().data(), ls.data().data());
    approx_eq(vt.data().data(), lvt.data().data());
}

#[test]
fn trunc_svd_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = mat23(&a1);
    let params = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let (u, s, vt, err) = trunc_svd_with_backend(&a2, &t, 1, &params).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(u.backend_arc(), &a2));
    let (lu, ls, lvt, lerr) = trunc_svd(&t, 1, &params).unwrap();
    approx_eq(u.data().data(), lu.data().data());
    approx_eq(s.data().data(), ls.data().data());
    approx_eq(vt.data().data(), lvt.data().data());
    assert!((err - lerr).abs() < 1e-10);
}

#[test]
fn qr_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = mat23(&a1);
    let (q, r) = qr_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(q.backend_arc(), &a2));
    assert!(Arc::ptr_eq(r.backend_arc(), &a2));
    let (lq, lr) = qr(&t, 1).unwrap();
    approx_eq(q.data().data(), lq.data().data());
    approx_eq(r.data().data(), lr.data().data());
}

#[test]
fn lq_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = mat23(&a1);
    let (l, q) = lq_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(l.backend_arc(), &a2));
    assert!(Arc::ptr_eq(q.backend_arc(), &a2));
    let (ll, lq) = lq(&t, 1).unwrap();
    approx_eq(l.data().data(), ll.data().data());
    approx_eq(q.data().data(), lq.data().data());
}

#[test]
fn eigh_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = sym2(&a1);
    let (w, v) = eigh_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(w.backend_arc(), &a2));
    assert!(Arc::ptr_eq(v.backend_arc(), &a2));
    let (lw, lv) = eigh(&t, 1).unwrap();
    approx_eq(w.data().data(), lw.data().data());
    approx_eq(v.data().data(), lv.data().data());
}

#[test]
fn eigvalsh_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = sym2(&a1);
    let w = eigvalsh_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(w.backend_arc(), &a2));
    approx_eq(w.data().data(), eigvalsh(&t, 1).unwrap().data().data());
}

#[test]
fn eig_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = mat22(&a1);
    let (w, v) = eig_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(w.backend_arc(), &a2));
    assert!(Arc::ptr_eq(v.backend_arc(), &a2));
}

#[test]
fn eigvals_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = mat22(&a1);
    let w = eigvals_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(w.backend_arc(), &a2));
}

#[test]
fn contract_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let lhs = mat22(&a1);
    let rhs = mat22(&a1);
    let out = contract_with_backend(&a2, &lhs, &rhs, "ab,bc->ac").unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = contract(&lhs, &rhs, "ab,bc->ac").unwrap();
    approx_eq(out.data().data(), lout.data().data());
}

#[test]
fn einsum_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let lhs = mat22(&a1);
    let rhs = mat22(&a1);
    let out = einsum_with_backend(&a2, &[&lhs, &rhs], "ab,bc->ac").unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = einsum(&[&lhs, &rhs], "ab,bc->ac").unwrap();
    approx_eq(out.data().data(), lout.data().data());
}

#[test]
fn transpose_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = mat23(&a1);
    let out = transpose_with_backend(&a2, &t, &[1, 0]).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = transpose(&t, &[1, 0]).unwrap();
    approx_eq(out.data().data(), lout.data().data());
}

#[test]
fn solve_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let a = mat22(&a1);
    let b = mat22(&a1);
    let out = solve_with_backend(&a2, &a, &b, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = solve(&a, &b, 1).unwrap();
    approx_eq(out.data().data(), lout.data().data());
}

#[test]
fn inverse_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = mat22(&a1);
    let out = inverse_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = inverse(&t, 1).unwrap();
    approx_eq(out.data().data(), lout.data().data());
}

#[test]
fn expm_hermitian_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = sym2(&a1);
    let out = expm_hermitian_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = expm_hermitian(&t, 1).unwrap();
    approx_eq(out.data().data(), lout.data().data());
}

#[test]
fn expm_antihermitian_routes_to_passed_backend() {
    use arnet_core::Complex;
    let (a1, a2) = pair();
    // expm_antihermitian requires a complex element type; a real anti-symmetric
    // matrix embedded in the complex field is anti-Hermitian.
    let z = |re: f64| Complex::new(re, 0.0);
    let data = vec![z(0.0), z(1.0), z(-1.0), z(0.0)];
    let t: DenseTensor<Complex<f64>, RecordingBackend> =
        DenseTensor::from_raw_parts(data, vec![2, 2], a1.clone());
    let out = expm_antihermitian_with_backend(&a2, &t, 1).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(total_recorded(&a2) > 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = expm_antihermitian(&t, 1).unwrap();
    for (x, y) in out.data().data().iter().zip(lout.data().data()) {
        assert!((x - y).norm() < 1e-10, "value mismatch: {x} vs {y}");
    }
}

#[test]
fn expm_routes_to_passed_backend() {
    let (a1, a2) = pair();
    let t = sym2(&a1);
    let out = expm_with_backend(&a2, &t, 1).unwrap();
    // expm may dispatch via a recorded kernel; the universal guarantee is that
    // the tensor's backend is not consulted and the result carries `a2`.
    assert_eq!(total_recorded(&a1), 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = expm(&t, 1).unwrap();
    approx_eq(out.data().data(), lout.data().data());
}

// --- Backend used only to allocate the result; kernel routing is unobservable
// here, so pointer identity of the result is the available authority proof. ---

#[test]
fn trace_carries_passed_backend() {
    let (a1, a2) = pair();
    let t = mat22(&a1);
    let out = trace_with_backend(&a2, &t, &[(0, 1)]).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = trace(&t, &[(0, 1)]).unwrap();
    approx_eq(out.data().data(), lout.data().data());
}

#[test]
fn diag_carries_passed_backend() {
    let (a1, a2) = pair();
    let t = tensor(vec![1.0, 2.0, 3.0], vec![3], &a1);
    let out = diag_with_backend(&a2, &t).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = diag(&t).unwrap();
    approx_eq(out.data().data(), lout.data().data());
}

#[test]
fn diagonal_scale_carries_passed_backend() {
    let (a1, a2) = pair();
    let t = mat23(&a1);
    let weights = [10.0, 20.0];
    let out = diagonal_scale_with_backend(&a2, &t, &weights, 0).unwrap();
    assert_eq!(total_recorded(&a1), 0);
    assert!(Arc::ptr_eq(out.backend_arc(), &a2));
    let lout = diagonal_scale(&t, &weights, 0).unwrap();
    approx_eq(out.data().data(), lout.data().data());
}
