//! Authority-routing tests for the dense explicit-backend paths.
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
//! dropped. For the allocation-only ops (`trace`, `diag`, `diagonal_scale`)
//! the backend drives no observable kernel, so only numerical correctness is
//! checked; the behaviorally-distinct routing proof is the Stage C
//! pluggability litmus.

use arnet_native::NativeBackend;
use arnet_tensor::DenseTensor;
use arnet_tensor::{ComputeBackendTensorExt, Host};

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

/// Compare two complex eigenvalue multisets order-insensitively: the backend
/// contract promises no eigenvalue ordering, so sort both by `(re, im)` before
/// the element-wise tolerance check.
fn eigvals_eq(a: &[arnet_core::Complex<f64>], b: &[arnet_core::Complex<f64>]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let key = |z: &arnet_core::Complex<f64>| (z.re, z.im);
    let mut a: Vec<_> = a.to_vec();
    let mut b: Vec<_> = b.to_vec();
    a.sort_by(|x, y| key(x).partial_cmp(&key(y)).unwrap());
    b.sort_by(|x, y| key(x).partial_cmp(&key(y)).unwrap());
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).norm() < 1e-10, "value mismatch: {x} vs {y}");
    }
}

#[test]
fn svd_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat23();
    let (u, s, vt) = svd_with_backend(&rec, &t, 1).unwrap();
    assert!(
        total_recorded(&rec) > 0,
        "passed backend must run the kernel"
    );
    let (hu, hs, hvt) = svd_with_backend(&host, &t, 1).unwrap();
    approx_eq(u.data().data(), hu.data().data());
    approx_eq(s.data().data(), hs.data().data());
    approx_eq(vt.data().data(), hvt.data().data());
}

#[test]
fn trunc_svd_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat23();
    let params = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let (u, s, vt, err) = trunc_svd_with_backend(&rec, &t, 1, &params).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hu, hs, hvt, herr) = trunc_svd_with_backend(&host, &t, 1, &params).unwrap();
    approx_eq(u.data().data(), hu.data().data());
    approx_eq(s.data().data(), hs.data().data());
    approx_eq(vt.data().data(), hvt.data().data());
    assert!((err - herr).abs() < 1e-10);
}

#[test]
fn qr_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat23();
    let (q, r) = qr_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hq, hr) = qr_with_backend(&host, &t, 1).unwrap();
    approx_eq(q.data().data(), hq.data().data());
    approx_eq(r.data().data(), hr.data().data());
}

#[test]
fn lq_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat23();
    let (l, q) = lq_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hl, hq) = lq_with_backend(&host, &t, 1).unwrap();
    approx_eq(l.data().data(), hl.data().data());
    approx_eq(q.data().data(), hq.data().data());
}

#[test]
fn eigh_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = sym2();
    let (w, v) = eigh_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hw, hv) = eigh_with_backend(&host, &t, 1).unwrap();
    approx_eq(w.data().data(), hw.data().data());
    approx_eq(v.data().data(), hv.data().data());
}

#[test]
fn eigvalsh_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = sym2();
    let w = eigvalsh_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    approx_eq(
        w.data().data(),
        eigvalsh_with_backend(&host, &t, 1).unwrap().data().data(),
    );
}

#[test]
fn eig_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat22();
    let (w, _v) = eig_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hw, _hv) = eig_with_backend(&host, &t, 1).unwrap();
    eigvals_eq(w.data().data(), hw.data().data());
}

#[test]
fn eigvals_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat22();
    let w = eigvals_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let hw = eigvals_with_backend(&host, &t, 1).unwrap();
    eigvals_eq(w.data().data(), hw.data().data());
}

#[test]
fn contract_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let lhs = mat22();
    let rhs = mat22();
    let out = contract_with_backend(&rec, &lhs, &rhs, "ab,bc->ac").unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = contract_with_backend(&host, &lhs, &rhs, "ab,bc->ac").unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

#[test]
fn einsum_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let lhs = mat22();
    let rhs = mat22();
    let out = einsum_with_backend(&rec, &[&lhs, &rhs], "ab,bc->ac").unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = einsum_with_backend(&host, &[&lhs, &rhs], "ab,bc->ac").unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

#[test]
fn transpose_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat23();
    let out = transpose_with_backend(&rec, &t, &[1, 0]).unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = transpose_with_backend(&host, &t, &[1, 0]).unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

#[test]
fn solve_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let a = mat22();
    let b = mat22();
    let out = solve_with_backend(&rec, &a, &b, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = solve_with_backend(&host, &a, &b, 1).unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

#[test]
fn inverse_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat22();
    let out = inverse_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = inverse_with_backend(&host, &t, 1).unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

#[test]
fn expm_hermitian_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = sym2();
    let out = expm_hermitian_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = expm_hermitian_with_backend(&host, &t, 1).unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

#[test]
fn expm_antihermitian_routes_to_passed_backend() {
    use arnet_core::Complex;
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    // expm_antihermitian requires a complex element type; a real anti-symmetric
    // matrix embedded in the complex field is anti-Hermitian.
    let z = |re: f64| Complex::new(re, 0.0);
    let data = vec![z(0.0), z(1.0), z(-1.0), z(0.0)];
    let t: DenseTensor<Complex<f64>> = Host::shared().dense(data, vec![2, 2]);
    let out = expm_antihermitian_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = expm_antihermitian_with_backend(&host, &t, 1).unwrap();
    for (x, y) in out.data().data().iter().zip(hout.data().data()) {
        assert!((x - y).norm() < 1e-10, "value mismatch: {x} vs {y}");
    }
}

#[test]
fn expm_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = sym2();
    let out = expm_with_backend(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = expm_with_backend(&host, &t, 1).unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

// --- Allocation-only ops: the backend drives no observable kernel, so only
// numerical correctness is checked here; the behaviorally-distinct routing
// proof is the Stage C pluggability litmus. ---

#[test]
fn trace_matches_host() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat22();
    let out = trace_with_backend(&rec, &t, &[(0, 1)]).unwrap();
    let hout = trace_with_backend(&host, &t, &[(0, 1)]).unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

#[test]
fn diag_matches_host() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let out = diag_with_backend(&rec, &t).unwrap();
    let hout = diag_with_backend(&host, &t).unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

#[test]
fn diagonal_scale_matches_host() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat23();
    let weights = [10.0, 20.0];
    let out = diagonal_scale_with_backend(&rec, &t, &weights, 0).unwrap();
    let hout = diagonal_scale_with_backend(&host, &t, &weights, 0).unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

/// `diagonal_scale_with_backend` admits non-`Scalar` element types
/// (`T: Clone + Mul`), unlike the `OpsFor`-gated kernel twins. This locks that
/// contract: a `Scalar`-keyed `OpsFor` bound here would make `i32` fail to
/// compile, so the test guards against re-tightening the bound.
#[test]
fn diagonal_scale_supports_non_scalar_elements() {
    let host = NativeBackend::new();
    let t = Host::shared().dense(vec![1, 2, 3, 4], vec![2, 2]);
    let out = diagonal_scale_with_backend(&host, &t, &[10, 100], 0).unwrap();
    // The contract under test is that non-`Scalar` `T` compiles and runs; the
    // value check is order-agnostic (the scaled multiset is layout-invariant):
    // {1,3} on row 0 ×10 and {2,4} on row 1 ×100 → {10, 30, 200, 400}.
    let mut got = out.data().data().to_vec();
    got.sort_unstable();
    assert_eq!(got, vec![10, 30, 200, 400]);
}
