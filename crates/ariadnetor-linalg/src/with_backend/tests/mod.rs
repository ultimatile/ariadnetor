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

use ariadnetor_core::backend::ExecPolicy;
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::DenseTensor;
use ariadnetor_tensor::{ComputeBackendTensorExt, Host};

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
fn eigvals_eq(a: &[ariadnetor_core::Complex<f64>], b: &[ariadnetor_core::Complex<f64>]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let key = |z: &ariadnetor_core::Complex<f64>| (z.re, z.im);
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
    let (u, s, vt) = svd(&rec, &t, 1).unwrap();
    assert!(
        total_recorded(&rec) > 0,
        "passed backend must run the kernel"
    );
    let (hu, hs, hvt) = svd(&host, &t, 1).unwrap();
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
    let (u, s, vt, err) = trunc_svd(&rec, &t, 1, &params).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hu, hs, hvt, herr) = trunc_svd(&host, &t, 1, &params).unwrap();
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
    let (q, r) = qr(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hq, hr) = qr(&host, &t, 1).unwrap();
    approx_eq(q.data().data(), hq.data().data());
    approx_eq(r.data().data(), hr.data().data());
}

#[test]
fn lq_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat23();
    let (l, q) = lq(&rec, &t, 1).unwrap();
    assert!(total_recorded(&rec) > 0);
    let (hl, hq) = lq(&host, &t, 1).unwrap();
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
    let out = contract(&rec, &lhs, &rhs, "ab,bc->ac").unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = contract(&host, &lhs, &rhs, "ab,bc->ac").unwrap();
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
fn permute_routes_to_passed_backend() {
    let rec = RecordingBackend::new();
    let host = NativeBackend::new();
    let t = mat23();
    let out = permute_with_backend(&rec, &t, &[1, 0]).unwrap();
    assert!(total_recorded(&rec) > 0);
    let hout = permute_with_backend(&host, &t, &[1, 0]).unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

/// An invalid `perm` must surface as `LinalgError::InvalidArgument` rather than
/// an index-out-of-bounds panic. `mat23` has rank 2, so a length-3 perm is too
/// long, axis 5 is out of range, and `[0, 0]` duplicates axis 0. The
/// auto-policy entry point routes through the shared `transpose_inner`
/// validation.
#[test]
fn permute_with_backend_rejects_invalid_perm() {
    let host = NativeBackend::new();
    let t = mat23();

    let err = permute_with_backend(&host, &t, &[0, 1, 2]).unwrap_err();
    assert!(matches!(err, LinalgError::InvalidArgument(_)));
    assert!(err.to_string().contains("perm length"));

    let err = permute_with_backend(&host, &t, &[0, 5]).unwrap_err();
    assert!(err.to_string().contains("out of range"));

    let err = permute_with_backend(&host, &t, &[0, 0]).unwrap_err();
    assert!(err.to_string().contains("duplicate"));
}

/// Expert-layer counterpart: the explicit-policy `expert::permute` shares the
/// same `transpose_inner` chokepoint, so the same three invalid perms are
/// rejected identically.
#[test]
fn expert_permute_rejects_invalid_perm() {
    let host = NativeBackend::new();
    let t = mat23();

    let err = expert::permute(&host, &t, &[0, 1, 2], ExecPolicy::Sequential).unwrap_err();
    assert!(matches!(err, LinalgError::InvalidArgument(_)));
    assert!(err.to_string().contains("perm length"));

    let err = expert::permute(&host, &t, &[0, 5], ExecPolicy::Sequential).unwrap_err();
    assert!(err.to_string().contains("out of range"));

    let err = expert::permute(&host, &t, &[0, 0], ExecPolicy::Sequential).unwrap_err();
    assert!(err.to_string().contains("duplicate"));
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
    use ariadnetor_core::Complex;
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
    let out = diagonal_scale(&rec, &t, &weights, 0).unwrap();
    let hout = diagonal_scale(&host, &t, &weights, 0).unwrap();
    approx_eq(out.data().data(), hout.data().data());
}

/// Dense counterpart of the block-sparse `expert::*` policy-forwarding tests:
/// the tensor-keyed `expert::svd` carries the caller's `ExecPolicy` through to
/// the dense kernel descriptor. Pairs with
/// `expert_svd_bsp_forwards_explicit_policy` so both public decomposition
/// surfaces have explicit-policy forwarding coverage.
#[test]
fn expert_svd_dense_forwards_explicit_policy() {
    let rec = RecordingBackend::new();
    let _ = expert::svd(&rec, &mat23(), 1, ExecPolicy::Parallel(0)).unwrap();
    assert_eq!(rec.svd_recorded(), vec![ExecPolicy::Parallel(0)]);
}
