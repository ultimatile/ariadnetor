//! Verify that the two-layer BSp API forwards `ExecPolicy` correctly
//! from the BSp wrapper down to every per-sector dense descriptor.
//!
//! Default BSp wrappers hardcode `Sequential`; `_with_policy` wrappers forward
//! the caller's policy to each per-sector dense `_with_policy` call. The
//! multi-sector fixture (`sample_u1_rank2`) produces two sectors so forwarding
//! is observed per-sector, not just once.

use arnet_core::backend::ExecPolicy;

use super::*;
use crate::test_util::RecordingBackend;

// A multi-sector rank-2 tensor: sectors (0,0) 2×2 and (1,1) 3×3 both
// yield non-trivial per-sector SVD/QR/LQ work — two entries per recorded list.
fn multi_sector() -> arnet_tensor::BlockSparseTensorData<f64, arnet_tensor::U1Sector> {
    super::sample_u1_rank2()
}

fn assert_all_eq(got: &[ExecPolicy], want: ExecPolicy, op: &str) {
    assert!(
        !got.is_empty(),
        "{op}: expected at least one recorded call, got zero"
    );
    for (i, p) in got.iter().enumerate() {
        assert_eq!(*p, want, "{op}: call #{i} forwarded {p:?}, want {want:?}");
    }
}

// ---- SVD ----------------------------------------------------------------

#[test]
fn svd_default_forwards_sequential() {
    let rec = RecordingBackend::new();
    let _ = svd_block_sparse_with_policy_dense(&rec, &multi_sector(), 1, ExecPolicy::Sequential)
        .unwrap();
    assert_all_eq(&rec.svd_recorded(), ExecPolicy::Sequential, "svd");
}

#[test]
fn svd_with_policy_forwards_parallel() {
    let rec = RecordingBackend::new();
    let _ = svd_block_sparse_with_policy_dense(&rec, &multi_sector(), 1, ExecPolicy::Parallel(0))
        .unwrap();
    assert_all_eq(&rec.svd_recorded(), ExecPolicy::Parallel(0), "svd");
}

// ---- Truncated SVD ------------------------------------------------------

#[test]
fn trunc_svd_default_forwards_sequential() {
    let rec = RecordingBackend::new();
    let params = TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: None,
    };
    let _ = trunc_svd_block_sparse_with_policy_dense(
        &rec,
        &multi_sector(),
        1,
        &params,
        ExecPolicy::Sequential,
    )
    .unwrap();
    assert_all_eq(&rec.svd_recorded(), ExecPolicy::Sequential, "trunc_svd");
}

#[test]
fn trunc_svd_with_policy_forwards_parallel() {
    let rec = RecordingBackend::new();
    let params = TruncSvdParams {
        chi_max: Some(3),
        target_trunc_err: None,
    };
    let _ = trunc_svd_block_sparse_with_policy_dense(
        &rec,
        &multi_sector(),
        1,
        &params,
        ExecPolicy::Parallel(0),
    )
    .unwrap();
    assert_all_eq(&rec.svd_recorded(), ExecPolicy::Parallel(0), "trunc_svd");
}

// ---- QR -----------------------------------------------------------------

#[test]
fn qr_default_forwards_sequential() {
    let rec = RecordingBackend::new();
    let _ = qr_block_sparse_with_policy_dense(&rec, &multi_sector(), 1, ExecPolicy::Sequential)
        .unwrap();
    assert_all_eq(&rec.qr_recorded(), ExecPolicy::Sequential, "qr");
}

#[test]
fn qr_with_policy_forwards_parallel() {
    let rec = RecordingBackend::new();
    let _ = qr_block_sparse_with_policy_dense(&rec, &multi_sector(), 1, ExecPolicy::Parallel(0))
        .unwrap();
    assert_all_eq(&rec.qr_recorded(), ExecPolicy::Parallel(0), "qr");
}

// ---- LQ -----------------------------------------------------------------

#[test]
fn lq_default_forwards_sequential() {
    let rec = RecordingBackend::new();
    let _ = lq_block_sparse_with_policy_dense(&rec, &multi_sector(), 1, ExecPolicy::Sequential)
        .unwrap();
    assert_all_eq(&rec.lq_recorded(), ExecPolicy::Sequential, "lq");
}

#[test]
fn lq_with_policy_forwards_parallel() {
    let rec = RecordingBackend::new();
    let _ = lq_block_sparse_with_policy_dense(&rec, &multi_sector(), 1, ExecPolicy::Parallel(0))
        .unwrap();
    assert_all_eq(&rec.lq_recorded(), ExecPolicy::Parallel(0), "lq");
}

// ---- Per-sector call count ---------------------------------------------
//
// A rank-2 two-sector fixture must produce exactly two per-sector calls.
// Guards against regressions where a wrapper only forwards to the first
// sector and hardcodes the rest.

#[test]
fn with_policy_reaches_every_sector() {
    let rec = RecordingBackend::new();
    let _ = svd_block_sparse_with_policy_dense(&rec, &multi_sector(), 1, ExecPolicy::Parallel(0))
        .unwrap();
    assert_eq!(
        rec.svd_recorded().len(),
        2,
        "expected one per-sector call per sector in the fixture"
    );
}
