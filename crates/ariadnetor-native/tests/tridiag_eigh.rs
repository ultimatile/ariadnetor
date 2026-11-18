//! Descriptor-length validation for the real symmetric tridiagonal
//! eigensolver.

use ariadnetor_core::backend::{
    BackendError, ComputeBackend, ExecPolicy, MemoryOrder, TridiagEighDescriptor,
};
use ariadnetor_native::NativeBackend;

const N: usize = 3;

/// Run `tridiag_eigh` at `n = N` with the given `d` / `e` / `w` / `v`
/// lengths; the valid lengths are `N` / `N - 1` / `N` / `N * N`.
fn tridiag_eigh_with_lengths(
    d_len: usize,
    e_len: usize,
    w_len: usize,
    v_len: usize,
) -> Result<(), BackendError> {
    let d = vec![2.0_f64; d_len];
    let e = vec![1.0_f64; e_len];
    let mut w = vec![0.0_f64; w_len];
    let mut v = vec![0.0_f64; v_len];
    NativeBackend::new().tridiag_eigh(TridiagEighDescriptor {
        n: N,
        d: &d,
        e: &e,
        w: &mut w,
        v: &mut v,
        order: MemoryOrder::ColumnMajor,
        policy: ExecPolicy::Sequential,
    })
}

#[track_caller]
fn assert_invalid_argument(result: Result<(), BackendError>) {
    assert!(
        matches!(result, Err(BackendError::InvalidArgument(_))),
        "expected InvalidArgument, got {result:?}"
    );
}

#[test]
fn tridiag_eigh_accepts_consistent_lengths() {
    tridiag_eigh_with_lengths(N, N - 1, N, N * N).unwrap();
}

#[test]
fn tridiag_eigh_rejects_wrong_diagonal_length() {
    assert_invalid_argument(tridiag_eigh_with_lengths(N - 1, N - 1, N, N * N));
}

#[test]
fn tridiag_eigh_rejects_wrong_subdiagonal_length() {
    assert_invalid_argument(tridiag_eigh_with_lengths(N, N, N, N * N));
}

#[test]
fn tridiag_eigh_rejects_wrong_eigenvalue_buffer_length() {
    assert_invalid_argument(tridiag_eigh_with_lengths(N, N - 1, N + 1, N * N));
}

#[test]
fn tridiag_eigh_rejects_wrong_eigenvector_buffer_length() {
    assert_invalid_argument(tridiag_eigh_with_lengths(N, N - 1, N, N * N - 1));
}
