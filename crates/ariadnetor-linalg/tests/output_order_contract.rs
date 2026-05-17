//! Contract tests: linalg operations return data in backend.preferred_order().
//!
//! These tests verify the representation invariant that all linalg outputs
//! are in `backend.preferred_order()` (ColumnMajor for NativeBackend).
//! The invariant is checked by feeding linalg outputs directly into other
//! linalg operations without manual reorder — if an output were in the
//! wrong order, the downstream operation would produce numerically wrong
//! results.
//!
//! Motivation: `Dense::order()` records each tensor's flat-data layout,
//! but downstream backend kernels still expect the active backend's
//! preferred order. These tests pin the metadata-vs-data consistency
//! at the linalg output boundary so a regression where an op tags its
//! output with the wrong order — or produces bytes in a different
//! order than its `Dense::order()` claims — surfaces as a numerical
//! mismatch rather than silently propagating downstream.

use arnet_linalg::{
    contract_dense as contract, diagonal_scale_dense as diagonal_scale, expm_dense as expm,
    inverse_dense as inverse, solve_dense as solve, svd_dense as svd, transpose_dense as transpose,
};
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, MemoryOrder, reorder};

/// Create Dense from conceptual row-major data, converted to CM for NativeBackend.
fn cm(data: Vec<f64>, shape: Vec<usize>) -> Dense<f64> {
    let rm = Dense::new(data, shape, MemoryOrder::RowMajor);
    reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)
}

// =========================================================================
// Contract A: Output order consistency
//
// Each test feeds the output of one linalg op directly into another.
// If the first op returns wrong-order data, the second op will
// misinterpret it and produce incorrect results.
// =========================================================================

#[test]
fn contract_output_feeds_into_contract() {
    // Use rectangular matrices so RM/CM mixups cannot hide as a simple transpose.
    // C = A * B (2×3 @ 3×2 → 2×2), then verify C * A = A * (B * A)
    let backend = NativeBackend::new();
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = cm(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

    let c = contract(&backend, &a, &b, "ij,jk->ik").unwrap(); // 2×2
    // Feed contract output (c) directly into another contract — no manual reorder
    let lhs = contract(&backend, &c, &a, "ij,jk->ik").unwrap(); // 2×3

    let ba = contract(&backend, &b, &a, "ij,jk->ik").unwrap(); // 3×3
    let rhs = contract(&backend, &a, &ba, "ij,jk->ik").unwrap(); // 2×3

    for (l, r) in lhs.data().iter().zip(rhs.data()) {
        assert!(
            (l - r).abs() < 1e-8,
            "contract output order mismatch: (A*B)*A != A*(B*A)"
        );
    }
}

#[test]
fn solve_output_feeds_into_contract() {
    // solve(A, B) returns X such that A*X = B
    // Verify by feeding X directly into contract (no manual reorder)
    let backend = NativeBackend::new();
    let a = cm(vec![2.0, 1.0, 5.0, 3.0], vec![2, 2]);
    let b = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    let x = solve(&backend, &a, &b, 1).unwrap();
    let ax = contract(&backend, &a, &x, "ij,jk->ik").unwrap();

    for (computed, expected) in ax.data().iter().zip(b.data()) {
        assert!(
            (computed - expected).abs() < 1e-10,
            "solve output fed to contract: A*X != B"
        );
    }
}

#[test]
fn inverse_output_feeds_into_contract() {
    // A * A^{-1} = I
    let backend = NativeBackend::new();
    let a = cm(vec![2.0, 1.0, 5.0, 3.0], vec![2, 2]);

    let a_inv = inverse(&backend, &a, 1).unwrap();
    let product = contract(&backend, &a, &a_inv, "ij,jk->ik").unwrap();

    // Expected: identity in CM = [1, 0, 0, 1]
    let identity = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    for (p, e) in product.data().iter().zip(identity.data()) {
        assert!(
            (p - e).abs() < 1e-10,
            "inverse output fed to contract: A*A^{{-1}} != I"
        );
    }
}

#[test]
fn svd_output_feeds_into_contract() {
    // SVD: A = U * diag(S) * Vt
    // Verify by feeding U, S, Vt directly into contract and diagonal_scale
    let backend = NativeBackend::new();
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let (u, s, vt) = svd(&backend, &a, 1).unwrap();

    // U * diag(S) * Vt — feed SVD outputs directly, no manual reorder
    let us = diagonal_scale(&backend, &u, s.data(), u.rank() - 1).unwrap();
    let reconstructed = contract(&backend, &us, &vt, "ij,jk->ik").unwrap();

    for (r, orig) in reconstructed.data().iter().zip(a.data()) {
        assert!(
            (r - orig).abs() < 1e-10,
            "SVD output fed to contract: U*S*Vt != A"
        );
    }
}

#[test]
fn transpose_output_feeds_into_contract() {
    // (A^T)^T = A
    let backend = NativeBackend::new();
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let at = transpose(&backend, &a, &[1, 0]).unwrap();
    let att = transpose(&backend, &at, &[1, 0]).unwrap();

    // Feed transpose output into transpose — roundtrip should recover original
    for (original, roundtrip) in a.data().iter().zip(att.data()) {
        assert!(
            (original - roundtrip).abs() < 1e-15,
            "transpose roundtrip: (A^T)^T != A"
        );
    }
}

#[test]
fn expm_output_feeds_into_contract() {
    // exp(A) * exp(-A) = I for any square matrix A
    let backend = NativeBackend::new();
    // Small Hermitian matrix so expm converges well
    let a = cm(vec![0.1, 0.2, 0.2, 0.3], vec![2, 2]);
    let neg_a = Dense::new(
        a.data().iter().map(|&x: &f64| -x).collect(),
        a.shape().to_vec(),
        MemoryOrder::ColumnMajor,
    );

    let exp_a = expm(&backend, &a, 1).unwrap();
    let exp_neg_a = expm(&backend, &neg_a, 1).unwrap();

    // Feed expm output directly into contract — no manual reorder
    let product = contract(&backend, &exp_a, &exp_neg_a, "ij,jk->ik").unwrap();

    // Expected: identity in CM
    let identity = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    for (p, e) in product.data().iter().zip(identity.data()) {
        assert!(
            (p - e).abs() < 1e-10,
            "expm output fed to contract: exp(A)*exp(-A) != I"
        );
    }
}
