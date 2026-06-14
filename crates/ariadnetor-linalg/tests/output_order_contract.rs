//! Contract tests: linalg operations return data in backend.preferred_order().
//!
//! These tests verify the representation invariant that all linalg outputs
//! are in `backend.preferred_order()` (ColumnMajor for NativeBackend).
//! The invariant is checked by feeding linalg outputs directly into other
//! linalg operations without manual reorder — if an output were in the
//! wrong order, the downstream operation would produce numerically wrong
//! results.
//!
//! Motivation: `DenseTensorData::order()` records each tensor's flat-data layout,
//! but downstream backend kernels still expect the active backend's
//! preferred order. These tests pin the metadata-vs-data consistency
//! at the linalg output boundary so a regression where an op tags its
//! output with the wrong order — or produces bytes in a different
//! order than its `DenseTensorData::order()` claims — surfaces as a numerical
//! mismatch rather than silently propagating downstream.

use arnet_linalg::DenseHostOps;
use arnet_tensor::{DenseTensor, DenseTensorData, MemoryOrder, reorder_data};

/// Build a `DenseTensor` from conceptual row-major data, reordered to CM.
fn cm(data: Vec<f64>, shape: Vec<usize>) -> DenseTensor<f64> {
    let rm = DenseTensorData::from_raw_parts(data, shape, MemoryOrder::RowMajor);
    let cm = reorder_data(&rm, MemoryOrder::ColumnMajor);
    DenseTensor::from_data(cm)
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
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = cm(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

    let c = a.contract(&b, "ij,jk->ik").unwrap(); // 2×2
    // Feed contract output (c) directly into another contract — no manual reorder
    let lhs = c.contract(&a, "ij,jk->ik").unwrap(); // 2×3

    let ba = b.contract(&a, "ij,jk->ik").unwrap(); // 3×3
    let rhs = a.contract(&ba, "ij,jk->ik").unwrap(); // 2×3

    for (l, r) in lhs.data_slice().iter().zip(rhs.data_slice()) {
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
    let a = cm(vec![2.0, 1.0, 5.0, 3.0], vec![2, 2]);
    let b = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    let x = a.solve(&b, 1).unwrap();
    let ax = a.contract(&x, "ij,jk->ik").unwrap();

    for (computed, expected) in ax.data_slice().iter().zip(b.data_slice()) {
        assert!(
            (computed - expected).abs() < 1e-10,
            "solve output fed to contract: A*X != B"
        );
    }
}

#[test]
fn inverse_output_feeds_into_contract() {
    // A * A^{-1} = I
    let a = cm(vec![2.0, 1.0, 5.0, 3.0], vec![2, 2]);

    let a_inv = a.inverse(1).unwrap();
    let product = a.contract(&a_inv, "ij,jk->ik").unwrap();

    // Expected: identity in CM = [1, 0, 0, 1]
    let identity = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    for (p, e) in product.data_slice().iter().zip(identity.data_slice()) {
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
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let (u, s, vt) = a.svd(1).unwrap();

    // U * diag(S) * Vt — feed SVD outputs directly, no manual reorder
    let us = u.diagonal_scale(s.data_slice(), u.rank() - 1).unwrap();
    let reconstructed = us.contract(&vt, "ij,jk->ik").unwrap();

    for (r, orig) in reconstructed.data_slice().iter().zip(a.data_slice()) {
        assert!(
            (r - orig).abs() < 1e-10,
            "SVD output fed to contract: U*S*Vt != A"
        );
    }
}

#[test]
fn transpose_output_feeds_into_contract() {
    // (A^T)^T = A
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let at = a.transpose(&[1, 0]).unwrap();
    let att = at.transpose(&[1, 0]).unwrap();

    // Feed transpose output into transpose — roundtrip should recover original
    for (original, roundtrip) in a.data_slice().iter().zip(att.data_slice()) {
        assert!(
            (original - roundtrip).abs() < 1e-15,
            "transpose roundtrip: (A^T)^T != A"
        );
    }
}

#[test]
fn expm_output_feeds_into_contract() {
    // exp(A) * exp(-A) = I for any square matrix A

    // Small Hermitian matrix so expm converges well
    let a = cm(vec![0.1, 0.2, 0.2, 0.3], vec![2, 2]);
    let neg_data: Vec<f64> = a.data_slice().iter().map(|&x: &f64| -x).collect();
    let neg_td =
        DenseTensorData::from_raw_parts(neg_data, a.shape().to_vec(), MemoryOrder::ColumnMajor);
    let neg_a = DenseTensor::from_data(neg_td);

    let exp_a = a.expm(1).unwrap();
    let exp_neg_a = neg_a.expm(1).unwrap();

    // Feed expm output directly into contract — no manual reorder
    let product = exp_a.contract(&exp_neg_a, "ij,jk->ik").unwrap();

    // Expected: identity in CM
    let identity = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    for (p, e) in product.data_slice().iter().zip(identity.data_slice()) {
        assert!(
            (p - e).abs() < 1e-10,
            "expm output fed to contract: exp(A)*exp(-A) != I"
        );
    }
}
