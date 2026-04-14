//! Contract tests: linalg operations return data in backend.preferred_order().
//!
//! These tests verify the representation invariant that all linalg outputs
//! are in `backend.preferred_order()` (ColumnMajor for NativeBackend).
//! The invariant is checked by feeding linalg outputs directly into other
//! linalg operations without manual reorder — if an output were in the
//! wrong order, the downstream operation would produce numerically wrong
//! results.
//!
//! Motivation: #126 refactoring removed the Dense.order field. Without it,
//! the only guarantee that data is in the correct order is that each operation
//! produces preferred_order output. A violation is silent (no compile error,
//! no panic) and only manifests as wrong numerical results.

use arnet_core::backend::ComputeBackend;
use arnet_linalg::{contract, diagonal_scale, inverse, reorder, solve, svd, transpose};
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, MemoryOrder};

/// Create Dense from conceptual row-major data, converted to CM for NativeBackend.
fn cm(data: Vec<f64>, shape: Vec<usize>) -> Dense<f64> {
    let rm = Dense::new(data, shape);
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
    // C = A * B, then verify C * B^T = A * (B * B^T)
    let backend = NativeBackend::new();
    let a = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = contract(&backend, &a, &b, "ij,jk->ik").unwrap();
    let bt = transpose(&backend, &b, &[1, 0]).unwrap();

    // Feed contract output (c) directly into another contract — no manual reorder
    let lhs = contract(&backend, &c, &bt, "ij,jk->ik").unwrap();

    let bbt = contract(&backend, &b, &bt, "ij,jk->ik").unwrap();
    let rhs = contract(&backend, &a, &bbt, "ij,jk->ik").unwrap();

    for (l, r) in lhs.data().iter().zip(rhs.data()) {
        assert!(
            (l - r).abs() < 1e-10,
            "contract output order mismatch: A*(B*Bt) != (A*B)*Bt"
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
    let order = backend.preferred_order();
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let (u, s, vt) = svd(&backend, &a, 1).unwrap();

    // U * diag(S) * Vt — feed SVD outputs directly, no manual reorder
    let us = diagonal_scale(&u, s.data(), u.rank() - 1, order).unwrap();
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

// =========================================================================
// Contract B: diagonal_scale layout invariance
//
// The same logical tensor, laid out in RM and CM, should produce
// logically identical results when diagonal_scale is called with
// the matching order parameter.
// =========================================================================

#[test]
fn diagonal_scale_rm_cm_invariance_axis0() {
    // Logical 2×3 matrix: [[1,2,3],[4,5,6]]
    // Scale rows by [10, 20] → [[10,20,30],[80,100,120]]
    let rm_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cm_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let t_rm = Dense::new(rm_data, vec![2, 3]);
    let t_cm = Dense::new(cm_data, vec![2, 3]);
    let weights = [10.0, 20.0];

    let r_rm = diagonal_scale(&t_rm, &weights, 0, MemoryOrder::RowMajor).unwrap();
    let r_cm = diagonal_scale(&t_cm, &weights, 0, MemoryOrder::ColumnMajor).unwrap();

    // Convert both to RM for logical comparison
    let r_cm_as_rm = reorder(&r_cm, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);

    assert_eq!(r_rm.data(), r_cm_as_rm.data(), "axis0 RM/CM mismatch");
}

#[test]
fn diagonal_scale_rm_cm_invariance_axis1() {
    // Logical 2×3 matrix: [[1,2,3],[4,5,6]]
    // Scale columns by [10, 20, 30] → [[10,40,90],[40,100,180]]
    let rm_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cm_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let t_rm = Dense::new(rm_data, vec![2, 3]);
    let t_cm = Dense::new(cm_data, vec![2, 3]);
    let weights = [10.0, 20.0, 30.0];

    let r_rm = diagonal_scale(&t_rm, &weights, 1, MemoryOrder::RowMajor).unwrap();
    let r_cm = diagonal_scale(&t_cm, &weights, 1, MemoryOrder::ColumnMajor).unwrap();

    let r_cm_as_rm = reorder(&r_cm, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);

    assert_eq!(r_rm.data(), r_cm_as_rm.data(), "axis1 RM/CM mismatch");
}

#[test]
fn diagonal_scale_rm_cm_invariance_rank3() {
    // Logical 2×2×2 tensor, scale along axis 1 by [3, 7]
    // RM: [[[a,b],[c,d]],[[e,f],[g,h]]] flattened as [a,b,c,d,e,f,g,h]
    let rm_data: Vec<f64> = (1..=8).map(|x| x as f64).collect();
    let t_rm = Dense::new(rm_data, vec![2, 2, 2]);

    // CM layout of same logical tensor
    let cm_data = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
    let t_cm = Dense::new(cm_data, vec![2, 2, 2]);

    let weights = [3.0, 7.0];

    let r_rm = diagonal_scale(&t_rm, &weights, 1, MemoryOrder::RowMajor).unwrap();
    let r_cm = diagonal_scale(&t_cm, &weights, 1, MemoryOrder::ColumnMajor).unwrap();

    let r_cm_as_rm = reorder(&r_cm, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);

    for (a, b) in r_rm.data().iter().zip(r_cm_as_rm.data()) {
        assert!(
            (a - b).abs() < 1e-10,
            "rank3 axis1 RM/CM mismatch: {a} vs {b}"
        );
    }
}
