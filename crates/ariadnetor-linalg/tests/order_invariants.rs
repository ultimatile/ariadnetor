//! Cross-crate invariant tests for the `Dense::order()` field.
//!
//! Pins assertions that span linalg ops (`linear_combine` mixed-order
//! rejection) and the cross-API regression (a `RowMajor` Dense paired
//! with a `ColumnMajor` backend must produce the same `contract` result
//! as the equivalent `ColumnMajor` Dense, demonstrating that op-entry
//! normalization is wired up).

use arnet_linalg::{contract, linear_combine};
use arnet_native::NativeBackend;
use arnet_tensor::{ComputeBackend, ComputeBackendTensorExt, Dense, MemoryOrder, reorder};

#[test]
fn linear_combine_rejects_mixed_orders() {
    let a = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let result = linear_combine(&[&a, &b], &[1.0, 1.0]);
    assert!(
        result.is_err(),
        "linear_combine must reject inputs with mismatched memory order"
    );
}

#[test]
fn linear_combine_accepts_matched_orders() {
    let a = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b = Dense::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], MemoryOrder::RowMajor);
    let result = linear_combine(&[&a, &b], &[1.0, 1.0]).expect("matched orders must combine");
    assert_eq!(result.order(), MemoryOrder::RowMajor);
}

/// Cross-API regression: a `RowMajor`-flagged Dense fed to `contract` on
/// a `ColumnMajor` backend must produce the same logical result as the
/// equivalent `ColumnMajor` Dense. If op-entry normalization regresses
/// (e.g. an op starts trusting `tensor.data()` is in `preferred_order`
/// without consulting `tensor.order()`), this test fails.
#[test]
fn contract_normalizes_row_major_input_against_column_major_backend() {
    let backend = NativeBackend::new();

    // Build the same logical 2x2 matrix product A * B for two layouts.
    // Logical A = [[1, 2], [3, 4]], Logical B = [[5, 6], [7, 8]].
    // Expected A*B = [[19, 22], [43, 50]].
    let a_rm = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b_rm = Dense::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], MemoryOrder::RowMajor);

    let a_cm = reorder(&a_rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);
    let b_cm = reorder(&b_rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);

    let c_rm_inputs = contract(&backend, &a_rm, &b_rm, "ij,jk->ik")
        .expect("contract with RM-flagged inputs must normalize and succeed");
    let c_cm_inputs = contract(&backend, &a_cm, &b_cm, "ij,jk->ik")
        .expect("contract with CM-flagged inputs is the reference path");

    // Outputs may carry either order — compare logically by first
    // normalizing both to RowMajor for byte comparison.
    let c_rm_norm = reorder(&c_rm_inputs, c_rm_inputs.order(), MemoryOrder::RowMajor);
    let c_cm_norm = reorder(&c_cm_inputs, c_cm_inputs.order(), MemoryOrder::RowMajor);

    assert_eq!(c_rm_norm.shape(), c_cm_norm.shape());
    assert_eq!(c_rm_norm.data(), c_cm_norm.data());
    assert_eq!(c_rm_norm.data(), &[19.0, 22.0, 43.0, 50.0]);
}

/// Companion to the no-permutation case above: a notation that
/// requires permuting the LHS forces `contract` through its
/// `transpose` preprocessing branch. If `transpose` regresses on
/// `tensor.order()` normalization, this case detects it (the
/// no-permutation path bypasses the transpose call entirely).
#[test]
fn contract_with_permutation_normalizes_row_major_input_against_column_major_backend() {
    let backend = NativeBackend::new();

    // Logical A = [[1, 2], [3, 4]] with notation `"ji,jk->ik"` →
    // contract over A's row index j (LHS axis 0) with B's row index j
    // (RHS axis 0). Effectively (A^T) * B.
    // (A^T) * B with A = [[1,2],[3,4]], B = [[5,6],[7,8]] gives
    // [[1,3],[2,4]] * [[5,6],[7,8]] = [[26, 30], [38, 44]].
    let a_rm = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b_rm = Dense::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], MemoryOrder::RowMajor);

    let a_cm = reorder(&a_rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);
    let b_cm = reorder(&b_rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);

    let c_rm_inputs = contract(&backend, &a_rm, &b_rm, "ji,jk->ik")
        .expect("contract with RM-flagged inputs must normalize and succeed");
    let c_cm_inputs = contract(&backend, &a_cm, &b_cm, "ji,jk->ik")
        .expect("contract with CM-flagged inputs is the reference path");

    let c_rm_norm = reorder(&c_rm_inputs, c_rm_inputs.order(), MemoryOrder::RowMajor);
    let c_cm_norm = reorder(&c_cm_inputs, c_cm_inputs.order(), MemoryOrder::RowMajor);

    assert_eq!(c_rm_norm.shape(), c_cm_norm.shape());
    assert_eq!(c_rm_norm.data(), c_cm_norm.data());
    assert_eq!(c_rm_norm.data(), &[26.0, 30.0, 38.0, 44.0]);
}

#[test]
fn backend_make_tensor_uses_preferred_order() {
    let backend = NativeBackend::new();
    let t = backend.make_tensor::<f64>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(t.order(), backend.preferred_order());
}

#[test]
fn backend_zeros_uses_preferred_order() {
    let backend = NativeBackend::new();
    let t = backend.zeros::<f64>(vec![2, 3]);
    assert_eq!(t.order(), backend.preferred_order());
}

#[test]
fn backend_ones_uses_preferred_order() {
    let backend = NativeBackend::new();
    let t = backend.ones::<f64>(vec![2, 3]);
    assert_eq!(t.order(), backend.preferred_order());
}

#[test]
fn backend_constant_uses_preferred_order() {
    let backend = NativeBackend::new();
    let t = backend.constant::<f64>(vec![2, 3], 7.0);
    assert_eq!(t.order(), backend.preferred_order());
}

#[test]
fn backend_eye_uses_preferred_order() {
    let backend = NativeBackend::new();
    let t = backend.eye::<f64>(3);
    assert_eq!(t.order(), backend.preferred_order());
}
