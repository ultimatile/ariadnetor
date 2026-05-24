//! Cross-crate invariant tests for the `Dense::order()` field.
//!
//! Pins assertions that span linalg ops (`linear_combine` mixed-order
//! rejection) and the cross-API regression (a `RowMajor` Dense paired
//! with a `ColumnMajor` backend must produce the same `contract` result
//! as the equivalent `ColumnMajor` Dense, demonstrating that op-entry
//! normalization is wired up).

use arnet_core::Scalar;
use arnet_linalg::{
    TruncSvdParams, contract, einsum, expm, inverse, linear_combine, solve, svd, trunc_svd,
};
use arnet_native::NativeBackend;
use arnet_tensor::{
    ComputeBackend, ComputeBackendTensorExt, Dense, DenseTensor, MemoryOrder, reorder,
};
use num_complex::Complex;

/// Wrap a legacy `Dense<T>` into the user-facing `DenseTensor<T, NativeBackend>`
/// for feeding into linalg pub fns. Preserves the order tag so the
/// order-normalization branches inside the linalg op are exercised the
/// same way as before the API migration.
fn wrap<T: Clone>(d: Dense<T>) -> DenseTensor<T, NativeBackend> {
    DenseTensor::with_backend(d.into_tensor_data(), NativeBackend::shared())
}

/// View a `DenseTensor` back as a legacy `Dense<T>` for assertions that
/// use `reorder` and slice-level data comparison.
fn as_dense<T: Clone>(t: &DenseTensor<T, NativeBackend>) -> Dense<T> {
    t.data().as_dense()
}

/// Construct a `Dense<T>` representing the same logical tensor as
/// `data_rm` (interpreted in RowMajor flat order) but tagged with the
/// requested `order`. For `order == RowMajor` the data is used directly;
/// for `ColumnMajor` it is reordered so the byte layout matches the tag.
/// This is the foundation of `assert_op_layout_invariance` — without it,
/// the same raw bytes under different tags would describe different
/// logical matrices, defeating the layout-invariance check.
fn build_tagged<T: Scalar>(data_rm: &[T], shape: &[usize], order: MemoryOrder) -> Dense<T> {
    let rm = Dense::<T>::new(data_rm.to_vec(), shape.to_vec(), MemoryOrder::RowMajor);
    if order == MemoryOrder::RowMajor {
        rm
    } else {
        reorder(&rm, MemoryOrder::RowMajor, order)
    }
}

/// Run `op_with_order` for `RowMajor` and `ColumnMajor` taggings of the
/// same logical input, then assert the two outputs agree after both are
/// normalized to `RowMajor` for byte comparison. Failure indicates a
/// silent transpose somewhere in the op — typically a `reorder` call
/// that passes `preferred_order` as the `from` argument when the input's
/// actual `.order()` differs. The closure must use `build_tagged` (or an
/// equivalent) so its inputs describe the same logical tensor under both
/// orders.
fn assert_op_layout_invariance<T, F>(op_label: &str, op_with_order: F)
where
    T: Scalar + PartialEq + std::fmt::Debug,
    F: Fn(MemoryOrder) -> Dense<T>,
{
    let out_rm = op_with_order(MemoryOrder::RowMajor);
    let out_cm = op_with_order(MemoryOrder::ColumnMajor);

    let rm_norm = reorder(&out_rm, out_rm.order(), MemoryOrder::RowMajor);
    let cm_norm = reorder(&out_cm, out_cm.order(), MemoryOrder::RowMajor);

    assert_eq!(
        rm_norm.shape(),
        cm_norm.shape(),
        "{op_label}: shape mismatch"
    );
    assert_eq!(
        rm_norm.data(),
        cm_norm.data(),
        "{op_label}: RM and CM inputs produced different data (silent transpose)"
    );
}

#[test]
fn linear_combine_rejects_mixed_orders() {
    let a = wrap(Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    ));
    let b = wrap(Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    ));
    let result = linear_combine(&[&a, &b], &[1.0, 1.0]);
    assert!(
        result.is_err(),
        "linear_combine must reject inputs with mismatched memory order"
    );
}

#[test]
fn linear_combine_accepts_matched_orders() {
    let a = wrap(Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    ));
    let b = wrap(Dense::<f64>::new(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    ));
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
    // Build the same logical 2x2 matrix product A * B for two layouts.
    // Logical A = [[1, 2], [3, 4]], Logical B = [[5, 6], [7, 8]].
    // Expected A*B = [[19, 22], [43, 50]].
    let a_rm = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b_rm = Dense::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], MemoryOrder::RowMajor);

    let a_cm = reorder(&a_rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);
    let b_cm = reorder(&b_rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);

    let c_rm_inputs = contract(&wrap(a_rm), &wrap(b_rm), "ij,jk->ik")
        .expect("contract with RM-flagged inputs must normalize and succeed");
    let c_cm_inputs = contract(&wrap(a_cm), &wrap(b_cm), "ij,jk->ik")
        .expect("contract with CM-flagged inputs is the reference path");

    let c_rm_inputs = as_dense(&c_rm_inputs);
    let c_cm_inputs = as_dense(&c_cm_inputs);

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
    // Logical A = [[1, 2], [3, 4]] with notation `"ji,jk->ik"` →
    // contract over A's row index j (LHS axis 0) with B's row index j
    // (RHS axis 0). Effectively (A^T) * B.
    // (A^T) * B with A = [[1,2],[3,4]], B = [[5,6],[7,8]] gives
    // [[1,3],[2,4]] * [[5,6],[7,8]] = [[26, 30], [38, 44]].
    let a_rm = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b_rm = Dense::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], MemoryOrder::RowMajor);

    let a_cm = reorder(&a_rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);
    let b_cm = reorder(&b_rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);

    let c_rm_inputs = contract(&wrap(a_rm), &wrap(b_rm), "ji,jk->ik")
        .expect("contract with RM-flagged inputs must normalize and succeed");
    let c_cm_inputs = contract(&wrap(a_cm), &wrap(b_cm), "ji,jk->ik")
        .expect("contract with CM-flagged inputs is the reference path");

    let c_rm_inputs = as_dense(&c_rm_inputs);
    let c_cm_inputs = as_dense(&c_cm_inputs);

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

#[test]
fn svd_s_tensor_uses_preferred_order() {
    let backend = NativeBackend::new();
    let a = wrap(Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    ));
    let (_u, s, _vt) = svd(&a, 1).expect("svd must succeed");
    assert_eq!(s.order(), backend.preferred_order());
}

#[test]
fn trunc_svd_s_tensor_uses_preferred_order() {
    let backend = NativeBackend::new();
    let a = wrap(Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    ));
    let params = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let (_u, s, _vt, _err) = trunc_svd(&a, 1, &params).expect("trunc_svd must succeed");
    assert_eq!(s.order(), backend.preferred_order());
}

#[test]
fn solve_normalizes_row_major_input() {
    // Asymmetric 2x2 system A = [[2, 1], [5, 3]] (det = 1, invertible).
    // Asymmetry is the property the test relies on: a symmetric matrix
    // would be invariant under transpose and mask the silent-transpose bug.
    let a_data = [2.0f64, 1.0, 5.0, 3.0];
    let b_data = [4.0f64, 7.0];
    assert_ne!(a_data[1], a_data[2], "fixture must be asymmetric");
    assert_op_layout_invariance("solve", |order| {
        let a = build_tagged(&a_data, &[2, 2], order);
        let b = build_tagged(&b_data, &[2, 1], order);
        let out = solve(&wrap(a), &wrap(b), 1).expect("solve must succeed");
        as_dense(&out)
    });
}

#[test]
fn inverse_normalizes_row_major_input() {
    // Same asymmetric 2x2 A as `solve_normalizes_row_major_input`.
    let a_data = [2.0f64, 1.0, 5.0, 3.0];
    assert_ne!(a_data[1], a_data[2], "fixture must be asymmetric");
    assert_op_layout_invariance("inverse", |order| {
        let a = build_tagged(&a_data, &[2, 2], order);
        let out = inverse(&wrap(a), 1).expect("inverse must succeed");
        as_dense(&out)
    });
}

#[test]
fn expm_normalizes_row_major_input() {
    // Asymmetric complex 2x2 M = [[0.1+0.2i, 0.3], [0.05, -0.2+0.1i]].
    // Non-Hermitian and asymmetric: forces the general `expm` Pade path
    // (rather than `expm_hermitian` / `expm_antihermitian`) and ensures
    // off-diagonal entries differ so a silent transpose is observable.
    let data = [
        Complex::new(0.1, 0.2),
        Complex::new(0.3_f64, 0.0),
        Complex::new(0.05, 0.0),
        Complex::new(-0.2, 0.1),
    ];
    assert_ne!(data[1], data[2], "fixture must be asymmetric");
    assert_op_layout_invariance("expm", |order| {
        let m = build_tagged(&data, &[2, 2], order);
        let out = expm(&wrap(m), 1).expect("expm must succeed");
        as_dense(&out)
    });
}

#[test]
fn batched_einsum_normalizes_row_major_input() {
    // Asymmetric rank-3 fixtures for batched notation "bik,bkj->bij":
    // batch=2, LHS and RHS are each 2x2x2. The notation has indices
    // already in canonical [batch, free, contracted] order, so
    // `batched_contract` takes the no-permutation branch — exactly the
    // branch where the latent reorder-source bug lives.
    let lhs_data: Vec<f64> = (1..=8).map(|x| x as f64).collect();
    let rhs_data: Vec<f64> = (9..=16).map(|x| x as f64).collect();
    assert_op_layout_invariance("batched_einsum", |order| {
        let a = build_tagged(&lhs_data, &[2, 2, 2], order);
        let b = build_tagged(&rhs_data, &[2, 2, 2], order);
        let wa = wrap(a);
        let wb = wrap(b);
        let out = einsum(&[&wa, &wb], "bik,bkj->bij").expect("einsum must succeed");
        as_dense(&out)
    });
}
