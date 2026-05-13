//! Invariant tests for `Dense<T>::order()` propagation across operations.
//!
//! Pins the layout-authority contract introduced when `Dense` gained an
//! explicit `order` field: every operation must declare what its output's
//! `order()` is, and consuming ops must enforce or propagate that order
//! consistently. Without these tests, a future refactor could silently
//! revert `order` to the implicit `backend.preferred_order()` model that
//! motivated the original boundary-contract bug.

use arnet_tensor::{Dense, MemoryOrder, normalize_to, reorder};
use num_complex::Complex;
use std::borrow::Cow;

#[test]
fn dense_new_round_trips_source_order_row_major() {
    let t = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    assert_eq!(t.order(), MemoryOrder::RowMajor);
}

#[test]
fn dense_new_round_trips_source_order_column_major() {
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    assert_eq!(t.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn reorder_outputs_target_order() {
    let rm = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let cm = reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);
    assert_eq!(cm.order(), MemoryOrder::ColumnMajor);
    let back = reorder(&cm, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);
    assert_eq!(back.order(), MemoryOrder::RowMajor);
}

#[test]
fn map_preserves_order_row_major() {
    let t = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let m = t.map(|x| x * 2.0);
    assert_eq!(m.order(), MemoryOrder::RowMajor);
}

#[test]
fn map_preserves_order_column_major() {
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let m = t.map(|x| x * 2.0);
    assert_eq!(m.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn reshape_preserves_order_row_major() {
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let r = t.reshape(vec![6]);
    assert_eq!(r.order(), MemoryOrder::RowMajor);
}

#[test]
fn reshape_preserves_order_column_major() {
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let r = t.reshape(vec![6]);
    assert_eq!(r.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn map_with_index_outputs_iteration_order() {
    // `map_with_index` iterates in `self.order()`; build one tensor
    // per order to verify the output `order()` matches the input's
    // storage order in each case.
    let t_rm = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let row_major_out = t_rm.map_with_index(|_idx, &x| x);
    assert_eq!(row_major_out.order(), MemoryOrder::RowMajor);

    let t_cm = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let col_major_out = t_cm.map_with_index(|_idx, &x| x);
    assert_eq!(col_major_out.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn dense_linear_combine_rejects_mixed_orders() {
    let a = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let result = Dense::linear_combine(&[&a, &b], &[1.0, 1.0]);
    assert!(
        result.is_err(),
        "Dense::linear_combine must reject inputs with mismatched memory order"
    );
}

#[test]
fn normalize_to_borrows_when_order_matches() {
    let t = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let cow = normalize_to(&t, MemoryOrder::RowMajor);
    assert!(
        matches!(cow, Cow::Borrowed(_)),
        "normalize_to must return Cow::Borrowed when source order already matches target"
    );
}

#[test]
fn normalize_to_owns_when_order_differs() {
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let cow = normalize_to(&t, MemoryOrder::RowMajor);
    assert!(
        matches!(cow, Cow::Owned(_)),
        "normalize_to must return Cow::Owned when a reorder is performed"
    );
    assert_eq!(cow.order(), MemoryOrder::RowMajor);
}

#[test]
fn dense_get_honors_storage_order() {
    // Two Dense holding the same logical matrix in their respective
    // layouts must return the same value at the same logical
    // `[i, j, ...]`. The chosen indices have distinct flat positions
    // under RowMajor vs ColumnMajor, so a regression to row-major-
    // only indexing on the CM-tagged storage would surface as a
    // value mismatch.

    // Rank-2: shape [2, 3], M[i, j] = 10 * (i * 3 + j) + 10.
    // RM flat: [10, 20, 30, 40, 50, 60].
    // CM flat: [10, 40, 20, 50, 30, 60].
    let m_rm = Dense::<f64>::new(
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let m_cm = Dense::<f64>::new(
        vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    // [0, 2] flat positions: RM=2, CM=4 (distinct).
    assert_eq!(m_rm.get(&[0, 2]), 30.0);
    assert_eq!(m_cm.get(&[0, 2]), 30.0);
    // [1, 1] flat positions: RM=4, CM=3 (distinct).
    assert_eq!(m_rm.get(&[1, 1]), 50.0);
    assert_eq!(m_cm.get(&[1, 1]), 50.0);

    // Rank-3: shape [2, 3, 4], M[i, j, k] = 100*i + 10*j + k.
    let mut rm_data = Vec::with_capacity(24);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                rm_data.push((i * 100 + j * 10 + k) as f64);
            }
        }
    }
    let mut cm_data = Vec::with_capacity(24);
    for k in 0..4 {
        for j in 0..3 {
            for i in 0..2 {
                cm_data.push((i * 100 + j * 10 + k) as f64);
            }
        }
    }
    let m_rm3 = Dense::<f64>::new(rm_data, vec![2, 3, 4], MemoryOrder::RowMajor);
    let m_cm3 = Dense::<f64>::new(cm_data, vec![2, 3, 4], MemoryOrder::ColumnMajor);
    // [0, 1, 2] flat positions: RM=6, CM=14 (distinct).
    assert_eq!(m_rm3.get(&[0, 1, 2]), 12.0);
    assert_eq!(m_cm3.get(&[0, 1, 2]), 12.0);
    // [1, 1, 2] flat positions: RM=18, CM=15 (distinct).
    assert_eq!(m_rm3.get(&[1, 1, 2]), 112.0);
    assert_eq!(m_cm3.get(&[1, 1, 2]), 112.0);
}

#[test]
#[should_panic(expected = "out of bounds for axis")]
fn dense_get_panics_on_oob_column_major() {
    // CM-tagged shape=[2, 3]: get(&[2, 0]) computes CM flat index
    // 2 + 0 * 2 = 2, which is within the 6-element data buffer.
    // Without explicit bounds checking the call would silently
    // return `data[2]` instead of panicking. Pins the contract
    // that out-of-bounds indices panic regardless of whether the
    // computed flat position happens to land in range.
    let t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let _ = t.get(&[2, 0]);
}

#[test]
#[should_panic(expected = "out of bounds for axis")]
fn dense_set_panics_on_oob_column_major() {
    // Mirror of `dense_get_panics_on_oob_column_major` for `set`.
    // Without explicit bounds checking the write would silently
    // overwrite `data[2]` instead of panicking.
    let mut t = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    t.set(&[2, 0], 0.0);
}

#[test]
#[should_panic(expected = "order")]
fn concatenate_panics_on_mixed_order() {
    // Mixed-order inputs walk the strip-copy in incompatible directions;
    // the explicit assert_eq! on `t.order()` must reject this regardless
    // of shape compatibility.
    let a = Dense::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b = Dense::<f64>::new(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let _c = Dense::concatenate(&[&a, &b], 0);
}

#[test]
#[should_panic(expected = "order")]
fn stack_panics_on_mixed_order() {
    // Stack reshapes each input and forwards to concatenate; the order
    // mismatch must propagate through reshape (which preserves order).
    let a = Dense::<f64>::new(vec![1.0, 2.0], vec![2], MemoryOrder::RowMajor);
    let b = Dense::<f64>::new(vec![3.0, 4.0], vec![2], MemoryOrder::ColumnMajor);
    let _s = Dense::stack(&[&a, &b], 0);
}

#[test]
#[should_panic(expected = "order")]
fn replace_slice_panics_on_mixed_order_rank2() {
    // At rank >= 2 the strip-copy direction depends on the destination's
    // order; a different-order sub would land bytes at the wrong logical
    // positions, so the call must panic. (Rank-0 / rank-1 / empty sub
    // are layout-insensitive and remain valid under mixed orders.)
    let mut dst = Dense::<f64>::new(vec![0.0; 9], vec![3, 3], MemoryOrder::RowMajor);
    let sub = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    dst.replace_slice(&sub, &[0, 0]);
}

#[test]
fn concatenate_propagates_input_order_column_major() {
    // The output of concatenate must adopt the shared input order. RM
    // propagation is implicit in `tests/dense.rs::test_concatenate_axis0`
    // (asserts the RM-laid output buffer); this test pins the CM case
    // explicitly so the contract holds for both orders.
    let a = Dense::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let b = Dense::<f64>::new(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    let c = Dense::concatenate(&[&a, &b], 0);
    assert_eq!(c.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn dense_set_honors_storage_order() {
    // `set` on two zero-initialized Dense with different orders must
    // write the value at the flat position dictated by each Dense's
    // own order. Verifying via `data()` (raw flat slice) catches a
    // regression where `set` ignored `self.order()` and always wrote
    // at the row-major flat position.

    // Rank-2: shape [2, 3], write at [0, 2].
    // RM flat position: 0 * 3 + 2 = 2.
    // CM flat position: 0 + 2 * 2 = 4.
    let mut m_rm = Dense::<f64>::new(vec![0.0; 6], vec![2, 3], MemoryOrder::RowMajor);
    let mut m_cm = Dense::<f64>::new(vec![0.0; 6], vec![2, 3], MemoryOrder::ColumnMajor);
    m_rm.set(&[0, 2], 9.0);
    m_cm.set(&[0, 2], 9.0);
    assert_eq!(m_rm.data()[2], 9.0);
    assert_eq!(m_cm.data()[4], 9.0);
    // The other Dense's flat slot is still zero — ruling out the
    // "set ignores order" regression where both writes would land at
    // flat position 2.
    assert_eq!(m_rm.data()[4], 0.0);
    assert_eq!(m_cm.data()[2], 0.0);

    // Rank-3: shape [2, 3, 4], write at [0, 1, 2].
    // RM flat position: 0 * 12 + 1 * 4 + 2 = 6.
    // CM flat position: 0 + 1 * 2 + 2 * 6 = 14.
    let mut m_rm3 = Dense::<f64>::new(vec![0.0; 24], vec![2, 3, 4], MemoryOrder::RowMajor);
    let mut m_cm3 = Dense::<f64>::new(vec![0.0; 24], vec![2, 3, 4], MemoryOrder::ColumnMajor);
    m_rm3.set(&[0, 1, 2], 7.5);
    m_cm3.set(&[0, 1, 2], 7.5);
    assert_eq!(m_rm3.data()[6], 7.5);
    assert_eq!(m_cm3.data()[14], 7.5);
    assert_eq!(m_rm3.data()[14], 0.0);
    assert_eq!(m_cm3.data()[6], 0.0);
}

// Row-major flat data of `σ^x ⊗ σ^y`, a 4×4 complex Hermitian
// matrix with non-real off-diagonal entries. Used as a boundary-
// contract fixture: real-symmetric fixtures hide row-major-vs-
// column-major mistagging because their transpose equals
// themselves, whereas a complex Hermitian H reinterpreted under
// the wrong layout becomes conj(H) and the swap is visible.
fn pauli_sigma_x_kron_sigma_y_row_major() -> Vec<Complex<f64>> {
    let z = Complex::new(0.0, 0.0);
    let i = Complex::new(0.0, 1.0);
    let ni = Complex::new(0.0, -1.0);
    vec![
        z, z, z, ni, // row 0: H[0,3] = -i
        z, z, i, z, //  row 1: H[1,2] =  i
        z, ni, z, z, // row 2: H[2,1] = -i
        i, z, z, z, //  row 3: H[3,0] =  i
    ]
}

#[test]
fn complex_hermitian_correct_source_order_matches_analytical_entries() {
    let data = pauli_sigma_x_kron_sigma_y_row_major();
    let t = Dense::<Complex<f64>>::new(data, vec![4, 4], MemoryOrder::RowMajor);
    let i = Complex::new(0.0, 1.0);
    let ni = Complex::new(0.0, -1.0);
    assert_eq!(t.get(&[0, 3]), ni);
    assert_eq!(t.get(&[3, 0]), i);
    assert_eq!(t.get(&[1, 2]), i);
    assert_eq!(t.get(&[2, 1]), ni);
}

#[test]
fn complex_hermitian_wrong_source_order_returns_conjugated_entries() {
    // The same row-major flat data tagged column-major is interpreted
    // as H^T. For Hermitian H this equals conj(H), so each non-real
    // off-diagonal entry flips sign on its imaginary part. A real-
    // symmetric fixture would silently pass under this mis-tag.
    let data = pauli_sigma_x_kron_sigma_y_row_major();
    let t = Dense::<Complex<f64>>::new(data, vec![4, 4], MemoryOrder::ColumnMajor);
    let i = Complex::new(0.0, 1.0);
    let ni = Complex::new(0.0, -1.0);
    assert_eq!(t.get(&[0, 3]), i);
    assert_eq!(t.get(&[3, 0]), ni);
    assert_eq!(t.get(&[1, 2]), ni);
    assert_eq!(t.get(&[2, 1]), i);
}
