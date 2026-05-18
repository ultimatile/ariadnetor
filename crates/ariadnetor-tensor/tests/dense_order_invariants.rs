//! Invariant tests for `DenseTensorData<T>::order()` propagation
//! across operations.
//!
//! Pins the layout-authority contract: every operation must declare
//! what its output's `order()` is, and consuming ops must enforce or
//! propagate that order consistently.

use arnet_tensor::{DenseTensorData, MemoryOrder, normalize_to, reorder};
use num_complex::Complex;
use std::borrow::Cow;

#[test]
fn from_raw_parts_round_trips_source_order_row_major() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    assert_eq!(t.order(), MemoryOrder::RowMajor);
}

#[test]
fn from_raw_parts_round_trips_source_order_column_major() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::ColumnMajor,
    );
    assert_eq!(t.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn reorder_outputs_target_order() {
    let rm = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let cm = reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);
    assert_eq!(cm.order(), MemoryOrder::ColumnMajor);
    let back = reorder(&cm, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);
    assert_eq!(back.order(), MemoryOrder::RowMajor);
}

#[test]
fn reshape_preserves_order_row_major() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let r = t.reshape(vec![6]);
    assert_eq!(r.order(), MemoryOrder::RowMajor);
}

#[test]
fn reshape_preserves_order_column_major() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let r = t.reshape(vec![6]);
    assert_eq!(r.order(), MemoryOrder::ColumnMajor);
}

#[test]
fn normalize_to_borrows_when_order_matches() {
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let cow = normalize_to(&t, MemoryOrder::RowMajor);
    assert!(
        matches!(cow, Cow::Borrowed(_)),
        "normalize_to must return Cow::Borrowed when source order already matches target"
    );
}

#[test]
fn normalize_to_owns_when_order_differs() {
    let t = DenseTensorData::<f64>::from_raw_parts(
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
fn get_honors_storage_order() {
    // Two DenseTensorData holding the same logical matrix in their
    // respective layouts must return the same value at the same
    // logical `[i, j, ...]`. The chosen indices have distinct flat
    // positions under RowMajor vs ColumnMajor, so a regression to
    // row-major-only indexing on the CM-tagged storage would surface
    // as a value mismatch.
    let m_rm = DenseTensorData::<f64>::from_raw_parts(
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let m_cm = DenseTensorData::<f64>::from_raw_parts(
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
    let m_rm3 =
        DenseTensorData::<f64>::from_raw_parts(rm_data, vec![2, 3, 4], MemoryOrder::RowMajor);
    let m_cm3 =
        DenseTensorData::<f64>::from_raw_parts(cm_data, vec![2, 3, 4], MemoryOrder::ColumnMajor);
    assert_eq!(m_rm3.get(&[0, 1, 2]), 12.0);
    assert_eq!(m_cm3.get(&[0, 1, 2]), 12.0);
    assert_eq!(m_rm3.get(&[1, 1, 2]), 112.0);
    assert_eq!(m_cm3.get(&[1, 1, 2]), 112.0);
}

#[test]
#[should_panic(expected = "out of bounds for axis")]
fn get_panics_on_oob_column_major() {
    // CM-tagged shape=[2, 3]: get(&[2, 0]) computes CM flat index
    // 2 + 0 * 2 = 2, which is within the 6-element data buffer.
    // Without explicit bounds checking the call would silently return
    // `data[2]` instead of panicking.
    let t = DenseTensorData::<f64>::from_raw_parts(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::ColumnMajor,
    );
    let _ = t.get(&[2, 0]);
}

// Row-major flat data of `σ^x ⊗ σ^y`, a 4×4 complex Hermitian matrix
// with non-real off-diagonal entries. Used as a boundary-contract
// fixture: real-symmetric fixtures hide row-major-vs-column-major
// mistagging because their transpose equals themselves, whereas a
// complex Hermitian H reinterpreted under the wrong layout becomes
// conj(H) and the swap is visible.
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
    let t =
        DenseTensorData::<Complex<f64>>::from_raw_parts(data, vec![4, 4], MemoryOrder::RowMajor);
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
    let t =
        DenseTensorData::<Complex<f64>>::from_raw_parts(data, vec![4, 4], MemoryOrder::ColumnMajor);
    let i = Complex::new(0.0, 1.0);
    let ni = Complex::new(0.0, -1.0);
    assert_eq!(t.get(&[0, 3]), i);
    assert_eq!(t.get(&[3, 0]), ni);
    assert_eq!(t.get(&[1, 2]), ni);
    assert_eq!(t.get(&[2, 1]), i);
}
