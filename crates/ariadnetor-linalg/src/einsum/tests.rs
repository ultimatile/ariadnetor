//! Unit tests for the einsum dispatcher that need in-crate test doubles
//! (e.g. the `pub(crate)` `RowMajorBackend`), which integration tests cannot
//! reach.

use super::einsum_dense;
use crate::reorder_route::reorder_via_backend;
use crate::test_util::RowMajorBackend;
use ariadnetor_core::backend::MemoryOrder;
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::DenseTensorData;

/// A batched contraction must produce the same result whichever memory order
/// the backend prefers. `RowMajorBackend` forces the RowMajor batch-axis
/// layout branch of `batched_contract` and its output reconstruction — the
/// native (ColumnMajor) backend never selects it, so without this the branch
/// added for backend-generality would ship unexercised.
#[test]
fn batched_contract_agrees_across_backend_orders() {
    // bik,bkj->bij with b=2, i=2, k=3, j=4 (asymmetric m=2, n=4, k=3).
    let lhs = DenseTensorData::from_raw_parts(
        (1..=12).map(|x| x as f64).collect(),
        vec![2, 2, 3],
        MemoryOrder::RowMajor,
    );
    let rhs = DenseTensorData::from_raw_parts(
        (1..=24).map(|x| x as f64).collect(),
        vec![2, 3, 4],
        MemoryOrder::RowMajor,
    );
    let cm = NativeBackend::new();
    let rm = RowMajorBackend::new();
    let out_cm = einsum_dense(&cm, &[&lhs, &rhs], "bik,bkj->bij").unwrap();
    let out_rm = einsum_dense(&rm, &[&lhs, &rhs], "bik,bkj->bij").unwrap();
    // Normalize both to a common order before the element-wise compare.
    let a = reorder_via_backend(&cm, &out_cm, MemoryOrder::RowMajor).unwrap();
    let b = reorder_via_backend(&cm, &out_rm, MemoryOrder::RowMajor).unwrap();
    assert_eq!(a.shape(), &[2, 2, 4]);
    assert_eq!(a.shape(), b.shape());
    for (x, y) in a.data().iter().zip(b.data().iter()) {
        assert!(
            (x - y).abs() < 1e-10,
            "backend order disagreement: {x} vs {y}"
        );
    }
}
