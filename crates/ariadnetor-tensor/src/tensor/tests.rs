use std::fmt;
use std::ops::Mul;

use arnet_core::Scalar;

use super::*;

fn assert_tensor_mutation<S>(zero: S, val: S, fill_val: S, scale_factor: S)
where
    S: Scalar + PartialEq + fmt::Debug + Mul<S, Output = S>,
{
    let mut t = DenseTensor::<S>::zeros(vec![2, 3]);

    // set / get round-trip
    t.set(&[1, 2], val);
    assert_eq!(t.get(&[1, 2]), val);
    assert_eq!(t.get(&[0, 0]), zero);

    // fill overwrites all elements
    t.fill(fill_val);
    assert_eq!(t.get(&[0, 0]), fill_val);
    assert_eq!(t.get(&[1, 2]), fill_val);

    // data_slice_mut provides mutable access
    t.data_slice_mut()[0] = val;
    assert_eq!(t.get(&[0, 0]), val);

    // scale multiplies all elements
    t.fill(val);
    t.scale(scale_factor);
    assert_eq!(t.get(&[0, 0]), val * scale_factor);
}

#[test]
fn test_tensor_mutation() {
    assert_tensor_mutation(0.0f64, 42.0, 2.72, 3.0);
    assert_tensor_mutation(0.0f32, 42.0, 2.72, 3.0);
}

#[test]
fn scaled_out_of_place_preserves_original() {
    let mut a = DenseTensor::<f64>::zeros(vec![2, 2]);
    a.fill(3.0);
    let b = a.scaled(2.0);
    // a unchanged
    assert_eq!(a.get(&[0, 0]), 3.0);
    // b scaled
    assert_eq!(b.get(&[0, 0]), 6.0);
    assert_eq!(b.get(&[1, 1]), 6.0);
    assert_eq!(b.shape(), a.shape());
}

#[test]
fn norm_matches_frobenius_definition() {
    let mut t = DenseTensor::<f64>::zeros(vec![2, 2]);
    t.set(&[0, 0], 3.0);
    t.set(&[1, 1], 4.0);
    // sqrt(9 + 16) = 5
    let n = t.norm();
    assert!((n - 5.0).abs() < 1e-12, "expected 5.0, got {n}");
}

#[test]
fn normalize_in_place_returns_original_norm_and_unitizes() {
    let mut t = DenseTensor::<f64>::zeros(vec![2]);
    t.set(&[0], 3.0);
    t.set(&[1], 4.0);
    let n = t.normalize();
    assert!((n - 5.0).abs() < 1e-12, "returned norm {n}, expected 5");
    // post-normalize Frobenius norm is 1
    assert!((t.norm() - 1.0).abs() < 1e-12);
    // elements scaled by 1/5
    assert!((t.get(&[0]) - 0.6).abs() < 1e-12);
    assert!((t.get(&[1]) - 0.8).abs() < 1e-12);
}

#[test]
fn normalized_out_of_place_keeps_original_intact() {
    let mut a = DenseTensor::<f64>::zeros(vec![2]);
    a.set(&[0], 3.0);
    a.set(&[1], 4.0);
    let (b, n) = a.normalized();
    assert!((n - 5.0).abs() < 1e-12);
    // original elements preserved
    assert_eq!(a.get(&[0]), 3.0);
    assert_eq!(a.get(&[1]), 4.0);
    // normalized copy has unit norm
    assert!((b.norm() - 1.0).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "Cannot normalize zero tensor")]
fn normalize_panics_on_zero_tensor() {
    let mut t = DenseTensor::<f64>::zeros(vec![3, 3]);
    t.normalize();
}

#[test]
fn linear_combine_sums_with_coefs() {
    let mut a = DenseTensor::<f64>::zeros(vec![2]);
    a.set(&[0], 1.0);
    a.set(&[1], 2.0);
    let mut b = DenseTensor::<f64>::zeros(vec![2]);
    b.set(&[0], 10.0);
    b.set(&[1], 20.0);
    let r = DenseTensor::linear_combine(&[&a, &b], &[3.0, 4.0]).unwrap();
    // 3*1 + 4*10 = 43; 3*2 + 4*20 = 86
    assert_eq!(r.get(&[0]), 43.0);
    assert_eq!(r.get(&[1]), 86.0);
    assert_eq!(r.shape(), a.shape());
}

#[test]
fn add_all_sums_with_unit_coefs() {
    let mut a = DenseTensor::<f64>::zeros(vec![2]);
    a.set(&[0], 1.0);
    a.set(&[1], 2.0);
    let mut b = DenseTensor::<f64>::zeros(vec![2]);
    b.set(&[0], 10.0);
    b.set(&[1], 20.0);
    let r = DenseTensor::add_all(&[&a, &b]).unwrap();
    assert_eq!(r.get(&[0]), 11.0);
    assert_eq!(r.get(&[1]), 22.0);
}

#[test]
fn linear_combine_rejects_empty_list() {
    let err = DenseTensor::<f64>::linear_combine(&[], &[]).unwrap_err();
    assert!(err.contains("empty"), "got: {err}");
}

#[test]
fn linear_combine_rejects_length_mismatch() {
    let a = DenseTensor::<f64>::zeros(vec![2]);
    let b = DenseTensor::<f64>::zeros(vec![2]);
    let err = DenseTensor::linear_combine(&[&a, &b], &[1.0]).unwrap_err();
    assert!(
        err.contains("tensors.len()") && err.contains("coefs.len()"),
        "got: {err}",
    );
}

#[test]
fn linear_combine_rejects_shape_mismatch() {
    let a = DenseTensor::<f64>::zeros(vec![2]);
    let b = DenseTensor::<f64>::zeros(vec![3]);
    let err = DenseTensor::linear_combine(&[&a, &b], &[1.0, 1.0]).unwrap_err();
    assert!(err.contains("shape mismatch"), "got: {err}");
}

#[test]
fn block_sparse_tensor_alias_resolves_and_basics_work() {
    use crate::{Direction, U1Sector};

    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let t = BlockSparseTensor::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));

    assert_eq!(t.shape(), &[5, 5]);
    assert_eq!(t.rank(), 2);
}

#[test]
fn dense_tensor_from_raw_parts_pairs_data_with_backend_and_order() {
    use arnet_core::backend::MemoryOrder;
    use arnet_native::NativeBackend;

    let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let backend = NativeBackend::shared();
    let t = DenseTensor::<f64>::from_raw_parts(
        data.clone(),
        vec![2, 3],
        MemoryOrder::RowMajor,
        backend,
    );

    assert_eq!(t.shape(), &[2, 3]);
    // Layout's order reflects the explicit argument, not the backend
    // (the joined Tier 1 check that orders agree is a downstream
    // concern, not enforced at the joined constructor).
    assert_eq!(t.data().layout().order(), MemoryOrder::RowMajor);
    assert_eq!(t.data_slice(), data.as_slice());
}

#[test]
fn block_sparse_tensor_zeros_with_backend_uses_backend_order() {
    use crate::{Direction, U1Sector};
    use arnet_core::backend::ComputeBackend;
    use arnet_native::NativeBackend;

    let idx = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let backend = NativeBackend::shared();
    let expected_order = backend.preferred_order();
    let t = BlockSparseTensor::<f64, U1Sector>::zeros_with_backend(
        vec![idx.clone(), idx],
        U1Sector(0),
        backend,
    );

    assert_eq!(t.rank(), 2);
    assert_eq!(t.data().layout().order(), expected_order);
}

#[test]
fn dense_tensor_conj_real_path_is_identity_and_shares_backend() {
    use std::sync::Arc;

    let mut t = DenseTensor::<f64>::zeros(vec![3, 3]);
    t.fill(2.5);
    let c = t.conj();

    assert_eq!(c.shape(), t.shape());
    assert_eq!(c.data_slice(), t.data_slice());
    assert!(Arc::ptr_eq(t.backend_arc(), c.backend_arc()));
}

#[test]
fn block_sparse_tensor_dagger_is_involutive() {
    use crate::{Direction, U1Sector};

    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let t = BlockSparseTensor::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));

    let t_dd = t.dagger().dagger();

    assert_eq!(t_dd.shape(), t.shape());
    assert_eq!(t_dd.flux(), t.flux());
    for (a, b) in t.indices().iter().zip(t_dd.indices().iter()) {
        assert_eq!(a.direction(), b.direction());
    }
}
