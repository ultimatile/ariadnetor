//! `ComputeBackendTensorExt::dense` is the one-call fusion of
//! `from_data(make_tensor(..))`. These tests pin that equivalence and
//! the preferred-order tagging so the convenience constructor cannot
//! drift from the explicit two-step it replaces.

use ariadnetor_tensor::{ComputeBackend, ComputeBackendTensorExt, DenseTensor, Host};

#[test]
fn dense_matches_explicit_from_data_make_tensor() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];

    let fused = Host::shared().dense(data.clone(), shape.clone());
    let explicit = DenseTensor::from_data(Host::shared().make_tensor(data.clone(), shape.clone()));

    assert_eq!(fused.shape(), explicit.shape());
    assert_eq!(fused.order(), explicit.order());
    assert_eq!(fused.data_slice(), explicit.data_slice());
    assert_eq!(fused.data_slice(), &data[..]);
}

#[test]
fn dense_tags_backend_preferred_order() {
    let t = Host::shared().dense(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(t.order(), Host::shared().preferred_order());
}
