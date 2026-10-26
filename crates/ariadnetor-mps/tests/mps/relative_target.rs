//! A relative truncation target passed through the MPS operations that
//! forward `TruncSvdParams` to the truncated SVD: scaling the state scales
//! every matrix truncated, so the kept bonds stay the same.

use ariadnetor_mps::{
    ApplyMethod, Mps, TensorChain, TruncErrTarget, TruncSvdParams, TruncateParams,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor};

use super::helpers::{apply_ok, cm_dense_tensor, make_3site_test_mpo, with_site_scaled};

type DenseMps = Mps<DenseStorage<f64>, DenseLayout>;

/// A site whose entry at flat index `i` is `sin(1.3 (i + offset))`.
fn generic_site(shape: Vec<usize>, offset: usize) -> DenseTensor<f64> {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|i| ((i + offset) as f64 * 1.3).sin())
        .collect();
    cm_dense_tensor(data, shape)
}

fn relative(fraction: f64) -> TruncateParams {
    TruncateParams::from(TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(TruncErrTarget::Relative(fraction)),
    })
}

#[test]
fn truncate_keeps_the_same_bonds_at_any_scale() {
    let backend = NativeBackend::new();
    let small: DenseMps = Mps::from_sites(vec![
        generic_site(vec![1, 2, 2], 0),
        generic_site(vec![2, 2, 4], 10),
        generic_site(vec![4, 2, 2], 30),
        generic_site(vec![2, 2, 1], 50),
    ]);
    let large = with_site_scaled(&small, 1, 1000.0);
    let params = relative(0.2);

    let (mut a, mut b) = (small.clone(), large);
    let ra = a.truncate(&backend, &params);
    let rb = b.truncate(&backend, &params);
    assert!(
        a.bond_dims() != small.bond_dims(),
        "fixture must actually truncate: {:?}",
        a.bond_dims()
    );
    assert_eq!(a.bond_dims(), b.bond_dims());
    assert!(
        (rb.error - 1000.0 * ra.error).abs() < 1e-9 * rb.error,
        "{} vs {}",
        ra.error,
        rb.error
    );
}

#[test]
fn zip_up_keeps_the_same_bonds_at_any_scale() {
    let backend = NativeBackend::new();
    let op = make_3site_test_mpo();
    let small: DenseMps = Mps::from_sites(vec![
        generic_site(vec![1, 2, 2], 0),
        generic_site(vec![2, 2, 2], 10),
        generic_site(vec![2, 2, 1], 30),
    ]);
    let large = with_site_scaled(&small, 0, 1000.0);
    let params = relative(0.2);

    let lossless = apply_ok(&backend, &op, &small, None, ApplyMethod::ZipUp);
    let a = apply_ok(&backend, &op, &small, Some(&params), ApplyMethod::ZipUp);
    let b = apply_ok(&backend, &op, &large, Some(&params), ApplyMethod::ZipUp);
    assert!(
        a.bond_dims() != lossless.bond_dims(),
        "fixture must actually truncate: {:?}",
        a.bond_dims()
    );
    assert_eq!(a.bond_dims(), b.bond_dims());
}
