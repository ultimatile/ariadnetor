//! MPO-MPS apply operation tests.

use approx::assert_abs_diff_eq;
use arnet::mps::{self, CanonicalForm, Mpo, Mps, TensorChain, TruncSvdParams, TruncateParams};
use arnet_tensor::{Dense, MemoryOrder};

use super::helpers::{make_4site_mps, make_identity_mpo, mps_to_dense};

#[test]
fn test_apply_identity_preserves_state() {
    let psi = Mps::from_storages(vec![
        Dense::from_data_with_order(vec![1.0, 0.0], vec![1, 2, 1], MemoryOrder::RowMajor),
        Dense::from_data_with_order(vec![0.0, 1.0], vec![1, 2, 1], MemoryOrder::RowMajor),
        Dense::from_data_with_order(vec![1.0, 0.0], vec![1, 2, 1], MemoryOrder::RowMajor),
    ]);
    let identity = make_identity_mpo(3, 2);

    let result = mps::apply(&identity, &psi, None);

    assert_eq!(result.len(), 3);

    // State vector should be the same
    let dense_orig = mps_to_dense(&psi);
    let dense_result = mps_to_dense(&result);
    for i in 0..dense_orig.len() {
        assert_abs_diff_eq!(
            dense_orig.data()[i],
            dense_result.data()[i],
            epsilon = 1e-12
        );
    }
}

#[test]
fn test_apply_increases_bond_dim() {
    // MPO with bond dim 2: doubles MPS bond dims
    let mpo_storages = vec![
        Dense::from_data_with_order(
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            vec![1, 2, 2, 2],
            MemoryOrder::RowMajor,
        ),
        Dense::from_data_with_order(
            (1..=8).map(|i| i as f64 * 0.1).collect(),
            vec![2, 2, 2, 1],
            MemoryOrder::RowMajor,
        ),
    ];
    let mpo = Mpo::from_storages(mpo_storages);

    let psi = Mps::from_storages(vec![
        Dense::from_data_with_order(
            vec![1.0, 0.0, 0.5, 0.5],
            vec![1, 2, 2],
            MemoryOrder::RowMajor,
        ),
        Dense::from_data_with_order(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2, 1],
            MemoryOrder::RowMajor,
        ),
    ]);

    let result = mps::apply(&mpo, &psi, None);

    assert_eq!(result.len(), 2);
    // Bond dim should be product of MPO and MPS bond dims
    // Original: bond 0 = 2, MPO bond 0 = 2 → fused = 4
    assert_eq!(result.bond_dim(0), 4);
}

#[test]
fn test_apply_with_truncation() {
    let psi = Mps::from_storages(vec![
        Dense::from_data_with_order(
            vec![1.0, 0.0, 0.5, 0.5],
            vec![1, 2, 2],
            MemoryOrder::RowMajor,
        ),
        Dense::from_data_with_order(
            (1..=8).map(|i| i as f64 * 0.1).collect(),
            vec![2, 2, 2],
            MemoryOrder::RowMajor,
        ),
        Dense::from_data_with_order(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2, 1],
            MemoryOrder::RowMajor,
        ),
    ]);
    let identity = make_identity_mpo(3, 2);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let result = mps::apply(&identity, &psi, Some(&params));

    // Bond dims should be capped at 2
    for d in result.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
    // Should be canonicalized (orthogonalize + truncate was called)
    assert_eq!(*result.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn test_apply_sz_expectation() {
    // Apply Sz MPO to |0⟩, then compute ⟨0|Sz|0⟩ via inner product
    let up = Mps::from_storages(vec![Dense::from_data_with_order(
        vec![1.0, 0.0],
        vec![1, 2, 1],
        MemoryOrder::RowMajor,
    )]);
    let sz_mpo = Mpo::from_storages(vec![Dense::from_data_with_order(
        vec![0.5, 0.0, 0.0, -0.5],
        vec![1, 2, 2, 1],
        MemoryOrder::RowMajor,
    )]);

    let sz_psi = mps::apply(&sz_mpo, &up, None);

    // ⟨0|Sz|0⟩ = inner(|0⟩, Sz|0⟩)
    let expect_val = mps::inner(&up, &sz_psi);
    assert_abs_diff_eq!(expect_val, 0.5, epsilon = 1e-12);
}

#[test]
fn test_apply_matches_expect() {
    let psi = make_4site_mps();
    let identity = make_identity_mpo(4, 2);

    // ⟨ψ|I|ψ⟩ via expect
    let expect_val = mps::braket(&psi, &identity, &psi);

    // ⟨ψ|I|ψ⟩ via apply + inner: inner(ψ, I·ψ)
    let i_psi = mps::apply(&identity, &psi, None);
    let apply_val = mps::inner(&psi, &i_psi);

    assert_abs_diff_eq!(expect_val, apply_val, epsilon = 1e-10);
}
