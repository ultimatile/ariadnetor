//! MPO-MPS apply operation tests.

use approx::assert_abs_diff_eq;
use arnet::mps::{
    self, ApplyMethod, CanonicalForm, Mpo, Mps, SvdAbsorb, TensorChain, TruncSvdParams,
    TruncateParams,
};
use arnet_tensor::Dense;

use super::helpers::{make_4site_mps, make_identity_mpo, mps_to_dense};

#[test]
fn test_apply_identity_preserves_state() {
    let psi = Mps::from_storages(vec![
        Dense::new(vec![1.0, 0.0], vec![1, 2, 1]),
        Dense::new(vec![0.0, 1.0], vec![1, 2, 1]),
        Dense::new(vec![1.0, 0.0], vec![1, 2, 1]),
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
        Dense::new(
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            vec![1, 2, 2, 2],
        ),
        Dense::new((1..=8).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 2, 1]),
    ];
    let mpo = Mpo::from_storages(mpo_storages);

    let psi = Mps::from_storages(vec![
        Dense::new(vec![1.0, 0.0, 0.5, 0.5], vec![1, 2, 2]),
        Dense::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1]),
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
        Dense::new(vec![1.0, 0.0, 0.5, 0.5], vec![1, 2, 2]),
        Dense::new((1..=8).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 2]),
        Dense::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1]),
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
    // Should be canonicalized (canonicalize + truncate was called)
    assert_eq!(*result.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn test_apply_sz_expectation() {
    // Apply Sz MPO to |0⟩, then compute ⟨0|Sz|0⟩ via inner product
    let up = Mps::from_storages(vec![Dense::new(vec![1.0, 0.0], vec![1, 2, 1])]);
    let sz_mpo = Mpo::from_storages(vec![Dense::new(
        vec![0.5, 0.0, 0.0, -0.5],
        vec![1, 2, 2, 1],
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

// ===========================================================================
// Zip-up algorithm tests
// ===========================================================================

/// 3-site MPS with bond dim 2 and physical dim 2. Deterministic content.
fn make_3site_test_mps() -> Mps<Dense<f64>> {
    Mps::from_storages(vec![
        Dense::new(vec![1.0, 0.0, 0.5, 0.5], vec![1, 2, 2]),
        Dense::new((1..=8).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 2]),
        Dense::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1]),
    ])
}

/// 3-site MPO with bond dim 2 and physical dim 2.
fn make_3site_test_mpo() -> Mpo<Dense<f64>> {
    Mpo::from_storages(vec![
        Dense::new((1..=8).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 2, 2]),
        Dense::new(
            (1..=16).map(|i| i as f64 * 0.05).collect(),
            vec![2, 2, 2, 2],
        ),
        Dense::new((1..=8).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 2, 1]),
    ])
}

fn assert_dense_close(a: &Dense<f64>, b: &Dense<f64>, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    for (i, (x, y)) in a.data().iter().zip(b.data().iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff < tol, "elem {i} mismatch: {x} vs {y} (diff {diff})");
    }
}

#[test]
fn test_apply_zipup_lossless_matches_naive_no_params() {
    // Forward QR pass alone is a gauge transformation: full state vector
    // must agree with the naive product elementwise.
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let phi_naive = mps::apply(&op, &psi, None);
    let phi_zipup = mps::apply_with_method(&op, &psi, None, ApplyMethod::ZipUp);

    let v_naive = mps_to_dense(&phi_naive);
    let v_zipup = mps_to_dense(&phi_zipup);
    assert_dense_close(&v_naive, &v_zipup, 1e-10);
}

#[test]
fn test_apply_zipup_lossless_matches_naive_large_chi() {
    // chi_max well above the inflated bond → no truncation in either path.
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();
    let lossless = TruncateParams::from(TruncSvdParams {
        chi_max: Some(64),
        target_trunc_err: None,
    });

    let phi_naive = mps::apply(&op, &psi, Some(&lossless));
    let phi_zipup = mps::apply_with_method(&op, &psi, Some(&lossless), ApplyMethod::ZipUp);

    let v_naive = mps_to_dense(&phi_naive);
    let v_zipup = mps_to_dense(&phi_zipup);
    assert_dense_close(&v_naive, &v_zipup, 1e-10);
}

#[test]
fn test_apply_zipup_identity_preserves_state() {
    let psi = make_3site_test_mps();
    let identity = make_identity_mpo(3, 2);

    let phi = mps::apply_with_method(&identity, &psi, None, ApplyMethod::ZipUp);

    let v_orig = mps_to_dense(&psi);
    let v_after = mps_to_dense(&phi);
    assert_dense_close(&v_orig, &v_after, 1e-10);
}

#[test]
fn test_apply_zipup_canonical_form() {
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    // No params → forward QR only, center at last site.
    let phi_none = mps::apply_with_method(&op, &psi, None, ApplyMethod::ZipUp);
    assert_eq!(
        *phi_none.canonical_form(),
        CanonicalForm::Mixed { center: 2 }
    );

    // With params → backward sweep moves the center to site 0.
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(8),
        target_trunc_err: None,
    });
    let phi_some = mps::apply_with_method(&op, &psi, Some(&params), ApplyMethod::ZipUp);
    assert_eq!(
        *phi_some.canonical_form(),
        CanonicalForm::Mixed { center: 0 }
    );
}

#[test]
fn test_apply_zipup_truncates_bond_dim() {
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let phi = mps::apply_with_method(&op, &psi, Some(&params), ApplyMethod::ZipUp);

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

/// Dispatch parity contract: every `TruncateParams` field that zip-up does
/// not yet honor must trigger an up-front panic. Silent divergence from the
/// naive path is forbidden.
///
/// When a new field is added to `TruncateParams` — or when zip-up gains
/// support for an existing one — extend the `unsupported` table below
/// accordingly. The point is to make the decision explicit at the test
/// boundary rather than discover the divergence in a downstream caller.
#[test]
fn test_apply_zipup_rejects_all_unsupported_truncate_params() {
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();
    let base = TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    };

    let n_minus_1 = psi.len() - 1;
    let unsupported: Vec<(&str, TruncateParams)> = vec![
        (
            "absorb=Left",
            TruncateParams {
                svd: base.clone(),
                absorb: SvdAbsorb::Left,
                center: None,
            },
        ),
        (
            "absorb=Both",
            TruncateParams {
                svd: base.clone(),
                absorb: SvdAbsorb::Both,
                center: None,
            },
        ),
        (
            "center=Some(1)",
            TruncateParams {
                svd: base.clone(),
                absorb: SvdAbsorb::Right,
                center: Some(1),
            },
        ),
        (
            "center=Some(N-1)",
            TruncateParams {
                svd: base.clone(),
                absorb: SvdAbsorb::Right,
                center: Some(n_minus_1),
            },
        ),
    ];

    for (name, params) in unsupported {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mps::apply_with_method(&op, &psi, Some(&params), ApplyMethod::ZipUp)
        }));
        assert!(
            result.is_err(),
            "expected apply_zipup to panic for unsupported params: {name}"
        );
    }
}

#[test]
fn test_apply_with_method_naive_dispatch_matches_apply() {
    // ApplyMethod::Naive must route through the existing apply path.
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });

    let phi_a = mps::apply(&op, &psi, Some(&params));
    let phi_b = mps::apply_with_method(&op, &psi, Some(&params), ApplyMethod::Naive);

    let v_a = mps_to_dense(&phi_a);
    let v_b = mps_to_dense(&phi_b);
    assert_dense_close(&v_a, &v_b, 1e-12);
}
