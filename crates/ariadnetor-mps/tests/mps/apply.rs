//! MPO-MPS apply operation tests.

use approx::assert_abs_diff_eq;
use ariadnetor_mps::{
    self as mps, ApplyMethod, CanonicalForm, Mpo, Mps, SvdAbsorb, TensorChain, TruncSvdParams,
    TruncateParams,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor};

use super::helpers::{
    cm_dense_tensor, dense_basis_site, make_4site_mps, make_identity_mpo, make_total_n_dense_mpo,
    mps_to_dense,
};

#[test]
fn test_apply_identity_preserves_state() {
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0, 0.0], vec![1, 2, 1]),
        cm_dense_tensor(vec![0.0, 1.0], vec![1, 2, 1]),
        cm_dense_tensor(vec![1.0, 0.0], vec![1, 2, 1]),
    ]);
    let identity = make_identity_mpo(3, 2);

    let result = mps::apply(&backend, &identity, &psi, None);

    assert_eq!(result.len(), 3);

    // State vector should be the same
    let dense_orig = mps_to_dense(&psi);
    let dense_result = mps_to_dense(&result);
    for i in 0..dense_orig.len() {
        assert_abs_diff_eq!(
            dense_orig.data_slice()[i],
            dense_result.data_slice()[i],
            epsilon = 1e-12
        );
    }
}

#[test]
fn test_apply_with_truncation() {
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0, 0.0, 0.5, 0.5], vec![1, 2, 2]),
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 2]),
        cm_dense_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1]),
    ]);
    let identity = make_identity_mpo(3, 2);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let result = mps::apply(&backend, &identity, &psi, Some(&params));

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
    let backend = NativeBackend::new();
    let up = Mps::from_sites(vec![cm_dense_tensor(vec![1.0, 0.0], vec![1, 2, 1])]);
    let sz_mpo = Mpo::from_sites(vec![cm_dense_tensor(
        vec![0.5, 0.0, 0.0, -0.5],
        vec![1, 2, 2, 1],
    )]);

    let sz_psi = mps::apply(&backend, &sz_mpo, &up, None);

    // ⟨0|Sz|0⟩ = inner(|0⟩, Sz|0⟩)
    let expect_val = mps::inner(&backend, &up, &sz_psi);
    assert_abs_diff_eq!(expect_val, 0.5, epsilon = 1e-12);
}

// ===========================================================================
// Analytical correctness anchors (mirror of apply_block_sparse.rs anchors)
// ===========================================================================

#[test]
fn test_apply_dense_total_n_mpo_acts_as_total_particle_number_2site_eigenstate() {
    // 2-site MPS in the total-N=1 subspace: ψ = 3|01⟩ + 8|10⟩.
    // Both basis vectors are N-eigenstates with eigenvalue 1, so
    // ⟨ψ|N|ψ⟩ = ⟨ψ|ψ⟩ = 9 + 64 = 73.
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![
        // Site 0 shape (1, 2, 2): bond carries the basis label.
        cm_dense_tensor(vec![3.0, 0.0, 0.0, 8.0], vec![1, 2, 2]),
        // Site 1 shape (2, 2, 1): bond=0 → phys=1 (|01⟩ branch),
        // bond=1 → phys=0 (|10⟩ branch).
        cm_dense_tensor(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2, 1]),
    ]);
    let n_op = make_total_n_dense_mpo(2);

    let psi_norm_sq = mps::inner(&backend, &psi, &psi);
    let n_psi = mps::apply(&backend, &n_op, &psi, None);
    let exp_n = mps::inner(&backend, &psi, &n_psi);

    assert_abs_diff_eq!(psi_norm_sq, 73.0, epsilon = 1e-10);
    assert_abs_diff_eq!(exp_n, 73.0, epsilon = 1e-10);
}

#[test]
fn test_apply_dense_total_n_mpo_3site_interior() {
    // 3-site basis state |010⟩: single particle at site 1, total N = 1.
    // Exercises one interior MPO site, where RowMajor and ColumnMajor
    // bond-fusion layouts disagree on the off-diagonal "I → n" entry.
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![
        dense_basis_site(0),
        dense_basis_site(1),
        dense_basis_site(0),
    ]);
    let n_op = make_total_n_dense_mpo(3);

    let psi_norm_sq = mps::inner(&backend, &psi, &psi);
    let n_psi = mps::apply(&backend, &n_op, &psi, None);
    let exp_n = mps::inner(&backend, &psi, &n_psi);

    assert_abs_diff_eq!(psi_norm_sq, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(exp_n, 1.0, epsilon = 1e-10);
}

#[test]
fn test_apply_dense_n_on_zero_state() {
    // |0000⟩ has total N = 0. Anchors the right-edge boundary
    // (bL=I → apply n_phys = 0 at charge 0) along the all-zero path.
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![
        dense_basis_site(0),
        dense_basis_site(0),
        dense_basis_site(0),
        dense_basis_site(0),
    ]);
    let n_op = make_total_n_dense_mpo(4);

    let psi_norm_sq = mps::inner(&backend, &psi, &psi);
    let n_psi = mps::apply(&backend, &n_op, &psi, None);
    let exp_n = mps::inner(&backend, &psi, &n_psi);

    assert_abs_diff_eq!(psi_norm_sq, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(exp_n, 0.0, epsilon = 1e-10);
}

#[test]
fn test_apply_dense_n_eigenvalue_on_multi_particle_basis_state() {
    // |1010⟩ on 4 sites: total N = 2. Two interior MPO sites are exercised
    // simultaneously (sites 1 and 2). The total-N contraction sums over
    // FSM paths where the single I → n transition can fire at any site,
    // so the eigenvalue is collected from the occupied sites 0 and 2.
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![
        dense_basis_site(1),
        dense_basis_site(0),
        dense_basis_site(1),
        dense_basis_site(0),
    ]);
    let n_op = make_total_n_dense_mpo(4);

    let psi_norm_sq = mps::inner(&backend, &psi, &psi);
    let n_psi = mps::apply(&backend, &n_op, &psi, None);
    let exp_n = mps::inner(&backend, &psi, &n_psi);

    assert_abs_diff_eq!(psi_norm_sq, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(exp_n, 2.0, epsilon = 1e-10);
}

#[test]
fn test_apply_dense_n_squared_via_composition() {
    // |11⟩ on 2 sites is an N-eigenstate with eigenvalue 2, so
    // ⟨ψ|N²|ψ⟩ = 4. Re-feeding the apply output back into apply
    // verifies that the result is a well-formed MPS the operator can
    // act on again — the algebraic eigenvalue identity acts as the
    // analytical anchor across the composition.
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![dense_basis_site(1), dense_basis_site(1)]);
    let n_op = make_total_n_dense_mpo(2);

    let n_psi = mps::apply(&backend, &n_op, &psi, None);
    let nn_psi = mps::apply(&backend, &n_op, &n_psi, None);
    let exp_n_sq = mps::inner(&backend, &psi, &nn_psi);

    assert_abs_diff_eq!(exp_n_sq, 4.0, epsilon = 1e-10);
}

#[test]
fn test_apply_matches_expect() {
    let backend = NativeBackend::new();
    let psi = make_4site_mps();
    let identity = make_identity_mpo(4, 2);

    // ⟨ψ|I|ψ⟩ via expect
    let expect_val = mps::braket(&backend, &psi, &identity, &psi);

    // ⟨ψ|I|ψ⟩ via apply + inner: inner(ψ, I·ψ)
    let i_psi = mps::apply(&backend, &identity, &psi, None);
    let apply_val = mps::inner(&backend, &psi, &i_psi);

    assert_abs_diff_eq!(expect_val, apply_val, epsilon = 1e-10);
}

// ===========================================================================
// Streaming-naive algorithm tests
// ===========================================================================

/// 3-site MPS with bond dim 2 and physical dim 2. Deterministic content.
fn make_3site_test_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0, 0.0, 0.5, 0.5], vec![1, 2, 2]),
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 2]),
        cm_dense_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1]),
    ])
}

/// 3-site MPO with bond dim 2 and physical dim 2.
fn make_3site_test_mpo() -> Mpo<DenseStorage<f64>, DenseLayout> {
    Mpo::from_sites(vec![
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 2, 2]),
        cm_dense_tensor(
            (1..=16).map(|i| i as f64 * 0.05).collect(),
            vec![2, 2, 2, 2],
        ),
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 2, 1]),
    ])
}

fn assert_dense_close(a: &DenseTensor<f64>, b: &DenseTensor<f64>, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    for (i, (x, y)) in a.data_slice().iter().zip(b.data_slice().iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff < tol, "elem {i} mismatch: {x} vs {y} (diff {diff})");
    }
}

#[test]
fn test_apply_streaming_naive_identity_preserves_state() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let identity = make_identity_mpo(3, 2);

    let phi = mps::apply(&backend, &identity, &psi, None);

    let v_orig = mps_to_dense(&psi);
    let v_after = mps_to_dense(&phi);
    assert_dense_close(&v_orig, &v_after, 1e-10);
}

#[test]
fn test_apply_streaming_naive_canonical_form() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    // No params: the forward QR sweep leaves the chain at center = n - 1.
    let phi_none = mps::apply(&backend, &op, &psi, None);
    assert_eq!(
        *phi_none.canonical_form(),
        CanonicalForm::Mixed { center: 2 }
    );

    // With params: canonicalize + truncate finishing parks the center at 0
    // by default (params.center == None → 0).
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(8),
        target_trunc_err: None,
    });
    let phi_some = mps::apply(&backend, &op, &psi, Some(&params));
    assert_eq!(
        *phi_some.canonical_form(),
        CanonicalForm::Mixed { center: 0 }
    );
}

#[test]
fn test_apply_streaming_naive_truncates_bond_dim() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let phi = mps::apply(&backend, &op, &psi, Some(&params));

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

/// Anchor: with `chi_max = 1`, the result pins the dominant Schmidt direction
/// of MPO·MPS and the final canonical form sits at the default center.
/// Mutating the forward sweep boundary or the post-sweep canonical-form tag
/// shifts the truncation gauge and trips this assertion.
#[test]
fn test_apply_streaming_naive_dense_truncated_chi1_baseline() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });

    let phi = mps::apply(&backend, &op, &psi, Some(&params));

    for d in phi.bond_dims() {
        assert!(d <= 1, "bond {d} exceeds chi_max=1");
    }
    assert_eq!(*phi.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

// ===========================================================================
// SvdAbsorb::{Left, Both} and arbitrary `params.center` support
// ===========================================================================

#[test]
fn test_apply_streaming_naive_absorb_left_yields_mixed_center() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let identity = make_identity_mpo(3, 2);
    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(4),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Left,
        center: None,
    };

    let phi = mps::apply(&backend, &identity, &psi, Some(&params));

    // absorb=Left parks the orthogonality center at the default site 0,
    // distinct from absorb=Both which leaves the chain in `Unknown`.
    assert_eq!(*phi.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn test_apply_streaming_naive_absorb_both_yields_unknown_canonical_form() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let identity = make_identity_mpo(3, 2);
    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(4),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Both,
        center: None,
    };

    let phi = mps::apply(&backend, &identity, &psi, Some(&params));

    // sqrt(S) on both sides leaves no per-site isometry; truncate_dense
    // records the canonical form as Unknown.
    assert_eq!(*phi.canonical_form(), CanonicalForm::Unknown);
}

#[test]
fn test_apply_streaming_naive_center_nonzero_parks_center_at_request() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let identity = make_identity_mpo(3, 2);
    let svd = TruncSvdParams {
        chi_max: Some(4),
        target_trunc_err: None,
    };

    for c in [1usize, 2usize] {
        let params = TruncateParams {
            svd: svd.clone(),
            absorb: SvdAbsorb::Right,
            center: Some(c),
        };
        let phi = mps::apply(&backend, &identity, &psi, Some(&params));
        assert_eq!(
            *phi.canonical_form(),
            CanonicalForm::Mixed { center: c },
            "params.center = Some({c}) should park the center at site {c}"
        );
    }
}

// ===========================================================================
// forward_cap exposure
// ===========================================================================

/// Tight `forward_cap` (factor 1) triggers the SVD branch at site 1 of the
/// 3-site fixture (natural forward rank = min(2*2, 4) = 4 > 1 * chi_max = 2),
/// so the intermediate is pre-truncated; the final chain still respects
/// `chi_max` end-to-end.
#[test]
fn test_apply_streaming_naive_forward_cap_factor_one_keeps_chi_max() {
    use std::num::NonZeroUsize;

    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let method = ApplyMethod::StreamingNaive {
        forward_cap: Some(NonZeroUsize::new(1).unwrap()),
    };

    let phi = mps::apply_with_method(&backend, &op, &psi, Some(&params), method);

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond {d} exceeds chi_max=2 under forward_cap=1");
    }
}

/// Tightening `forward_cap` from `None` to `Some(1)` must be observable on
/// this fixture: the forward SVD branch at site 1 (rank 4, cap 2) drops
/// singular vectors that the backward `chi_max = 2` sweep would otherwise
/// weigh into its truncation, so the two output states differ by more than
/// roundoff. The capped output is also no closer to the effectively-lossless
/// `chi_max = 64` reference than the lossless-forward output (monotonicity).
///
/// If `forward_cap` were silently ignored, both branches would produce
/// numerically identical output and the observable-difference assertion
/// would fail.
#[test]
fn test_apply_streaming_naive_forward_cap_observably_changes_output() {
    use std::num::NonZeroUsize;

    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();
    let chi_2 = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let ref_params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(64),
        target_trunc_err: None,
    });

    let phi_ref = mps::apply(&backend, &op, &psi, Some(&ref_params));
    let phi_lossless_forward = mps::apply(&backend, &op, &psi, Some(&chi_2));
    let phi_capped_forward = mps::apply_with_method(
        &backend,
        &op,
        &psi,
        Some(&chi_2),
        ApplyMethod::StreamingNaive {
            forward_cap: Some(NonZeroUsize::new(1).unwrap()),
        },
    );

    let v_ref = mps_to_dense(&phi_ref);
    let v_lossless = mps_to_dense(&phi_lossless_forward);
    let v_capped = mps_to_dense(&phi_capped_forward);

    // 1. The two outputs must differ by more than roundoff. This fails if
    //    `forward_cap` is ignored (both branches would take QR).
    let observed_diff = dense_frobenius_distance(&v_lossless, &v_capped);
    assert!(
        observed_diff > 1e-8,
        "forward_cap = Some(1) and forward_cap = None should produce \
         observably different states on this fixture; got Frobenius \
         distance {observed_diff}"
    );

    // 2. Monotonicity: tightening the cap can only move the result away
    //    from the high-chi reference. The `+ 1e-12` slack allows numerical
    //    equality without permitting a regression of the direction.
    let dist_lossless = dense_frobenius_distance(&v_lossless, &v_ref);
    let dist_capped = dense_frobenius_distance(&v_capped, &v_ref);
    assert!(
        dist_capped + 1e-12 >= dist_lossless,
        "monotonicity violated: capped={dist_capped} < lossless-forward={dist_lossless}"
    );
}

fn dense_frobenius_distance(a: &DenseTensor<f64>, b: &DenseTensor<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape(), "shape mismatch in frobenius_distance");
    a.data_slice()
        .iter()
        .zip(b.data_slice().iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// The default `apply()` and an explicit `ApplyMethod::default()` dispatch
/// must produce numerically identical output. Anchors the
/// `apply == apply_with_method(..., default())` contract.
#[test]
fn test_apply_streaming_naive_default_method_matches_free_apply() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });

    let phi_apply = mps::apply(&backend, &op, &psi, Some(&params));
    let phi_method =
        mps::apply_with_method(&backend, &op, &psi, Some(&params), ApplyMethod::default());

    let v_apply = mps_to_dense(&phi_apply);
    let v_method = mps_to_dense(&phi_method);
    assert_dense_close(&v_apply, &v_method, 1e-12);
}

// ===========================================================================
// Zip-up algorithm tests
// ===========================================================================

#[test]
fn test_apply_zipup_identity_preserves_state() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let identity = make_identity_mpo(3, 2);

    let phi = mps::apply_with_method(&backend, &identity, &psi, None, ApplyMethod::ZipUp);

    let v_orig = mps_to_dense(&psi);
    let v_after = mps_to_dense(&phi);
    assert_dense_close(&v_orig, &v_after, 1e-10);
}

/// With no truncation, the zip-up sweep is a lossless refactoring of the
/// exact MPO·MPS product, so its state vector matches the lossless
/// streaming-naive baseline.
#[test]
fn test_apply_zipup_lossless_matches_streaming_naive() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let phi_zipup = mps::apply_with_method(&backend, &op, &psi, None, ApplyMethod::ZipUp);
    let phi_baseline = mps::apply(&backend, &op, &psi, None);

    let v_zipup = mps_to_dense(&phi_zipup);
    let v_baseline = mps_to_dense(&phi_baseline);
    assert_dense_close(&v_zipup, &v_baseline, 1e-10);
}

#[test]
fn test_apply_zipup_truncates_bond_dim() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let phi = mps::apply_with_method(&backend, &op, &psi, Some(&params), ApplyMethod::ZipUp);

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

/// Zip-up ends its forward sweep left-canonical, parking the center at the
/// last site regardless of whether truncation ran.
#[test]
fn test_apply_zipup_canonical_form() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let phi_none = mps::apply_with_method(&backend, &op, &psi, None, ApplyMethod::ZipUp);
    assert_eq!(
        *phi_none.canonical_form(),
        CanonicalForm::Mixed { center: 2 }
    );

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let phi_trunc = mps::apply_with_method(&backend, &op, &psi, Some(&params), ApplyMethod::ZipUp);
    assert_eq!(
        *phi_trunc.canonical_form(),
        CanonicalForm::Mixed { center: 2 }
    );
}

/// `params.center` is documented as not consulted by zip-up. Passing an
/// explicit center must not move the result's orthogonality center off the
/// last site.
#[test]
fn test_apply_zipup_ignores_params_center() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::default(),
        center: Some(0),
    };
    let phi = mps::apply_with_method(&backend, &op, &psi, Some(&params), ApplyMethod::ZipUp);
    assert_eq!(*phi.canonical_form(), CanonicalForm::Mixed { center: 2 });
}
