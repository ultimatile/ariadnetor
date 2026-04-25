//! apply tests: apply a BlockSparse MPO to a BlockSparse MPS.

use approx::assert_abs_diff_eq;
use arnet::mps::{
    self, ApplyMethod, CanonicalForm, Mpo, Mps, SvdAbsorb, TensorChain, TruncSvdParams,
    TruncateParams, apply, inner, norm,
};
use arnet_tensor::{BlockSparse, Direction, U1Sector};

use super::helpers::{
    assert_block_sparse_close, bsp_mps_contract_full, make_2site_entangled_u1_mps,
    make_4site_u1_mps, make_identity_u1_mpo,
};

// ---------------------------------------------------------------------------
// Identity MPO preserves MPS
// ---------------------------------------------------------------------------

#[test]
fn identity_mpo_preserves_norm_4site() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply(&identity, &mps, None);

    let norm_before = norm(&mps);
    let norm_after = norm(&result);
    assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-10);
}

#[test]
fn identity_mpo_preserves_inner_product() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply(&identity, &mps, None);

    let inner_psi_psi = inner(&mps, &mps);
    let inner_psi_result = inner(&mps, &result);
    assert_abs_diff_eq!(inner_psi_psi, inner_psi_result, epsilon = 1e-10);
}

#[test]
fn identity_mpo_preserves_entangled_state() {
    let mps = make_2site_entangled_u1_mps();
    let identity = make_identity_u1_mpo(2);

    let result = apply(&identity, &mps, None);

    let inner_before = inner(&mps, &mps);
    let inner_after = inner(&result, &result);
    assert_abs_diff_eq!(inner_before, inner_after, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// Output structure
// ---------------------------------------------------------------------------

#[test]
fn output_is_rank3_mps() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply(&identity, &mps, None);

    assert_eq!(result.len(), mps.len());
    for j in 0..result.len() {
        assert_eq!(result.storage(j).rank(), 3, "site {j} should be rank-3");
        let indices = result.storage(j).indices();
        assert_eq!(indices[0].direction(), Direction::Out, "site {j} left bond");
        assert_eq!(indices[1].direction(), Direction::Out, "site {j} physical");
        assert_eq!(indices[2].direction(), Direction::In, "site {j} right bond");
    }
}

#[test]
fn output_flux_preserved() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply(&identity, &mps, None);

    for j in 0..result.len() {
        assert_eq!(
            result.storage(j).flux(),
            &U1Sector(0),
            "site {j} flux should be 0"
        );
    }
}

// ---------------------------------------------------------------------------
// With truncation
// ---------------------------------------------------------------------------

#[test]
fn apply_with_truncation() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(4),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Right,
        center: Some(0),
    };

    let result = apply(&identity, &mps, Some(&params));

    assert_eq!(result.len(), 4);
    for j in 0..result.len() {
        assert_eq!(result.storage(j).rank(), 3);
        for meta in result.storage(j).block_metas() {
            let data = result.storage(j).block_data(&meta.coord).unwrap();
            for &v in data {
                assert!(v.is_finite(), "site {j} has non-finite value");
            }
        }
    }

    // Canonical form should be Mixed { center: 0 } after canonicalize + truncate
    assert_eq!(*result.canonical_form(), CanonicalForm::Mixed { center: 0 });

    // Norm should be approximately preserved (identity MPO, only truncation error)
    let norm_before = norm(&mps);
    let norm_after = norm(&result);
    assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// Panics
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "MPO and MPS lengths must match")]
fn length_mismatch_panics() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(3);
    apply(&identity, &mps, None);
}

#[test]
#[should_panic(expected = "must have at least one site")]
fn empty_mps_panics() {
    let mps = Mps::<BlockSparse<f64, U1Sector>>::from_storages(vec![]);
    let mpo = Mpo::<BlockSparse<f64, U1Sector>>::from_storages(vec![]);
    apply(&mpo, &mps, None);
}

// ===========================================================================
// Zip-up algorithm tests (BlockSparse)
// ===========================================================================

#[test]
fn zipup_lossless_matches_naive_no_params() {
    // Forward QR pass alone is a gauge transformation: the contracted full
    // tensor must agree with the naive product, charge-by-charge.
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let phi_naive = apply(&identity, &psi, None);
    let phi_zipup = mps::apply_with_method(&identity, &psi, None, ApplyMethod::ZipUp);

    let v_naive = bsp_mps_contract_full(&phi_naive);
    let v_zipup = bsp_mps_contract_full(&phi_zipup);
    assert_block_sparse_close(&v_naive, &v_zipup, 1e-10);
}

#[test]
fn zipup_lossless_matches_naive_large_chi() {
    // chi_max above any inflated bond → no truncation in either path.
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);
    let lossless = TruncateParams::from(TruncSvdParams {
        chi_max: Some(64),
        target_trunc_err: None,
    });

    let phi_naive = apply(&identity, &psi, Some(&lossless));
    let phi_zipup = mps::apply_with_method(&identity, &psi, Some(&lossless), ApplyMethod::ZipUp);

    let v_naive = bsp_mps_contract_full(&phi_naive);
    let v_zipup = bsp_mps_contract_full(&phi_zipup);
    assert_block_sparse_close(&v_naive, &v_zipup, 1e-10);
}

#[test]
fn zipup_identity_preserves_state() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let phi = mps::apply_with_method(&identity, &psi, None, ApplyMethod::ZipUp);

    let v_before = bsp_mps_contract_full(&psi);
    let v_after = bsp_mps_contract_full(&phi);
    assert_block_sparse_close(&v_before, &v_after, 1e-10);
}

#[test]
fn zipup_canonical_form() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let phi_none = mps::apply_with_method(&identity, &psi, None, ApplyMethod::ZipUp);
    assert_eq!(
        *phi_none.canonical_form(),
        CanonicalForm::Mixed { center: 3 }
    );

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(8),
        target_trunc_err: None,
    });
    let phi_some = mps::apply_with_method(&identity, &psi, Some(&params), ApplyMethod::ZipUp);
    assert_eq!(
        *phi_some.canonical_form(),
        CanonicalForm::Mixed { center: 0 }
    );
}

#[test]
fn zipup_output_structure_and_flux() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let phi = mps::apply_with_method(&identity, &psi, None, ApplyMethod::ZipUp);

    assert_eq!(phi.len(), psi.len());
    for j in 0..phi.len() {
        let site = phi.storage(j);
        assert_eq!(site.rank(), 3, "site {j} should be rank-3");
        let indices = site.indices();
        assert_eq!(indices[0].direction(), Direction::Out, "site {j} left bond");
        assert_eq!(indices[1].direction(), Direction::Out, "site {j} physical");
        assert_eq!(indices[2].direction(), Direction::In, "site {j} right bond");
        assert_eq!(site.flux(), &U1Sector(0), "site {j} flux");
    }
}

#[test]
fn zipup_truncates_bond_dim() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let phi = mps::apply_with_method(&identity, &psi, Some(&params), ApplyMethod::ZipUp);

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

/// Block-sparse mirror of the dense dispatch parity contract: every
/// `TruncateParams` field zip-up does not yet honor for `BlockSparse` must
/// trigger an up-front panic. Extend the `unsupported` table when zip-up
/// gains support for a field.
#[test]
fn zipup_rejects_all_unsupported_truncate_params() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);
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
            mps::apply_with_method(&identity, &psi, Some(&params), ApplyMethod::ZipUp)
        }));
        assert!(
            result.is_err(),
            "expected apply_zipup to panic for unsupported params: {name}"
        );
    }
}
