//! Truncate tests for block-sparse MPS.
//!
//! Covers `truncate_block_sparse` on a 4-site U(1)-symmetric chain.
//! Mirrors the Dense truncate test structure, adapted for BlockSparse
//! invariants (per-sector isometry, flux preservation, block-level
//! state comparison).

use arnet::mps::{
    CanonicalForm, Mps, SvdAbsorb, TensorChain, TruncSvdParams, TruncateParams,
    canonicalize_block_sparse, truncate_block_sparse,
};
use arnet_tensor::BlockSparse;
use arnet_tensor::U1Sector;

use super::helpers::{
    assert_block_sparse_close, bsp_mps_contract_full, is_left_canonical_bsp,
    is_right_canonical_bsp, make_2site_entangled_u1_mps, make_4site_u1_mps,
};

const TOL: f64 = 1e-10;

// --------------------------------------------------------------------------
// No-op truncation (chi_max large enough)
// --------------------------------------------------------------------------

#[test]
fn truncate_bsp_no_change_within_tolerance() {
    let mut mps = make_4site_u1_mps();
    canonicalize_block_sparse(&mut mps, 2);

    let state_before = bsp_mps_contract_full(&mps);
    let bond_dims_before = mps.bond_dims();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(100),
        target_trunc_err: None,
    });
    let result = truncate_block_sparse(&mut mps, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
    assert!(
        result.error < TOL,
        "expected near-zero error: {}",
        result.error
    );
    for (before, after) in bond_dims_before.iter().zip(mps.bond_dims().iter()) {
        assert!(*after <= *before);
    }

    let state_after = bsp_mps_contract_full(&mps);
    assert_block_sparse_close(&state_before, &state_after, TOL);
}

// --------------------------------------------------------------------------
// State preservation (approximate)
// --------------------------------------------------------------------------

#[test]
fn truncate_bsp_preserves_state_approximately() {
    let mut mps = make_2site_entangled_u1_mps();
    canonicalize_block_sparse(&mut mps, 0);
    let state_before = bsp_mps_contract_full(&mps);
    let norm_before = state_before.norm();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    truncate_block_sparse(&mut mps, &params);
    let state_after = bsp_mps_contract_full(&mps);
    let norm_after = state_after.norm();

    // Compute per-block overlap (normalized)
    let mut overlap = 0.0;
    for meta in state_before.block_metas() {
        let a = state_before.block_data(&meta.coord).unwrap();
        if let Some(b) = state_after.block_data(&meta.coord) {
            for (aa, bb) in a.iter().zip(b.iter()) {
                overlap += (aa / norm_before) * (bb / norm_after);
            }
        }
    }
    assert!(overlap > 0.5, "overlap too low: {overlap}");
}

// --------------------------------------------------------------------------
// Bond dimension reduction
// --------------------------------------------------------------------------

#[test]
fn truncate_bsp_reduces_bond_dim() {
    let mut mps = make_2site_entangled_u1_mps();
    canonicalize_block_sparse(&mut mps, 0);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = truncate_block_sparse(&mut mps, &params);

    for d in mps.bond_dims() {
        assert!(d <= 1, "bond dim {d} exceeds chi_max=1");
    }
    assert!(result.error > 0.0, "expected positive truncation error");
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

// --------------------------------------------------------------------------
// Canonical form and isometry checks
// --------------------------------------------------------------------------

#[test]
fn truncate_bsp_absorb_right_isometry() {
    let mut mps = make_4site_u1_mps();
    canonicalize_block_sparse(&mut mps, 2);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    truncate_block_sparse(&mut mps, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
    // Sites 0 and 1 left-canonical, site 2 is center, site 3 right-canonical
    for j in 0..2 {
        assert!(
            is_left_canonical_bsp(mps.storage(j), TOL),
            "site {j} not left-canonical after truncate (SvdAbsorb::Right)"
        );
    }
    for j in 3..mps.len() {
        assert!(
            is_right_canonical_bsp(mps.storage(j), TOL),
            "site {j} not right-canonical after truncate (SvdAbsorb::Right)"
        );
    }
}

#[test]
fn truncate_bsp_absorb_left_isometry() {
    let mut mps = make_4site_u1_mps();
    canonicalize_block_sparse(&mut mps, 1);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Left,
        center: None,
    };
    let result = truncate_block_sparse(&mut mps, &params);

    assert!(result.error >= 0.0);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 1 });

    // Site 0 left-canonical, sites 2..4 right-canonical
    assert!(
        is_left_canonical_bsp(mps.storage(0), TOL),
        "site 0 not left-canonical with SvdAbsorb::Left"
    );
    for j in 2..mps.len() {
        assert!(
            is_right_canonical_bsp(mps.storage(j), TOL),
            "site {j} not right-canonical with SvdAbsorb::Left"
        );
    }
}

#[test]
fn truncate_bsp_absorb_both_sets_unknown() {
    let mut mps = make_4site_u1_mps();
    canonicalize_block_sparse(&mut mps, 1);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Both,
        center: None,
    };
    let result = truncate_block_sparse(&mut mps, &params);

    assert!(result.error >= 0.0);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

// --------------------------------------------------------------------------
// Single-site chain
// --------------------------------------------------------------------------

#[test]
fn truncate_bsp_single_site() {
    use arnet_tensor::{BlockCoord, Direction, QNIndex};

    let left = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let phys = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
    let mut site = BlockSparse::<f64, U1Sector>::zeros(vec![left, phys, right], U1Sector(0));
    site.block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .expect("allowed block")[0] = 3.0;

    let mut mps: Mps<BlockSparse<f64, U1Sector>> = Mps::from_storages(vec![site]);
    canonicalize_block_sparse(&mut mps, 0);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = truncate_block_sparse(&mut mps, &params);

    assert!(result.error < TOL, "single-site should have zero error");
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

// --------------------------------------------------------------------------
// Auto-canonicalization
// --------------------------------------------------------------------------

#[test]
fn truncate_bsp_auto_canonicalizes_from_unknown() {
    let mut mps = make_4site_u1_mps();
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Right,
        center: Some(2),
    };
    let result = truncate_block_sparse(&mut mps, &params);

    assert!(result.error >= 0.0);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
    for d in mps.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

// --------------------------------------------------------------------------
// Truncation error
// --------------------------------------------------------------------------

#[test]
fn truncate_bsp_error_is_positive_when_truncated() {
    let mut mps = make_2site_entangled_u1_mps();
    canonicalize_block_sparse(&mut mps, 0);
    let norm_before = bsp_mps_contract_full(&mps).norm();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = truncate_block_sparse(&mut mps, &params);

    assert!(
        result.error > 0.0,
        "expected positive truncation error with chi_max=1"
    );
    assert!(
        result.error < norm_before,
        "truncation error {} exceeds norm {}",
        result.error,
        norm_before
    );
}
