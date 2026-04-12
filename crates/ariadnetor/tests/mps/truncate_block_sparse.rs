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
use arnet_tensor::block_sparse::BlockSparse;
use arnet_tensor::sector::U1Sector;

use super::helpers::{
    assert_block_sparse_close, bsp_mps_contract_full, is_left_canonical_bsp,
    is_right_canonical_bsp, make_4site_u1_mps,
};

/// Frobenius norm of a block-sparse tensor (sum of squared elements, then sqrt).
fn bsp_frobenius_norm(t: &BlockSparse<f64, U1Sector>) -> f64 {
    let mut sum_sq = 0.0;
    for meta in t.block_metas() {
        for &v in t.block_data(&meta.coord).unwrap() {
            sum_sq += v * v;
        }
    }
    sum_sq.sqrt()
}

/// Build a 2-site U(1)-symmetric MPS in the total-charge-1 sector.
///
/// Physical charges {0, 1}, boundary left={0:1}, right={1:1}.
/// The state spans two basis vectors: |01⟩ (coeff 3) and |10⟩ (coeff 8),
/// giving bond dim 2 with two non-zero singular values — genuine
/// entanglement that truncation can meaningfully discard.
fn make_2site_entangled_u1_mps() -> Mps<BlockSparse<f64, U1Sector>> {
    use arnet_tensor::block_sparse::{BlockCoord, Direction, QNIndex};

    // Site 0: left{0:1}(Out), phys{0:1,1:1}(Out), right{0:1,1:1}(In), flux=0
    let left0 = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let phys0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let mut site0 = BlockSparse::<f64, U1Sector>::zeros(vec![left0, phys0, right0], U1Sector(0));
    // Block (0,0,0): left=0 + phys=0 - right=0 = 0 ✓
    site0.block_data_mut(&BlockCoord(vec![0, 0, 0])).unwrap()[0] = 1.0;
    // Block (0,1,1): left=0 + phys=1 - right=1 = 0 ✓
    site0.block_data_mut(&BlockCoord(vec![0, 1, 1])).unwrap()[0] = 2.0;

    // Site 1: left{0:1,1:1}(Out), phys{0:1,1:1}(Out), right{1:1}(In), flux=0
    let left1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let phys1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right1 = QNIndex::new(vec![(U1Sector(1), 1)], Direction::In);
    let mut site1 = BlockSparse::<f64, U1Sector>::zeros(vec![left1, phys1, right1], U1Sector(0));
    // Block (0,1,0): left=0 + phys=1 - right=1 = 0 ✓  (|01⟩ path)
    site1.block_data_mut(&BlockCoord(vec![0, 1, 0])).unwrap()[0] = 3.0;
    // Block (1,0,0): left=1 + phys=0 - right=1 = 0 ✓  (|10⟩ path)
    site1.block_data_mut(&BlockCoord(vec![1, 0, 0])).unwrap()[0] = 4.0;

    Mps::from_storages(vec![site0, site1])
}

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
    let norm_before = bsp_frobenius_norm(&state_before);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    truncate_block_sparse(&mut mps, &params);
    let state_after = bsp_mps_contract_full(&mps);
    let norm_after = bsp_frobenius_norm(&state_after);

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
    // Sites 0..2 left-canonical, site 3 right-canonical
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
    use arnet_tensor::block_sparse::{BlockCoord, Direction, QNIndex};

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
    let norm_before = bsp_frobenius_norm(&bsp_mps_contract_full(&mps));

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
