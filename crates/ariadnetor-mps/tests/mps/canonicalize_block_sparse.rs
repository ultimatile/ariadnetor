//! Canonicalize tests for block-sparse MPS.
//!
//! Covers `canonicalize` on a 4-site U(1)-symmetric chain with
//! non-trivial per-sector bond structure. The fixture is designed so that at
//! least one QR/LQ factorization on every site is genuinely non-trivial
//! (multi-element sector blocks), which is essential for catching mutants in
//! the per-sector sweep logic.

use arnet_mps::{CanonicalForm, Mps, TensorChain, canonicalize};
use arnet_native::NativeBackend;
use arnet_tensor::{BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, U1Sector};

use super::helpers::{
    assert_block_sparse_close, bsp_mps_contract_full, is_left_canonical_bsp,
    is_right_canonical_bsp, make_4site_u1_mps,
};

const TOL: f64 = 1e-10;

// --------------------------------------------------------------------------
// Canonical form transitions
// --------------------------------------------------------------------------

#[test]
fn canonicalize_bsp_sets_mixed_form_from_unknown() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_u1_mps();
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);

    canonicalize(&backend, &mut mps, 2);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
}

// --------------------------------------------------------------------------
// Per-center isometry structure
// --------------------------------------------------------------------------

#[test]
fn canonicalize_bsp_center_0_all_right_isometric() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_u1_mps();
    canonicalize(&backend, &mut mps, 0);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
    // All sites past the center must be right-canonical.
    for j in 1..mps.len() {
        assert!(
            is_right_canonical_bsp(mps.site(j), TOL),
            "site {j} is not right-canonical after canonicalize(center=0)"
        );
    }
}

#[test]
fn canonicalize_bsp_center_last_all_left_isometric() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_u1_mps();
    let last = mps.len() - 1;
    canonicalize(&backend, &mut mps, last);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: last });
    // Sites 0..last must be left-canonical.
    for j in 0..last {
        assert!(
            is_left_canonical_bsp(mps.site(j), TOL),
            "site {j} is not left-canonical after canonicalize(center=last)"
        );
    }
}

#[test]
fn canonicalize_bsp_center_middle_has_mixed_isometry() {
    let backend = NativeBackend::new();
    let mut mps = make_4site_u1_mps();
    canonicalize(&backend, &mut mps, 2);

    // 0..2 left-canonical, 3..4 right-canonical; site 2 is the orthogonality center.
    for j in 0..2 {
        assert!(
            is_left_canonical_bsp(mps.site(j), TOL),
            "site {j} is not left-canonical after canonicalize(center=2)"
        );
    }
    for j in 3..mps.len() {
        assert!(
            is_right_canonical_bsp(mps.site(j), TOL),
            "site {j} is not right-canonical after canonicalize(center=2)"
        );
    }
}

// --------------------------------------------------------------------------
// State preservation (primary correctness witness)
// --------------------------------------------------------------------------

#[test]
fn canonicalize_bsp_preserves_full_chain_state_center_0() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    let state_before = bsp_mps_contract_full(&mps);

    let mut mps_after = mps.clone();
    canonicalize(&backend, &mut mps_after, 0);
    let state_after = bsp_mps_contract_full(&mps_after);

    assert_block_sparse_close(&state_before, &state_after, TOL);
}

#[test]
fn canonicalize_bsp_preserves_full_chain_state_center_middle() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    let state_before = bsp_mps_contract_full(&mps);

    let mut mps_after = mps.clone();
    canonicalize(&backend, &mut mps_after, 2);
    let state_after = bsp_mps_contract_full(&mps_after);

    assert_block_sparse_close(&state_before, &state_after, TOL);
}

#[test]
fn canonicalize_bsp_preserves_full_chain_state_center_last() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    let state_before = bsp_mps_contract_full(&mps);

    let mut mps_after = mps.clone();
    let last = mps_after.len() - 1;
    canonicalize(&backend, &mut mps_after, last);
    let state_after = bsp_mps_contract_full(&mps_after);

    assert_block_sparse_close(&state_before, &state_after, TOL);
}

// --------------------------------------------------------------------------
// Zero-flux fixture is preserved through canonicalize
// --------------------------------------------------------------------------

/// `canonicalize` accepts arbitrary per-site flux, but the
/// standard MPS convention — and the fixture used throughout this file — is
/// that every site carries identity flux. This test pins that the fixture
/// really starts at identity and that canonicalize leaves the labelling
/// unchanged for the zero-flux case. Charged chains are covered separately
/// by `canonicalize_bsp_accepts_charged_single_site`.
#[test]
fn canonicalize_bsp_zero_flux_chain_stays_identity_flux() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    // Precondition: fixture really is a zero-flux chain. If the fixture
    // ever changes to carry charge, this test is no longer meaningful.
    for j in 0..mps.len() {
        assert_eq!(
            *mps.site(j).flux(),
            U1Sector(0),
            "fixture site {j} unexpectedly has non-identity flux"
        );
    }

    let mut mps_after = mps;
    canonicalize(&backend, &mut mps_after, 2);

    for j in 0..mps_after.len() {
        assert_eq!(
            *mps_after.site(j).flux(),
            U1Sector(0),
            "site {j} flux changed through canonicalize of a zero-flux chain"
        );
    }
}

/// A charged single-site "chain" is already a valid mixed-canonical form
/// (there are no bonds to isometrize), so `canonicalize` must
/// accept non-identity flux there and leave the site data untouched —
/// only the canonical-form tag should flip. This regression test also pins
/// the contract that charged input is not silently rejected.
#[test]
fn canonicalize_bsp_accepts_charged_single_site() {
    use arnet_tensor::{BlockCoord, Direction, QNIndex};

    let backend = NativeBackend::new();

    let left = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let phys = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
    let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(vec![left, phys, right], U1Sector(1));
    // flux=1 forces the unique allowed block to be (left=0, phys=1, right=0).
    site.block_data_mut(&BlockCoord(vec![0, 1, 0]))
        .expect("allowed block for flux=1")[0] = 7.5;

    let data_before: Vec<f64> = site
        .block_metas()
        .iter()
        .flat_map(|m| site.block_data(&m.coord).unwrap().iter().copied())
        .collect();

    let mut mps: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![site]);
    canonicalize(&backend, &mut mps, 0);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
    assert_eq!(*mps.site(0).flux(), U1Sector(1));

    let data_after: Vec<f64> = mps
        .site(0)
        .block_metas()
        .iter()
        .flat_map(|m| mps.site(0).block_data(&m.coord).unwrap().iter().copied())
        .collect();
    assert_eq!(data_before, data_after);
}

// --------------------------------------------------------------------------
// Edge case: single-site chain
// --------------------------------------------------------------------------

#[test]
fn canonicalize_bsp_single_site_only_updates_canonical_form() {
    // A single-site chain has no bonds to sweep, so canonicalize is a pure
    // canonical-form update. We still require the site data to be unchanged.
    let backend = NativeBackend::new();
    let site = make_4site_u1_mps().site(0).clone();
    let data_before: Vec<f64> = site
        .block_metas()
        .iter()
        .flat_map(|m| site.block_data(&m.coord).unwrap().iter().copied())
        .collect();

    let mut mps: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![site]);
    canonicalize(&backend, &mut mps, 0);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });

    let data_after: Vec<f64> = mps
        .site(0)
        .block_metas()
        .iter()
        .flat_map(|m| mps.site(0).block_data(&m.coord).unwrap().iter().copied())
        .collect();
    assert_eq!(data_before, data_after);
}
