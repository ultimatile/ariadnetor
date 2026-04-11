//! Canonicalize tests for block-sparse MPS.
//!
//! Covers `canonicalize_block_sparse` on a 4-site U(1)-symmetric chain with
//! non-trivial per-sector bond structure. The fixture is designed so that at
//! least one QR/LQ factorization on every site is genuinely non-trivial
//! (multi-element sector blocks), which is essential for catching mutants in
//! the per-sector sweep logic.

use arnet::mps::{CanonicalForm, Mps, TensorChain, canonicalize_block_sparse};
use arnet_tensor::block_sparse::BlockSparse;
use arnet_tensor::sector::U1Sector;

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
    let mut mps = make_4site_u1_mps();
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);

    canonicalize_block_sparse(&mut mps, 2);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
}

// --------------------------------------------------------------------------
// Per-center isometry structure
// --------------------------------------------------------------------------

#[test]
fn canonicalize_bsp_center_0_all_right_isometric() {
    let mut mps = make_4site_u1_mps();
    canonicalize_block_sparse(&mut mps, 0);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
    // All sites past the center must be right-canonical.
    for j in 1..mps.len() {
        assert!(
            is_right_canonical_bsp(mps.storage(j), TOL),
            "site {j} is not right-canonical after canonicalize(center=0)"
        );
    }
}

#[test]
fn canonicalize_bsp_center_last_all_left_isometric() {
    let mut mps = make_4site_u1_mps();
    let last = mps.len() - 1;
    canonicalize_block_sparse(&mut mps, last);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: last });
    // Sites 0..last must be left-canonical.
    for j in 0..last {
        assert!(
            is_left_canonical_bsp(mps.storage(j), TOL),
            "site {j} is not left-canonical after canonicalize(center=last)"
        );
    }
}

#[test]
fn canonicalize_bsp_center_middle_has_mixed_isometry() {
    let mut mps = make_4site_u1_mps();
    canonicalize_block_sparse(&mut mps, 2);

    // 0..2 left-canonical, 3..4 right-canonical; site 2 is the orthogonality center.
    for j in 0..2 {
        assert!(
            is_left_canonical_bsp(mps.storage(j), TOL),
            "site {j} is not left-canonical after canonicalize(center=2)"
        );
    }
    for j in 3..mps.len() {
        assert!(
            is_right_canonical_bsp(mps.storage(j), TOL),
            "site {j} is not right-canonical after canonicalize(center=2)"
        );
    }
}

// --------------------------------------------------------------------------
// State preservation (primary correctness witness)
// --------------------------------------------------------------------------

#[test]
fn canonicalize_bsp_preserves_full_chain_state_center_0() {
    let mps = make_4site_u1_mps();
    let state_before = bsp_mps_contract_full(&mps);

    let mut mps_after = mps.clone();
    canonicalize_block_sparse(&mut mps_after, 0);
    let state_after = bsp_mps_contract_full(&mps_after);

    assert_block_sparse_close(&state_before, &state_after, TOL);
}

#[test]
fn canonicalize_bsp_preserves_full_chain_state_center_middle() {
    let mps = make_4site_u1_mps();
    let state_before = bsp_mps_contract_full(&mps);

    let mut mps_after = mps.clone();
    canonicalize_block_sparse(&mut mps_after, 2);
    let state_after = bsp_mps_contract_full(&mps_after);

    assert_block_sparse_close(&state_before, &state_after, TOL);
}

#[test]
fn canonicalize_bsp_preserves_full_chain_state_center_last() {
    let mps = make_4site_u1_mps();
    let state_before = bsp_mps_contract_full(&mps);

    let mut mps_after = mps.clone();
    let last = mps_after.len() - 1;
    canonicalize_block_sparse(&mut mps_after, last);
    let state_after = bsp_mps_contract_full(&mps_after);

    assert_block_sparse_close(&state_before, &state_after, TOL);
}

// --------------------------------------------------------------------------
// Flux preservation per site
// --------------------------------------------------------------------------

#[test]
fn canonicalize_bsp_preserves_per_site_flux() {
    let mps = make_4site_u1_mps();
    let fluxes_before: Vec<U1Sector> = (0..mps.len()).map(|j| *mps.storage(j).flux()).collect();

    let mut mps_after = mps.clone();
    canonicalize_block_sparse(&mut mps_after, 2);

    for (j, expected) in fluxes_before.iter().enumerate() {
        assert_eq!(
            mps_after.storage(j).flux(),
            expected,
            "site {j} flux changed through canonicalize"
        );
    }
}

// --------------------------------------------------------------------------
// Edge case: single-site chain
// --------------------------------------------------------------------------

#[test]
fn canonicalize_bsp_single_site_only_updates_canonical_form() {
    // A single-site chain has no bonds to sweep, so canonicalize is a pure
    // canonical-form update. We still require the site data to be unchanged.
    let site = make_4site_u1_mps().storage(0).clone();
    let data_before: Vec<f64> = site
        .block_metas()
        .iter()
        .flat_map(|m| site.block_data(&m.coord).unwrap().iter().copied())
        .collect();

    let mut mps: Mps<BlockSparse<f64, U1Sector>> = Mps::from_storages(vec![site]);
    canonicalize_block_sparse(&mut mps, 0);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });

    let data_after: Vec<f64> = mps
        .storage(0)
        .block_metas()
        .iter()
        .flat_map(|m| mps.storage(0).block_data(&m.coord).unwrap().iter().copied())
        .collect();
    assert_eq!(data_before, data_after);
}
