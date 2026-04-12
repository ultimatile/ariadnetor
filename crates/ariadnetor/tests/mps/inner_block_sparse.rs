//! Inner product and norm tests for block-sparse MPS.

use approx::assert_abs_diff_eq;
use arnet::mps::{
    CanonicalForm, Mps, TensorChain, canonicalize_block_sparse, inner_block_sparse,
    norm_block_sparse,
};
use arnet_tensor::block_sparse::{BlockCoord, BlockSparse, Direction, QNIndex};
use arnet_tensor::sector::U1Sector;

use super::helpers::{bsp_mps_contract_full, make_2site_entangled_u1_mps, make_4site_u1_mps};

// --------------------------------------------------------------------------
// inner_block_sparse
// --------------------------------------------------------------------------

#[test]
fn inner_self_equals_norm_squared() {
    let mps = make_4site_u1_mps();
    let overlap = inner_block_sparse(&mps, &mps);
    let n = norm_block_sparse(&mps);
    assert_abs_diff_eq!(overlap, n * n, epsilon = 1e-10);
}

#[test]
fn inner_self_equals_frobenius_norm_squared() {
    // For a 1D Hilbert space (4-site zero-charge), inner = norm² = Frobenius²
    let mps = make_4site_u1_mps();
    let state = bsp_mps_contract_full(&mps);
    let frob = state.norm();
    let overlap = inner_block_sparse(&mps, &mps);
    assert_abs_diff_eq!(overlap, frob * frob, epsilon = 1e-10);
}

#[test]
fn inner_entangled_fixture() {
    // 2-site: state = 3|01⟩ + 8|10⟩, norm² = 9 + 64 = 73
    let mps = make_2site_entangled_u1_mps();
    let overlap = inner_block_sparse(&mps, &mps);
    assert_abs_diff_eq!(overlap, 73.0, epsilon = 1e-10);
}

#[test]
fn inner_preserved_by_canonicalize() {
    let mps_a = make_2site_entangled_u1_mps();
    let mut mps_b = make_2site_entangled_u1_mps();

    let overlap_before = inner_block_sparse(&mps_a, &mps_b);
    canonicalize_block_sparse(&mut mps_b, 0);
    let overlap_after = inner_block_sparse(&mps_a, &mps_b);

    assert_abs_diff_eq!(overlap_before, overlap_after, epsilon = 1e-10);
}

#[test]
fn inner_single_site() {
    let left = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let phys = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
    let mut site = BlockSparse::<f64, U1Sector>::zeros(vec![left, phys, right], U1Sector(0));
    site.block_data_mut(&BlockCoord(vec![0, 0, 0])).unwrap()[0] = 3.0;

    let mps = Mps::from_storages(vec![site]);
    let overlap = inner_block_sparse(&mps, &mps);
    assert_abs_diff_eq!(overlap, 9.0, epsilon = 1e-12);
}

// --------------------------------------------------------------------------
// norm_block_sparse
// --------------------------------------------------------------------------

#[test]
fn norm_agrees_with_full_contraction() {
    let mps = make_2site_entangled_u1_mps();
    let n = norm_block_sparse(&mps);
    assert_abs_diff_eq!(n, 73.0_f64.sqrt(), epsilon = 1e-10);
}

#[test]
fn norm_left_canonical_returns_one() {
    let mut mps = make_2site_entangled_u1_mps();
    let last = mps.len() - 1;
    canonicalize_block_sparse(&mut mps, last);
    mps.set_canonical_form(CanonicalForm::Left);
    assert_abs_diff_eq!(norm_block_sparse(&mps), 1.0, epsilon = 1e-12);
}

#[test]
fn norm_right_canonical_returns_one() {
    let mut mps = make_2site_entangled_u1_mps();
    canonicalize_block_sparse(&mut mps, 0);
    mps.set_canonical_form(CanonicalForm::Right);
    assert_abs_diff_eq!(norm_block_sparse(&mps), 1.0, epsilon = 1e-12);
}

#[test]
fn norm_mixed_uses_center_tensor() {
    let mut mps = make_2site_entangled_u1_mps();
    let norm_full = norm_block_sparse(&mps);

    canonicalize_block_sparse(&mut mps, 1);
    let norm_mixed = norm_block_sparse(&mps);

    assert_abs_diff_eq!(norm_full, norm_mixed, epsilon = 1e-10);
    let center_norm = mps.storage(1).norm();
    assert_abs_diff_eq!(norm_mixed, center_norm, epsilon = 1e-12);
}

#[test]
fn norm_unknown_uses_full_contraction() {
    let mps = make_4site_u1_mps();
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);

    let n = norm_block_sparse(&mps);
    let state = bsp_mps_contract_full(&mps);
    let frob = state.norm();
    assert_abs_diff_eq!(n, frob, epsilon = 1e-10);
}
