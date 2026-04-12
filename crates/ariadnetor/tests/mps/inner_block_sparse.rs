//! Inner product and norm tests for block-sparse MPS.

use approx::assert_abs_diff_eq;
use arnet::mps::{
    CanonicalForm, Mpo, Mps, TensorChain, braket_block_sparse, canonicalize_block_sparse,
    inner_block_sparse, norm_block_sparse,
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

// --------------------------------------------------------------------------
// braket_block_sparse
// --------------------------------------------------------------------------

/// Build a U(1) identity MPO for the given number of sites.
///
/// MPO convention: (Out, In, Out, In) = (χ_L, d_ket, d_bra, χ_R).
/// Physical charges {0, 1}. Flux = 0 per site.
/// Allowed blocks: (0,0,0,0) and (0,1,1,0), each with data [1.0].
fn make_identity_u1_mpo(n: usize) -> Mpo<BlockSparse<f64, U1Sector>> {
    let storages = (0..n)
        .map(|_| {
            let left = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
            let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
            let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
            let right = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
            let mut site =
                BlockSparse::<f64, U1Sector>::zeros(vec![left, ket, bra, right], U1Sector(0));
            // Block (0,0,0,0): identity on charge-0 subspace
            site.block_data_mut(&BlockCoord(vec![0, 0, 0, 0])).unwrap()[0] = 1.0;
            // Block (0,1,1,0): identity on charge-1 subspace
            site.block_data_mut(&BlockCoord(vec![0, 1, 1, 0])).unwrap()[0] = 1.0;
            site
        })
        .collect();
    Mpo::from_storages(storages)
}

#[test]
fn braket_identity_equals_inner_4site() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let inner_val = inner_block_sparse(&mps, &mps);
    let braket_val = braket_block_sparse(&mps, &identity, &mps);

    assert_abs_diff_eq!(inner_val, braket_val, epsilon = 1e-10);
}

#[test]
fn braket_identity_equals_inner_entangled() {
    let mps = make_2site_entangled_u1_mps();
    let identity = make_identity_u1_mpo(2);

    let inner_val = inner_block_sparse(&mps, &mps);
    let braket_val = braket_block_sparse(&mps, &identity, &mps);

    assert_abs_diff_eq!(inner_val, braket_val, epsilon = 1e-10);
}

#[test]
fn braket_identity_after_canonicalize() {
    let mut mps = make_2site_entangled_u1_mps();
    canonicalize_block_sparse(&mut mps, 0);
    let identity = make_identity_u1_mpo(2);

    let braket_val = braket_block_sparse(&mps, &identity, &mps);
    let inner_val = inner_block_sparse(&mps, &mps);
    // ⟨ψ|I|ψ⟩ = ⟨ψ|ψ⟩ regardless of canonical form
    assert_abs_diff_eq!(braket_val, inner_val, epsilon = 1e-10);
}

#[test]
fn braket_diagonal_single_site() {
    // Sz-like operator: diag(0.5, -0.5) on a single site.
    // MPO: (Out:{0:1}, In:{0:1,1:1}, Out:{0:1,1:1}, In:{0:1}), flux=0
    let left = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
    let mut sz_site = BlockSparse::<f64, U1Sector>::zeros(vec![left, ket, bra, right], U1Sector(0));
    sz_site
        .block_data_mut(&BlockCoord(vec![0, 0, 0, 0]))
        .unwrap()[0] = 0.5;
    sz_site
        .block_data_mut(&BlockCoord(vec![0, 1, 1, 0]))
        .unwrap()[0] = -0.5;
    let sz_mpo: Mpo<BlockSparse<f64, U1Sector>> = Mpo::from_storages(vec![sz_site]);

    // |0⟩ state: charge-0 physical only
    let s_left = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let s_phys = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let s_right = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
    let mut up_site =
        BlockSparse::<f64, U1Sector>::zeros(vec![s_left, s_phys, s_right], U1Sector(0));
    up_site.block_data_mut(&BlockCoord(vec![0, 0, 0])).unwrap()[0] = 1.0;
    let up: Mps<BlockSparse<f64, U1Sector>> = Mps::from_storages(vec![up_site]);

    // ⟨0|Sz|0⟩ = 0.5
    assert_abs_diff_eq!(braket_block_sparse(&up, &sz_mpo, &up), 0.5, epsilon = 1e-12);
}
