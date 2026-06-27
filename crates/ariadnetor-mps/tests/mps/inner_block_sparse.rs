//! Inner product and norm tests for block-sparse MPS.

use approx::assert_abs_diff_eq;
use arnet_mps::{CanonicalForm, Mpo, Mps, TensorChain, braket, canonicalize, inner, norm};
use arnet_native::NativeBackend;
use arnet_tensor::U1Sector;
use arnet_tensor::test_fixtures::legs;
use arnet_tensor::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, Direction,
};

use super::helpers::{
    bsp_mps_contract_full, make_2site_entangled_u1_mps, make_4site_u1_mps, make_identity_u1_mpo,
};

// --------------------------------------------------------------------------
// inner
// --------------------------------------------------------------------------

#[test]
fn inner_self_equals_norm_squared() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    let overlap = inner(&backend, &mps, &mps);
    let n = norm(&backend, &mps);
    assert_abs_diff_eq!(overlap, n * n, epsilon = 1e-10);
}

#[test]
fn inner_self_equals_frobenius_norm_squared() {
    let backend = NativeBackend::new();
    // For a 1D Hilbert space (4-site zero-charge), inner = norm² = Frobenius²
    let mps = make_4site_u1_mps();
    let state = bsp_mps_contract_full(&mps);
    let frob = state.norm();
    let overlap = inner(&backend, &mps, &mps);
    assert_abs_diff_eq!(overlap, frob * frob, epsilon = 1e-10);
}

#[test]
fn inner_entangled_fixture() {
    let backend = NativeBackend::new();
    // 2-site: state = 3|01⟩ + 8|10⟩, norm² = 9 + 64 = 73
    let mps = make_2site_entangled_u1_mps();
    let overlap = inner(&backend, &mps, &mps);
    assert_abs_diff_eq!(overlap, 73.0, epsilon = 1e-10);
}

#[test]
fn inner_preserved_by_canonicalize() {
    let backend = NativeBackend::new();
    let mps_a = make_2site_entangled_u1_mps();
    let mut mps_b = make_2site_entangled_u1_mps();

    let overlap_before = inner(&backend, &mps_a, &mps_b);
    canonicalize(&backend, &mut mps_b, 0);
    let overlap_after = inner(&backend, &mps_a, &mps_b);

    assert_abs_diff_eq!(overlap_before, overlap_after, epsilon = 1e-10);
}

#[test]
fn inner_single_site() {
    let backend = NativeBackend::new();
    let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 1)], Direction::In),
        ]),
        U1Sector(0),
    );
    site.block_data_mut(&BlockCoord(vec![0, 0, 0])).unwrap()[0] = 3.0;

    let mps = Mps::from_sites(vec![site]);
    let overlap = inner(&backend, &mps, &mps);
    assert_abs_diff_eq!(overlap, 9.0, epsilon = 1e-12);
}

// --------------------------------------------------------------------------
// norm
// --------------------------------------------------------------------------

#[test]
fn norm_agrees_with_full_contraction() {
    let backend = NativeBackend::new();
    let mps = make_2site_entangled_u1_mps();
    let n = norm(&backend, &mps);
    assert_abs_diff_eq!(n, 73.0_f64.sqrt(), epsilon = 1e-10);
}

#[test]
fn norm_left_canonical_returns_one() {
    let backend = NativeBackend::new();
    let mut mps = make_2site_entangled_u1_mps();
    let last = mps.len() - 1;
    canonicalize(&backend, &mut mps, last);
    mps.set_canonical_form(CanonicalForm::Left);
    assert_abs_diff_eq!(norm(&backend, &mps), 1.0, epsilon = 1e-12);
}

#[test]
fn norm_right_canonical_returns_one() {
    let backend = NativeBackend::new();
    let mut mps = make_2site_entangled_u1_mps();
    canonicalize(&backend, &mut mps, 0);
    mps.set_canonical_form(CanonicalForm::Right);
    assert_abs_diff_eq!(norm(&backend, &mps), 1.0, epsilon = 1e-12);
}

#[test]
fn norm_mixed_uses_center_tensor() {
    let backend = NativeBackend::new();
    let mut mps = make_2site_entangled_u1_mps();
    let norm_full = norm(&backend, &mps);

    canonicalize(&backend, &mut mps, 1);
    let norm_mixed = norm(&backend, &mps);

    assert_abs_diff_eq!(norm_full, norm_mixed, epsilon = 1e-10);
    let center_norm = mps.site(1).norm();
    assert_abs_diff_eq!(norm_mixed, center_norm, epsilon = 1e-12);
}

#[test]
fn norm_unknown_uses_full_contraction() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    assert_eq!(*mps.canonical_form(), CanonicalForm::Unknown);

    let n = norm(&backend, &mps);
    let state = bsp_mps_contract_full(&mps);
    let frob = state.norm();
    assert_abs_diff_eq!(n, frob, epsilon = 1e-10);
}

// --------------------------------------------------------------------------
// braket
// --------------------------------------------------------------------------

#[test]
fn braket_identity_equals_inner_4site() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let inner_val = inner(&backend, &mps, &mps);
    let braket_val = braket(&backend, &mps, &identity, &mps);

    assert_abs_diff_eq!(inner_val, braket_val, epsilon = 1e-10);
}

#[test]
fn braket_identity_equals_inner_entangled() {
    let backend = NativeBackend::new();
    let mps = make_2site_entangled_u1_mps();
    let identity = make_identity_u1_mpo(2);

    let inner_val = inner(&backend, &mps, &mps);
    let braket_val = braket(&backend, &mps, &identity, &mps);

    assert_abs_diff_eq!(inner_val, braket_val, epsilon = 1e-10);
}

#[test]
fn braket_identity_after_canonicalize() {
    let backend = NativeBackend::new();
    let mut mps = make_2site_entangled_u1_mps();
    canonicalize(&backend, &mut mps, 0);
    let identity = make_identity_u1_mpo(2);

    let braket_val = braket(&backend, &mps, &identity, &mps);
    let inner_val = inner(&backend, &mps, &mps);
    // ⟨ψ|I|ψ⟩ = ⟨ψ|ψ⟩ regardless of canonical form
    assert_abs_diff_eq!(braket_val, inner_val, epsilon = 1e-10);
}

#[test]
fn braket_diagonal_single_site() {
    let backend = NativeBackend::new();
    // Sz-like operator: diag(0.5, -0.5) on a single site.
    // MPO: (Out:{0:1}, In:{0:1,1:1}, Out:{0:1,1:1}, In:{0:1}), flux=0
    let mut sz_site = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 1)], Direction::In),
        ]),
        U1Sector(0),
    );
    sz_site
        .block_data_mut(&BlockCoord(vec![0, 0, 0, 0]))
        .unwrap()[0] = 0.5;
    sz_site
        .block_data_mut(&BlockCoord(vec![0, 1, 1, 0]))
        .unwrap()[0] = -0.5;
    let sz_mpo: Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mpo::from_sites(vec![sz_site]);

    // |0⟩ state: charge-0 physical only
    let mut up_site = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 1)], Direction::In),
        ]),
        U1Sector(0),
    );
    up_site.block_data_mut(&BlockCoord(vec![0, 0, 0])).unwrap()[0] = 1.0;
    let up: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![up_site]);

    // ⟨0|Sz|0⟩ = 0.5
    assert_abs_diff_eq!(braket(&backend, &up, &sz_mpo, &up), 0.5, epsilon = 1e-12);
}
