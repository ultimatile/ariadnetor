//! apply_block_sparse tests: apply a BlockSparse MPO to a BlockSparse MPS.

use approx::assert_abs_diff_eq;
use arnet::mps::{
    Mpo, Mps, SvdAbsorb, TensorChain, TruncSvdParams, TruncateParams, apply_block_sparse,
    inner_block_sparse, norm_block_sparse,
};
use arnet_tensor::{BlockCoord, BlockSparse, Direction, QNIndex, U1Sector};

use super::helpers::{make_2site_entangled_u1_mps, make_4site_u1_mps};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a U(1) identity MPO for the given number of sites.
///
/// MPO convention: (Out, In, Out, In) = (w_L, d_ket, d_bra, w_R).
/// Physical charges {0, 1}. Bond dim = 1. Flux = 0 per site.
fn make_identity_u1_mpo(n: usize) -> Mpo<BlockSparse<f64, U1Sector>> {
    let storages = (0..n)
        .map(|_| {
            let left = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
            let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
            let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
            let right = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
            let mut site =
                BlockSparse::<f64, U1Sector>::zeros(vec![left, ket, bra, right], U1Sector(0));
            // Identity on charge-0 subspace
            site.block_data_mut(&BlockCoord(vec![0, 0, 0, 0])).unwrap()[0] = 1.0;
            // Identity on charge-1 subspace
            site.block_data_mut(&BlockCoord(vec![0, 1, 1, 0])).unwrap()[0] = 1.0;
            site
        })
        .collect();
    Mpo::from_storages(storages)
}

// ---------------------------------------------------------------------------
// Identity MPO preserves MPS
// ---------------------------------------------------------------------------

#[test]
fn identity_mpo_preserves_norm_4site() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply_block_sparse(&identity, &mps, None);

    let norm_before = norm_block_sparse(&mps);
    let norm_after = norm_block_sparse(&result);
    assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-10);
}

#[test]
fn identity_mpo_preserves_inner_product() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply_block_sparse(&identity, &mps, None);

    // ⟨ψ|I|ψ⟩ = ⟨ψ|ψ⟩ = ⟨ψ|result⟩ (since I|ψ⟩ = |ψ⟩)
    let inner_psi_psi = inner_block_sparse(&mps, &mps);
    let inner_psi_result = inner_block_sparse(&mps, &result);
    assert_abs_diff_eq!(inner_psi_psi, inner_psi_result, epsilon = 1e-10);
}

#[test]
fn identity_mpo_preserves_entangled_state() {
    let mps = make_2site_entangled_u1_mps();
    let identity = make_identity_u1_mpo(2);

    let result = apply_block_sparse(&identity, &mps, None);

    // ⟨ψ|ψ⟩ = 73 for this fixture
    let inner_before = inner_block_sparse(&mps, &mps);
    let inner_after = inner_block_sparse(&result, &result);
    assert_abs_diff_eq!(inner_before, inner_after, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// Output structure
// ---------------------------------------------------------------------------

#[test]
fn output_is_rank3_mps() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply_block_sparse(&identity, &mps, None);

    assert_eq!(result.len(), mps.len());
    for j in 0..result.len() {
        assert_eq!(result.storage(j).rank(), 3, "site {j} should be rank-3");
        // Verify MPS convention: (Out, Out, In)
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

    let result = apply_block_sparse(&identity, &mps, None);

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

    let result = apply_block_sparse(&identity, &mps, Some(&params));

    // Truncated result should still have finite elements and correct structure
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

    // Norm should be approximately preserved (identity MPO, only truncation error)
    let norm_before = norm_block_sparse(&mps);
    let norm_after = norm_block_sparse(&result);
    assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// Panics
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "MPO and MPS lengths must match")]
fn length_mismatch_panics() {
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(3); // 3 != 4
    apply_block_sparse(&identity, &mps, None);
}

#[test]
#[should_panic(expected = "must have at least one site")]
fn empty_mps_panics() {
    let mps = Mps::<BlockSparse<f64, U1Sector>>::from_storages(vec![]);
    let mpo = Mpo::<BlockSparse<f64, U1Sector>>::from_storages(vec![]);
    apply_block_sparse(&mpo, &mps, None);
}
