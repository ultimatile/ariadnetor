//! Truncate tests for block-sparse MPS.
//!
//! Covers `truncate` on a 4-site U(1)-symmetric chain.
//! Mirrors the Dense truncate test structure, adapted for BlockSparse
//! invariants (per-sector isometry, flux preservation, block-level
//! state comparison).

use arnet::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, Direction, QNIndex,
    U1Sector,
};
use arnet_mps::{
    CanonicalForm, Mps, SvdAbsorb, TensorChain, TruncSvdParams, TruncateParams, canonicalize,
    truncate,
};

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
    canonicalize(&mut mps, 2);

    let state_before = bsp_mps_contract_full(&mps);
    let bond_dims_before = mps.bond_dims();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(100),
        target_trunc_err: None,
    });
    let result = truncate(&mut mps, &params);

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
    canonicalize(&mut mps, 0);
    let state_before = bsp_mps_contract_full(&mps);
    let norm_before = state_before.norm();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    truncate(&mut mps, &params);
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
    canonicalize(&mut mps, 0);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = truncate(&mut mps, &params);

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
    canonicalize(&mut mps, 2);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    truncate(&mut mps, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 2 });
    // Sites 0 and 1 left-canonical, site 2 is center, site 3 right-canonical
    for j in 0..2 {
        assert!(
            is_left_canonical_bsp(mps.site(j), TOL),
            "site {j} not left-canonical after truncate (SvdAbsorb::Right)"
        );
    }
    for j in 3..mps.len() {
        assert!(
            is_right_canonical_bsp(mps.site(j), TOL),
            "site {j} not right-canonical after truncate (SvdAbsorb::Right)"
        );
    }
}

#[test]
fn truncate_bsp_absorb_left_isometry() {
    let mut mps = make_4site_u1_mps();
    canonicalize(&mut mps, 1);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Left,
        center: None,
    };
    let result = truncate(&mut mps, &params);

    assert!(result.error >= 0.0);
    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 1 });

    // Site 0 left-canonical, sites 2..4 right-canonical
    assert!(
        is_left_canonical_bsp(mps.site(0), TOL),
        "site 0 not left-canonical with SvdAbsorb::Left"
    );
    for j in 2..mps.len() {
        assert!(
            is_right_canonical_bsp(mps.site(j), TOL),
            "site {j} not right-canonical with SvdAbsorb::Left"
        );
    }
}

#[test]
fn truncate_bsp_absorb_both_sets_unknown() {
    let mut mps = make_4site_u1_mps();
    canonicalize(&mut mps, 1);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Both,
        center: None,
    };
    let result = truncate(&mut mps, &params);

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
    use arnet::{BlockCoord, Direction, QNIndex};

    let left = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let phys = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
    let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(vec![left, phys, right], U1Sector(0));
    site.block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .expect("allowed block")[0] = 3.0;

    let mut mps: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![site]);
    canonicalize(&mut mps, 0);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = truncate(&mut mps, &params);

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
    let result = truncate(&mut mps, &params);

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
    canonicalize(&mut mps, 0);
    let norm_before = bsp_mps_contract_full(&mps).norm();

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = truncate(&mut mps, &params);

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

/// Verify reported error² matches ||before||² - ||after||² (Pythagorean identity).
///
/// Catches: `total_err_sq + step` → `- step` or `* step`,
///          `err * err` → `err + err` in right/left_trunc_step_block_sparse.
#[test]
fn truncate_bsp_error_matches_reconstruction_error() {
    let mut mps = make_4site_u1_mps();
    canonicalize(&mut mps, 2);
    let state_before = bsp_mps_contract_full(&mps);
    let norm_before = state_before.norm();
    let norm_sq_before = norm_before * norm_before;

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = truncate(&mut mps, &params);
    let state_after = bsp_mps_contract_full(&mps);
    let norm_after = state_after.norm();
    let norm_sq_after = norm_after * norm_after;

    let expected_err_sq = norm_sq_before - norm_sq_after;
    let reported_err_sq = result.error * result.error;
    assert!(
        (reported_err_sq - expected_err_sq).abs() < 1e-6 * norm_sq_before,
        "reported²={reported_err_sq}, expected²={expected_err_sq}, \
         diff={}",
        (reported_err_sq - expected_err_sq).abs(),
    );
}

// --------------------------------------------------------------------------
// Per-step error accumulator pinning
// --------------------------------------------------------------------------

/// 3-site U(1) MPS in the total-charge-1 sector with non-trivial Schmidt
/// structure on every bond. Block coefficients chosen so that
/// `canonicalize(center=1)` followed by `chi_max=1` truncation produces a
/// nonzero err contribution at the right sweep (j=1), the left sweep (j=1),
/// and a zero contribution at the final right sweep — making 5 of the 6 sweep
/// positions distinguish the original arithmetic from each missed mutant on
/// `truncate.rs:349/354/424/482`.
fn make_3site_u1_truncate_fixture() -> Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    let left0 = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let phys0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let mut site0 =
        BlockSparseTensor::<f64, U1Sector>::zeros(vec![left0, phys0, right0], U1Sector(0));
    site0
        .block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .expect("site0 block [0,0,0]")[0] = 1.0;
    site0
        .block_data_mut(&BlockCoord(vec![0, 1, 1]))
        .expect("site0 block [0,1,1]")[0] = 2.0;

    let left1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let phys1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let mut site1 =
        BlockSparseTensor::<f64, U1Sector>::zeros(vec![left1, phys1, right1], U1Sector(0));
    site1
        .block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .expect("site1 block [0,0,0]")[0] = 3.0;
    site1
        .block_data_mut(&BlockCoord(vec![0, 1, 1]))
        .expect("site1 block [0,1,1]")[0] = 4.0;
    site1
        .block_data_mut(&BlockCoord(vec![1, 0, 1]))
        .expect("site1 block [1,0,1]")[0] = 5.0;

    let left2 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let phys2 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right2 = QNIndex::new(vec![(U1Sector(1), 1)], Direction::In);
    let mut site2 =
        BlockSparseTensor::<f64, U1Sector>::zeros(vec![left2, phys2, right2], U1Sector(0));
    site2
        .block_data_mut(&BlockCoord(vec![0, 1, 0]))
        .expect("site2 block [0,1,0]")[0] = 6.0;
    site2
        .block_data_mut(&BlockCoord(vec![1, 0, 0]))
        .expect("site2 block [1,0,0]")[0] = 7.0;

    Mps::from_sites(vec![site0, site1, site2])
}

/// Pin the reported truncation error against the Pythagorean reconstruction
/// identity on a fixture with genuine multi-Schmidt structure per bond.
///
/// `truncate_bsp` accumulates squared step errors across three sweeps. Each
/// step in `right_trunc_step_block_sparse` / `left_trunc_step_block_sparse`
/// returns `err * err` (squared), and `truncate_bsp` adds them at three
/// `+`-loops (right, left, final-right). For the identity
/// `result.error² == ||before||² - ||after||²` to hold under the original
/// arithmetic, every `+` and every `*` must be present as written:
///
/// * `+` → `-` (line 349, left sweep): flips the sign of left contributions,
///   so the accumulated total drops below the reconstruction value (or goes
///   negative → NaN).
/// * `+` → `*` (line 349 or 354): zeros the entire accumulator the first
///   time a step value is zero, giving `result.error = 0` while
///   `||before||² > ||after||²`.
/// * `* → +` in `right_trunc_step_block_sparse` (line 424) /
///   `left_trunc_step_block_sparse` (line 482): each step returns
///   `err + err = 2err` instead of `err²`. The accumulator becomes a sum of
///   `2*err_step` instead of `err_step²`, which matches the reconstruction
///   identity only at the trivial `err_step = 0` case the prior 4-site
///   fixture happened to hit.
///
/// `center=1` exercises the right sweep range `1..2 = {1}` AND the final
/// right sweep range `0..1 = {0}`, so `truncate.rs:354 + → *` is observable
/// (its prior accumulator is nonzero from the right + left sweeps, and the
/// final-right step multiplies by 0 instead of adding 0).
#[test]
fn truncate_bsp_error_pins_step_arithmetic_3site() {
    let mut mps = make_3site_u1_truncate_fixture();
    canonicalize(&mut mps, 1);
    let state_before = bsp_mps_contract_full(&mps);
    let norm_before = state_before.norm();
    let norm_sq_before = norm_before * norm_before;

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let result = truncate(&mut mps, &params);
    let state_after = bsp_mps_contract_full(&mps);
    let norm_after = state_after.norm();
    let norm_sq_after = norm_after * norm_after;

    let expected_err_sq = norm_sq_before - norm_sq_after;
    let reported_err_sq = result.error * result.error;

    assert!(
        expected_err_sq > 1e-3 * norm_sq_before,
        "fixture must have genuine truncation error, got expected²={expected_err_sq} \
         (norm_sq_before={norm_sq_before}) — Schmidt structure too trivial",
    );
    // Tolerance 1e-8 leaves headroom over BLAS / SVD rounding noise (typically
    // O(1e-13) in f64) while still catching every targeted mutation. The
    // observable mutation classes shift the accumulator by at least the size
    // of one step's `err²` (≥ 1e-3 of `norm_sq_before` on this fixture) or
    // zero it entirely (`+ → *` paired with a zero step), so any tolerance
    // well below 1e-3 of `norm_sq_before` suffices.
    assert!(
        (reported_err_sq - expected_err_sq).abs() < 1e-8 * norm_sq_before,
        "reported²={reported_err_sq}, expected²={expected_err_sq}, \
         diff={}, tolerance={}",
        (reported_err_sq - expected_err_sq).abs(),
        1e-8 * norm_sq_before,
    );
}

// --------------------------------------------------------------------------
// CanonicalForm::Left / Right match arms
// --------------------------------------------------------------------------

/// Catches: delete match arm CanonicalForm::Left, `n - 1` → `n + 1`/`n / 1`.
#[test]
fn truncate_bsp_left_form_center_is_last_site() {
    let mut mps = make_4site_u1_mps();
    let n = mps.len();
    canonicalize(&mut mps, n - 1);
    mps.set_canonical_form(CanonicalForm::Left);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    truncate(&mut mps, &params);

    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Mixed { center: n - 1 }
    );
    for j in 0..n - 1 {
        assert!(
            is_left_canonical_bsp(mps.site(j), TOL),
            "site {j} not left-canonical after truncate from Left form"
        );
    }
}

/// Catches: delete match arm CanonicalForm::Right.
/// Right form uses center=0 regardless of params.center.
#[test]
fn truncate_bsp_right_form_ignores_center_param() {
    let mut mps = make_4site_u1_mps();
    canonicalize(&mut mps, 0);
    mps.set_canonical_form(CanonicalForm::Right);

    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Right,
        center: Some(2),
    };
    truncate(&mut mps, &params);

    assert_eq!(*mps.canonical_form(), CanonicalForm::Mixed { center: 0 });
}
