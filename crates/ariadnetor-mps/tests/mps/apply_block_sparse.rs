//! apply tests: apply a BlockSparse MPO to a BlockSparse MPS.

use approx::assert_abs_diff_eq;
use arnet::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, Direction, NativeBackend,
    QNIndex, U1Sector,
};
use arnet_mps::{
    self as mps, ApplyMethod, CanonicalForm, Mpo, Mps, SvdAbsorb, TensorChain, TruncSvdParams,
    TruncateParams, apply, inner, norm,
};

use super::helpers::{
    assert_block_sparse_close, bsp_mps_contract_full, make_2site_entangled_u1_mps,
    make_4site_u1_mps, make_identity_u1_mpo, make_total_n_u1_mpo,
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
        assert_eq!(result.site(j).rank(), 3, "site {j} should be rank-3");
        let indices = result.site(j).indices();
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
            result.site(j).flux(),
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
        assert_eq!(result.site(j).rank(), 3);
        for meta in result.site(j).block_metas() {
            let data = result.site(j).block_data(&meta.coord).unwrap();
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
    let mps = Mps::<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>, NativeBackend>::empty(
        NativeBackend::shared(),
    );
    let mpo = Mpo::<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>, NativeBackend>::empty(
        NativeBackend::shared(),
    );
    apply(&mpo, &mps, None);
}

// ===========================================================================
// Streaming-naive algorithm tests (BlockSparse)
// ===========================================================================

#[test]
fn streaming_naive_identity_preserves_state() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let phi = mps::apply(&identity, &psi, None);

    let v_before = bsp_mps_contract_full(&psi);
    let v_after = bsp_mps_contract_full(&phi);
    assert_block_sparse_close(&v_before, &v_after, 1e-10);
}

#[test]
fn streaming_naive_canonical_form() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    // No params: the forward QR sweep leaves the chain at center = n - 1.
    let phi_none = mps::apply(&identity, &psi, None);
    assert_eq!(
        *phi_none.canonical_form(),
        CanonicalForm::Mixed { center: 3 }
    );

    // With params: canonicalize + truncate finishing parks the center at 0.
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(8),
        target_trunc_err: None,
    });
    let phi_some = mps::apply(&identity, &psi, Some(&params));
    assert_eq!(
        *phi_some.canonical_form(),
        CanonicalForm::Mixed { center: 0 }
    );
}

#[test]
fn streaming_naive_output_structure_and_flux() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let phi = mps::apply(&identity, &psi, None);

    assert_eq!(phi.len(), psi.len());
    for j in 0..phi.len() {
        let site = phi.site(j);
        assert_eq!(site.rank(), 3, "site {j} should be rank-3");
        let indices = site.indices();
        assert_eq!(indices[0].direction(), Direction::Out, "site {j} left bond");
        assert_eq!(indices[1].direction(), Direction::Out, "site {j} physical");
        assert_eq!(indices[2].direction(), Direction::In, "site {j} right bond");
        assert_eq!(site.flux(), &U1Sector(0), "site {j} flux");
    }
}

/// Single-basis-state U(1) MPS site with bond dim 1 and the requested
/// integer charges on each leg. Used to construct definite-particle-number
/// product states for MPO correctness anchors.
fn bsp_basis_site(left_c: i32, phys_c: usize, right_c: i32) -> BlockSparseTensor<f64, U1Sector> {
    assert!(phys_c <= 1, "physical dim assumed to be 2 (charges 0, 1)");
    let left = QNIndex::new(vec![(U1Sector(left_c), 1)], Direction::Out);
    let phys = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right = QNIndex::new(vec![(U1Sector(right_c), 1)], Direction::In);
    let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(vec![left, phys, right], U1Sector(0));
    site.block_data_mut(&BlockCoord(vec![0, phys_c, 0]))
        .unwrap()[0] = 1.0;
    site
}

#[test]
fn apply_bsp_n_on_zero_state() {
    // |0000⟩ has total N = 0. The right-edge charge-0 block (bL=I → apply
    // n_phys = 0) is the only one that fires here, so this anchors the
    // boundary case.
    let psi = Mps::from_sites(vec![
        bsp_basis_site(0, 0, 0),
        bsp_basis_site(0, 0, 0),
        bsp_basis_site(0, 0, 0),
        bsp_basis_site(0, 0, 0),
    ]);
    let n_op = make_total_n_u1_mpo(4);

    let psi_norm_sq = inner(&psi, &psi);
    let n_psi = apply(&n_op, &psi, None);
    let exp_n = inner(&psi, &n_psi);

    assert_abs_diff_eq!(psi_norm_sq, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(exp_n, 0.0, epsilon = 1e-10);
}

#[test]
fn apply_bsp_n_eigenvalue_on_multi_particle_basis_state() {
    // |1010⟩ on 4 sites: total N = 2, two interior MPO sites exercised
    // simultaneously. With 2 particles distributed across 4 sites, the FSM
    // bond traverses I → n → n → n on sites 0, 1, 2, 3 (the I → n transition
    // fires at site 0, then stays at n until the right boundary).
    let psi = Mps::from_sites(vec![
        bsp_basis_site(0, 1, 1),
        bsp_basis_site(1, 0, 1),
        bsp_basis_site(1, 1, 2),
        bsp_basis_site(2, 0, 2),
    ]);
    let n_op = make_total_n_u1_mpo(4);

    let psi_norm_sq = inner(&psi, &psi);
    let n_psi = apply(&n_op, &psi, None);
    let exp_n = inner(&psi, &n_psi);

    assert_abs_diff_eq!(psi_norm_sq, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(exp_n, 2.0, epsilon = 1e-10);
}

#[test]
fn apply_bsp_n_squared_via_composition() {
    // |11⟩ on 2 sites has N|11⟩ = 2|11⟩, so ⟨ψ|N²|ψ⟩ = 4. Re-feeding the
    // apply output back into apply tests that the result is a well-formed
    // MPS the operator can act on again — the algebraic eigenvalue
    // identity acts as the analytical anchor across the composition.
    let psi = Mps::from_sites(vec![bsp_basis_site(0, 1, 1), bsp_basis_site(1, 1, 2)]);
    let n_op = make_total_n_u1_mpo(2);

    let n_psi = apply(&n_op, &psi, None);
    let nn_psi = apply(&n_op, &n_psi, None);
    let exp_n_sq = inner(&psi, &nn_psi);

    assert_abs_diff_eq!(exp_n_sq, 4.0, epsilon = 1e-10);
}

#[test]
fn total_n_mpo_acts_as_total_particle_number_3site_interior() {
    // Correctness anchor that exercises an *interior* MPO site (n >= 3).
    // The 2-site case is purely boundary and would not catch a wrong bond
    // layout in the interior block, where the (2, 1, 1, 2) shape's two
    // non-trivial axes interact non-trivially under RowMajor vs ColumnMajor.
    //
    // |010⟩: single particle at site 1, total N = 1, norm² = 1.
    let psi = Mps::from_sites(vec![
        bsp_basis_site(0, 0, 0),
        bsp_basis_site(0, 1, 1),
        bsp_basis_site(1, 0, 1),
    ]);
    let n_op = make_total_n_u1_mpo(3);

    let psi_norm_sq = inner(&psi, &psi);
    let n_psi = apply(&n_op, &psi, None);
    let exp_n = inner(&psi, &n_psi);

    assert_abs_diff_eq!(psi_norm_sq, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(exp_n, 1.0, epsilon = 1e-10);
}

#[test]
fn total_n_mpo_acts_as_total_particle_number() {
    // Correctness check on the make_total_n_u1_mpo fixture itself: apply it
    // to a state lying entirely in the total-N=1 subspace and verify that
    // ⟨ψ|N|ψ⟩ = 1 · ⟨ψ|ψ⟩. Independent analytical anchor for the MPO data
    // layout; other tests that compute observables on the apply output cannot
    // detect a wrong operator if the same operator drives both branches of
    // the comparison.
    let psi = make_2site_entangled_u1_mps(); // 3|01⟩ + 8|10⟩, both N=1
    let n_op = make_total_n_u1_mpo(2);

    let psi_norm_sq = inner(&psi, &psi);
    let n_psi = apply(&n_op, &psi, None);
    let exp_n = inner(&psi, &n_psi);

    // Total particle number on this state is 1 → ⟨ψ|N|ψ⟩ = ⟨ψ|ψ⟩.
    assert_abs_diff_eq!(exp_n, psi_norm_sq, epsilon = 1e-10);
}

#[test]
fn streaming_naive_truncates_bond_dim() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let phi = mps::apply(&identity, &psi, Some(&params));

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

/// Anchor: with `chi_max = 1`, the chi=1 approximation pins the dominant
/// Schmidt direction and the final canonical form sits at the default
/// center. Mutating the forward sweep boundary or the post-sweep canonical
/// form tag perturbs this fixture.
#[test]
fn streaming_naive_truncated_chi1_baseline() {
    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });

    let phi = mps::apply(&op, &psi, Some(&params));

    for d in phi.bond_dims() {
        assert!(d <= 1, "bond {d} exceeds chi_max=1");
    }
    assert_eq!(*phi.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

// ===========================================================================
// SvdAbsorb::{Left, Both} and arbitrary `params.center` support (BlockSparse)
// ===========================================================================

#[test]
fn streaming_naive_absorb_left_yields_mixed_center() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);
    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(8),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Left,
        center: None,
    };

    let phi = mps::apply(&identity, &psi, Some(&params));

    assert_eq!(*phi.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn streaming_naive_absorb_both_yields_unknown_canonical_form() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);
    let params = TruncateParams {
        svd: TruncSvdParams {
            chi_max: Some(8),
            target_trunc_err: None,
        },
        absorb: SvdAbsorb::Both,
        center: None,
    };

    let phi = mps::apply(&identity, &psi, Some(&params));

    assert_eq!(*phi.canonical_form(), CanonicalForm::Unknown);
}

#[test]
fn streaming_naive_center_nonzero_parks_center_at_request() {
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);
    let svd = TruncSvdParams {
        chi_max: Some(8),
        target_trunc_err: None,
    };

    for c in [1usize, 2usize, 3usize] {
        let params = TruncateParams {
            svd: svd.clone(),
            absorb: SvdAbsorb::Right,
            center: Some(c),
        };
        let phi = mps::apply(&identity, &psi, Some(&params));
        assert_eq!(
            *phi.canonical_form(),
            CanonicalForm::Mixed { center: c },
            "params.center = Some({c}) should park the center at site {c}"
        );
    }
}

// ===========================================================================
// forward_cap exposure (BlockSparse)
// ===========================================================================

/// Tight `forward_cap` (factor 1) produces a chain bounded by `chi_max`
/// end-to-end. Anchors the cap parameter's plumbing through the BSP
/// forward sweep. The observable-difference property — that
/// `forward_cap = Some(_)` and `forward_cap = None` produce numerically
/// distinct outputs — is pinned by the dense companion test
/// `test_apply_streaming_naive_forward_cap_observably_changes_output`;
/// the BSP fixtures available here are U(1)-symmetric in a way that
/// makes per-sector forward and backward truncations coincide, so the
/// observable-difference check does not fire even though
/// `apply_streaming_naive_bsp` shares its forward `match` /
/// `forward_rank_estimate_bsp > cap` selection with the dense path.
#[test]
fn streaming_naive_forward_cap_factor_one_keeps_chi_max() {
    use std::num::NonZeroUsize;

    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let method = ApplyMethod::StreamingNaive {
        forward_cap: Some(NonZeroUsize::new(1).unwrap()),
    };

    let phi = mps::apply_with_method(&op, &psi, Some(&params), method);

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond {d} exceeds chi_max=2 under forward_cap=1");
    }
}
