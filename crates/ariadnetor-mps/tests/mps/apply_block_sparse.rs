//! apply tests: apply a BlockSparse MPO to a BlockSparse MPS.

use approx::assert_abs_diff_eq;
use ariadnetor_mps::{
    self as mps, ApplyMethod, CanonicalForm, Mpo, Mps, SvdAbsorb, TensorChain, TruncSvdParams,
    TruncateParams, apply, inner,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::test_fixtures::legs;
use ariadnetor_tensor::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, Direction, U1Sector,
};

use super::helpers::{
    assert_block_sparse_close, bsp_mps_contract_full, make_2site_entangled_u1_mps,
    make_3site_u1_mps_multipath_middle, make_4site_u1_mps, make_identity_u1_mpo,
    make_total_n_u1_mpo,
};

// ---------------------------------------------------------------------------
// Identity MPO preserves MPS
// ---------------------------------------------------------------------------

#[test]
fn identity_mpo_preserves_norm_4site() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply(&backend, &identity, &mps, None);

    let norm_before = mps.norm(&backend);
    let norm_after = result.norm(&backend);
    assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-10);
}

#[test]
fn identity_mpo_preserves_inner_product() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply(&backend, &identity, &mps, None);

    let inner_psi_psi = inner(&backend, &mps, &mps);
    let inner_psi_result = inner(&backend, &mps, &result);
    assert_abs_diff_eq!(inner_psi_psi, inner_psi_result, epsilon = 1e-10);
}

#[test]
fn identity_mpo_preserves_entangled_state() {
    let backend = NativeBackend::new();
    let mps = make_2site_entangled_u1_mps();
    let identity = make_identity_u1_mpo(2);

    let result = apply(&backend, &identity, &mps, None);

    let inner_before = inner(&backend, &mps, &mps);
    let inner_after = inner(&backend, &result, &result);
    assert_abs_diff_eq!(inner_before, inner_after, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// Output structure
// ---------------------------------------------------------------------------

#[test]
fn output_is_rank3_mps() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply(&backend, &identity, &mps, None);

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
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let result = apply(&backend, &identity, &mps, None);

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
    let backend = NativeBackend::new();
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

    let result = apply(&backend, &identity, &mps, Some(&params));

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
    let norm_before = mps.norm(&backend);
    let norm_after = result.norm(&backend);
    assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// Panics
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "MPO and MPS lengths must match")]
fn length_mismatch_panics() {
    let backend = NativeBackend::new();
    let mps = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(3);
    apply(&backend, &identity, &mps, None);
}

#[test]
#[should_panic(expected = "must have at least one site")]
fn empty_mps_panics() {
    let backend = NativeBackend::new();
    let mps = Mps::<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>::empty();
    let mpo = Mpo::<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>::empty();
    apply(&backend, &mpo, &mps, None);
}

// ===========================================================================
// Streaming-naive algorithm tests (BlockSparse)
// ===========================================================================

#[test]
fn streaming_naive_identity_preserves_state() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let phi = mps::apply(&backend, &identity, &psi, None);

    let v_before = bsp_mps_contract_full(&psi);
    let v_after = bsp_mps_contract_full(&phi);
    assert_block_sparse_close(&v_before, &v_after, 1e-10);
}

#[test]
fn streaming_naive_canonical_form() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    // No params: the forward QR sweep leaves the chain at center = n - 1.
    let phi_none = mps::apply(&backend, &identity, &psi, None);
    assert_eq!(
        *phi_none.canonical_form(),
        CanonicalForm::Mixed { center: 3 }
    );

    // With params: canonicalize + truncate finishing parks the center at 0.
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(8),
        target_trunc_err: None,
    });
    let phi_some = mps::apply(&backend, &identity, &psi, Some(&params));
    assert_eq!(
        *phi_some.canonical_form(),
        CanonicalForm::Mixed { center: 0 }
    );
}

#[test]
fn streaming_naive_output_structure_and_flux() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let phi = mps::apply(&backend, &identity, &psi, None);

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
    let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(left_c), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(right_c), 1)], Direction::In),
        ]),
        U1Sector(0),
    );
    site.block_data_mut(&BlockCoord(vec![0, phys_c, 0]))
        .unwrap()[0] = 1.0;
    site
}

#[test]
fn apply_bsp_n_on_zero_state() {
    // |0000⟩ has total N = 0. The right-edge charge-0 block (bL=I → apply
    // n_phys = 0) is the only one that fires here, so this anchors the
    // boundary case.
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![
        bsp_basis_site(0, 0, 0),
        bsp_basis_site(0, 0, 0),
        bsp_basis_site(0, 0, 0),
        bsp_basis_site(0, 0, 0),
    ]);
    let n_op = make_total_n_u1_mpo(4);

    let psi_norm_sq = inner(&backend, &psi, &psi);
    let n_psi = apply(&backend, &n_op, &psi, None);
    let exp_n = inner(&backend, &psi, &n_psi);

    assert_abs_diff_eq!(psi_norm_sq, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(exp_n, 0.0, epsilon = 1e-10);
}

#[test]
fn apply_bsp_n_eigenvalue_on_multi_particle_basis_state() {
    // |1010⟩ on 4 sites: total N = 2, two interior MPO sites exercised
    // simultaneously. With 2 particles distributed across 4 sites, the FSM
    // bond traverses I → n → n → n on sites 0, 1, 2, 3 (the I → n transition
    // fires at site 0, then stays at n until the right boundary).
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![
        bsp_basis_site(0, 1, 1),
        bsp_basis_site(1, 0, 1),
        bsp_basis_site(1, 1, 2),
        bsp_basis_site(2, 0, 2),
    ]);
    let n_op = make_total_n_u1_mpo(4);

    let psi_norm_sq = inner(&backend, &psi, &psi);
    let n_psi = apply(&backend, &n_op, &psi, None);
    let exp_n = inner(&backend, &psi, &n_psi);

    assert_abs_diff_eq!(psi_norm_sq, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(exp_n, 2.0, epsilon = 1e-10);
}

#[test]
fn apply_bsp_n_squared_via_composition() {
    // |11⟩ on 2 sites has N|11⟩ = 2|11⟩, so ⟨ψ|N²|ψ⟩ = 4. Re-feeding the
    // apply output back into apply tests that the result is a well-formed
    // MPS the operator can act on again — the algebraic eigenvalue
    // identity acts as the analytical anchor across the composition.
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![bsp_basis_site(0, 1, 1), bsp_basis_site(1, 1, 2)]);
    let n_op = make_total_n_u1_mpo(2);

    let n_psi = apply(&backend, &n_op, &psi, None);
    let nn_psi = apply(&backend, &n_op, &n_psi, None);
    let exp_n_sq = inner(&backend, &psi, &nn_psi);

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
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![
        bsp_basis_site(0, 0, 0),
        bsp_basis_site(0, 1, 1),
        bsp_basis_site(1, 0, 1),
    ]);
    let n_op = make_total_n_u1_mpo(3);

    let psi_norm_sq = inner(&backend, &psi, &psi);
    let n_psi = apply(&backend, &n_op, &psi, None);
    let exp_n = inner(&backend, &psi, &n_psi);

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
    let backend = NativeBackend::new();
    let psi = make_2site_entangled_u1_mps(); // 3|01⟩ + 8|10⟩, both N=1
    let n_op = make_total_n_u1_mpo(2);

    let psi_norm_sq = inner(&backend, &psi, &psi);
    let n_psi = apply(&backend, &n_op, &psi, None);
    let exp_n = inner(&backend, &psi, &n_psi);

    // Total particle number on this state is 1 → ⟨ψ|N|ψ⟩ = ⟨ψ|ψ⟩.
    assert_abs_diff_eq!(exp_n, psi_norm_sq, epsilon = 1e-10);
}

#[test]
fn streaming_naive_truncates_bond_dim() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let phi = mps::apply(&backend, &identity, &psi, Some(&params));

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
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });

    let phi = mps::apply(&backend, &op, &psi, Some(&params));

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
    let backend = NativeBackend::new();
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

    let phi = mps::apply(&backend, &identity, &psi, Some(&params));

    assert_eq!(*phi.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

#[test]
fn streaming_naive_absorb_both_yields_unknown_canonical_form() {
    let backend = NativeBackend::new();
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

    let phi = mps::apply(&backend, &identity, &psi, Some(&params));

    assert_eq!(*phi.canonical_form(), CanonicalForm::Unknown);
}

#[test]
fn streaming_naive_center_nonzero_parks_center_at_request() {
    let backend = NativeBackend::new();
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
        let phi = mps::apply(&backend, &identity, &psi, Some(&params));
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
/// forward sweep.
#[test]
fn streaming_naive_forward_cap_factor_one_keeps_chi_max() {
    use std::num::NonZeroUsize;

    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let method = ApplyMethod::StreamingNaive {
        forward_cap: Some(NonZeroUsize::new(1).unwrap()),
    };

    let phi = mps::apply_with_method(&backend, &op, &psi, Some(&params), method);

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond {d} exceeds chi_max=2 under forward_cap=1");
    }
}

/// Tightening `forward_cap` from `None` to `Some(1)` on a fixture whose
/// middle bond has a sector with dim 3 fed by two distinct (left, phys)
/// pathways must change the contracted output observably. The multi-path
/// structure breaks the forward `(left*phys, right)` vs backward
/// `(left, phys*right)` unfolding symmetry: per-sector SVD truncation
/// chooses different rank-2 subspaces in the two sweep directions, so a
/// `forward_cap = Some(1)` cap that pre-truncates the forward
/// intermediate produces a chi=2 chain genuinely different from the
/// lossless-forward + backward-chi=2 path.
///
/// Fixtures with single-pathway per sector and per-sector phys dim 1
/// (e.g. `make_4site_u1_mps`) collapse the two unfoldings into the same
/// matrix and hide the cap's effect end-to-end, which is why this test
/// needs the dedicated multipath fixture.
///
/// If `forward_cap` were ignored — e.g. `forward_rank_estimate_bsp > cap`
/// always evaluated to `false`, or the SVD branch routed back to QR —
/// both runs would take the QR branch and the contracted outputs would
/// match within roundoff.
#[test]
fn streaming_naive_forward_cap_observably_changes_output() {
    use std::num::NonZeroUsize;

    let backend = NativeBackend::new();
    let psi = make_3site_u1_mps_multipath_middle();
    let op = make_identity_u1_mpo(3);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let capped = ApplyMethod::StreamingNaive {
        forward_cap: Some(NonZeroUsize::new(1).unwrap()),
    };

    let phi_lossless = mps::apply(&backend, &op, &psi, Some(&params));
    let phi_capped = mps::apply_with_method(&backend, &op, &psi, Some(&params), capped);

    // Both paths must still respect the chi_max budget.
    for d in phi_capped.bond_dims() {
        assert!(d <= 2, "bond {d} exceeds chi_max=2 under forward_cap=1");
    }

    let v_lossless = bsp_mps_contract_full(&phi_lossless);
    let v_capped = bsp_mps_contract_full(&phi_capped);
    let same = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_block_sparse_close(&v_lossless, &v_capped, 1e-8);
    }))
    .is_ok();
    assert!(
        !same,
        "forward_cap = Some(1) and forward_cap = None should produce \
         observably different states on the thick-middle fixture, but \
         the contracted outputs matched within 1e-8"
    );
}

// ---------------------------------------------------------------------------
// Zip-up algorithm tests
// ---------------------------------------------------------------------------

/// With no truncation, the zip-up sweep is a lossless refactoring of the
/// exact MPO·MPS product, so its contracted state matches the lossless
/// streaming-naive baseline.
#[test]
fn zipup_lossless_matches_streaming_naive() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);

    let phi_zipup = mps::apply_with_method(&backend, &op, &psi, None, ApplyMethod::ZipUp);
    let phi_baseline = mps::apply(&backend, &op, &psi, None);

    let v_zipup = bsp_mps_contract_full(&phi_zipup);
    let v_baseline = bsp_mps_contract_full(&phi_baseline);
    assert_block_sparse_close(&v_zipup, &v_baseline, 1e-10);
}

/// A `chi_max` cap bounds every bond of the zip-up result. The thick-middle
/// fixture has a sector of dim 3, so `chi_max = 2` genuinely truncates.
#[test]
fn zipup_truncates_bond_dim() {
    let backend = NativeBackend::new();
    let psi = make_3site_u1_mps_multipath_middle();
    let op = make_identity_u1_mpo(3);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });

    let phi = mps::apply_with_method(&backend, &op, &psi, Some(&params), ApplyMethod::ZipUp);

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond {d} exceeds chi_max=2");
    }
}

/// Zip-up ends its forward sweep left-canonical, parking the center at the
/// last site.
#[test]
fn zipup_canonical_form() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);

    let phi = mps::apply_with_method(&backend, &op, &psi, None, ApplyMethod::ZipUp);
    assert_eq!(*phi.canonical_form(), CanonicalForm::Mixed { center: 3 });
}

// ---------------------------------------------------------------------------
// Density-matrix algorithm tests
// ---------------------------------------------------------------------------

/// With no truncation, the density-matrix sweep is a lossless refactoring of
/// the exact MPO·MPS product, so its contracted state matches the lossless
/// streaming-naive baseline. This also pins the block-sparse `ρ = θ R θ†`
/// leg-direction wiring (`trunc_svd(&ρ, 2, …)` must accept the rank-4 `ρ` and
/// the carry `U† θ` must contract into the next site).
#[test]
fn density_matrix_lossless_matches_streaming_naive() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);

    let phi_dm = mps::apply_with_method(&backend, &op, &psi, None, ApplyMethod::DensityMatrix);
    let phi_baseline = mps::apply(&backend, &op, &psi, None);

    let v_dm = bsp_mps_contract_full(&phi_dm);
    let v_baseline = bsp_mps_contract_full(&phi_baseline);
    assert_block_sparse_close(&v_dm, &v_baseline, 1e-10);
}

/// A `chi_max` cap bounds every bond of the density-matrix result. The
/// thick-middle fixture has a sector of dim 3, so `chi_max = 2` genuinely
/// truncates.
#[test]
fn density_matrix_truncates_bond_dim() {
    let backend = NativeBackend::new();
    let psi = make_3site_u1_mps_multipath_middle();
    let op = make_identity_u1_mpo(3);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });

    let phi = mps::apply_with_method(
        &backend,
        &op,
        &psi,
        Some(&params),
        ApplyMethod::DensityMatrix,
    );

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

/// The density-matrix sweep ends its forward pass left-canonical, parking the
/// center at the last site.
#[test]
fn density_matrix_canonical_form() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);

    let phi = mps::apply_with_method(&backend, &op, &psi, None, ApplyMethod::DensityMatrix);
    assert_eq!(*phi.canonical_form(), CanonicalForm::Mixed { center: 3 });
}
