//! Tests for `ApplyMethod::SuccessiveRandomized` (SRC).
//!
//! The method is randomized, so the assertions follow two regimes:
//! - at (or clamped to) the exactly representable rank, the result is exact
//!   with probability one, so agreement with lossless streaming naive is
//!   checked at deterministic tolerances;
//! - in adaptive mode the cutoff is a per-site stopping criterion, so the
//!   realized relative error is only checked against `100 * cutoff` — the
//!   same acceptance factor the reference implementation applies to its own
//!   results.

use ariadnetor_core::{Complex, Scalar};
use ariadnetor_mps::{
    self as mps, ApplyMethod, CanonicalForm, Mps, SuccessiveRandomizedParams, TensorChain,
    TruncSvdParams, TruncateParams,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseLayout, DenseStorage};
use num_traits::NumCast;

use super::helpers::{
    assert_dense_close, cm_dense_tensor, densify, is_right_canonical, make_3site_test_mpo,
    make_3site_test_mps, make_4site_mps, make_identity_mpo, make_total_n_dense_mpo, mps_to_dense,
    rm_dense_tensor,
};

/// Fixed-rank SRC at a rank high enough to be clamped to the exactly
/// representable rank at every bond.
fn exact_rank_method(seed: u64) -> ApplyMethod {
    ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        output_dim: Some(usize::MAX),
        seed,
        ..Default::default()
    })
}

/// Relative Frobenius distance ‖a − e‖ / ‖e‖ over the densified states.
///
/// Densifying keeps full floating-point resolution; the inner-product form
/// `‖a‖² + ‖e‖² − 2 Re⟨e|a⟩` cancels catastrophically and cannot resolve
/// relative errors below ~sqrt(machine epsilon).
fn relative_error<T: Scalar>(
    backend: &NativeBackend,
    approx: &Mps<DenseStorage<T>, DenseLayout>,
    exact: &Mps<DenseStorage<T>, DenseLayout>,
) -> f64 {
    let a = densify(backend, approx);
    let e = densify(backend, exact);
    assert_eq!(a.shape(), e.shape(), "state shapes must agree");
    // Both operands come from the same contraction pipeline, so their
    // physical orders match and zipping raw storage is element-correct
    // (the assert_dense_close precedent).
    assert_eq!(a.order(), e.order(), "storage orders must agree");
    let mut diff_sq = 0.0_f64;
    let mut norm_sq = 0.0_f64;
    for (&av, &ev) in a.data_slice().iter().zip(e.data_slice().iter()) {
        // Scalar carries Add but not Sub; negate through scale_real.
        let neg_one = -<T::Real as num_traits::One>::one();
        let d: f64 = <f64 as NumCast>::from((av + ev.scale_real(neg_one)).abs()).expect("fits f64");
        let n: f64 = <f64 as NumCast>::from(ev.abs()).expect("fits f64");
        diff_sq += d * d;
        norm_sq += n * n;
    }
    diff_sq.sqrt() / norm_sq.sqrt()
}

// ===========================================================================
// Exactness and agreement
// ===========================================================================

#[test]
fn src_exact_rank_matches_streaming_naive() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let exact = mps::apply(&backend, &op, &psi, None);
    let out = mps::apply_with_method(&backend, &op, &psi, None, exact_rank_method(7));

    assert_dense_close(&mps_to_dense(&out), &mps_to_dense(&exact), 1e-10);
}

#[test]
fn src_adaptive_meets_relative_tolerance() {
    let backend = NativeBackend::new();
    let psi = make_4site_mps();
    let op = make_total_n_dense_mpo(4);
    let cutoff = 1e-6;

    let exact = mps::apply(&backend, &op, &psi, None);
    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: Some(cutoff),
        sketch_dim: 1,
        sketch_increment: 1,
        seed: 42,
        ..Default::default()
    });
    let out = mps::apply_with_method(&backend, &op, &psi, None, method);

    let rel = relative_error(&backend, &out, &exact);
    assert!(
        rel <= 100.0 * cutoff,
        "relative error {rel} exceeds 100x cutoff"
    );

    // Every output bond stays within the exactly representable rank of the
    // corresponding cut (left MPO bond * left MPS bond of the site pair).
    for j in 0..out.len() - 1 {
        let cap = op.site(j + 1).shape()[0] * psi.site(j + 1).shape()[0];
        assert!(
            out.bond_dim(j) <= cap,
            "bond {j} = {} exceeds representable rank {cap}",
            out.bond_dim(j)
        );
    }
}

#[test]
fn src_complex_matches_streaming_naive() {
    type C = Complex<f64>;
    let backend = NativeBackend::new();
    // Small complex chain with genuinely complex entries so the cap
    // update's conjugation is load-bearing (a sign error there is
    // invisible for real scalars).
    let c = |re: f64, im: f64| C::new(re, im);
    let psi: Mps<DenseStorage<C>, DenseLayout> = Mps::from_sites(vec![
        cm_dense_tensor(
            vec![c(1.0, 0.2), c(0.0, -0.3), c(0.5, 0.1), c(0.5, 0.4)],
            vec![1, 2, 2],
        ),
        cm_dense_tensor(
            (1..=8)
                .map(|i| c(i as f64 * 0.1, -(i as f64) * 0.05))
                .collect(),
            vec![2, 2, 2],
        ),
        cm_dense_tensor(
            vec![c(1.0, -0.1), c(0.0, 0.6), c(0.2, 0.0), c(1.0, 0.3)],
            vec![2, 2, 1],
        ),
    ]);
    let op = mps::Mpo::from_sites(vec![
        cm_dense_tensor(
            (1..=8)
                .map(|i| c(i as f64 * 0.1, (i as f64) * 0.07))
                .collect(),
            vec![1, 2, 2, 2],
        ),
        cm_dense_tensor(
            (1..=16)
                .map(|i| c(i as f64 * 0.05, -(i as f64) * 0.02))
                .collect(),
            vec![2, 2, 2, 2],
        ),
        cm_dense_tensor(
            (1..=8)
                .map(|i| c(-(i as f64) * 0.1, i as f64 * 0.04))
                .collect(),
            vec![2, 2, 2, 1],
        ),
    ]);

    let exact = mps::apply(&backend, &op, &psi, None);
    let out = mps::apply_with_method(&backend, &op, &psi, None, exact_rank_method(11));

    let rel = relative_error(&backend, &out, &exact);
    assert!(rel < 1e-10, "complex SRC deviates: relative error {rel}");
}

#[test]
fn src_single_site_chain_is_exact() {
    let backend = NativeBackend::new();
    let psi: Mps<DenseStorage<f64>, DenseLayout> =
        Mps::from_sites(vec![cm_dense_tensor(vec![0.6, 0.8], vec![1, 2, 1])]);
    let op = make_identity_mpo(1, 2);

    let out = mps::apply_with_method(&backend, &op, &psi, None, exact_rank_method(3));

    assert_eq!(*out.canonical_form(), CanonicalForm::Mixed { center: 0 });
    assert_dense_close(&mps_to_dense(&out), &mps_to_dense(&psi), 1e-12);
}

#[test]
fn src_rank_deficient_sketch_stops_exactly() {
    let backend = NativeBackend::new();
    // Product state carried on an artificially inflated bond (second
    // channel zero): the caps allow p = 2 while the true rank is 1, so the
    // sketch's R factor is singular and the rank guard must stop without
    // inverting it.
    let psi: Mps<DenseStorage<f64>, DenseLayout> = Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0, 0.0, 2.0, 0.0], vec![1, 2, 2]),
        cm_dense_tensor(vec![0.5, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0], vec![2, 2, 2]),
        cm_dense_tensor(vec![1.0, 0.0, 3.0, 0.0], vec![2, 2, 1]),
    ]);
    let op = make_identity_mpo(3, 2);

    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: Some(1e-10),
        sketch_dim: 2,
        seed: 5,
        ..Default::default()
    });
    let out = mps::apply_with_method(&backend, &op, &psi, None, method);

    assert_dense_close(&mps_to_dense(&out), &mps_to_dense(&psi), 1e-10);
}

#[test]
fn src_rank_deficiency_after_growth_restores_isometry() {
    let backend = NativeBackend::new();
    // Sum of three product states u_c^{x4} with u = {(1,0), (0,1), (1,1)},
    // carried on bonds padded to 6: the true rank at every interior cut is
    // 3 while the caps allow up to 6. Starting at sketch_dim = 2 the first
    // sketch block is full-rank; the growth round to 4 then crosses the
    // true rank, so a LATER block is the rank-deficient one — the case
    // where the assembled basis can lose orthonormality. At the leftmost
    // cut the cap (6) is above 4, so the rank-deficient outcome, not the
    // cap, is what stops the growth there.
    const PAD: usize = 6;
    let u = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let mut first = vec![0.0; 2 * PAD];
    let mut middle = vec![0.0; PAD * 2 * PAD];
    let mut last = vec![0.0; PAD * 2];
    for (c, uc) in u.iter().enumerate() {
        for (d, v) in uc.iter().enumerate() {
            first[d * PAD + c] = *v;
            middle[c * 2 * PAD + d * PAD + c] = *v;
            last[c * 2 + d] = *v;
        }
    }
    let psi: Mps<DenseStorage<f64>, DenseLayout> = Mps::from_sites(vec![
        rm_dense_tensor(first, vec![1, 2, PAD]),
        rm_dense_tensor(middle.clone(), vec![PAD, 2, PAD]),
        rm_dense_tensor(middle, vec![PAD, 2, PAD]),
        rm_dense_tensor(last, vec![PAD, 2, 1]),
    ]);
    let op = make_identity_mpo(4, 2);

    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: Some(1e-12),
        sketch_dim: 2,
        sketch_increment: 2,
        seed: 3,
        ..Default::default()
    });
    let out = mps::apply_with_method(&backend, &op, &psi, None, method);

    assert_dense_close(&mps_to_dense(&out), &mps_to_dense(&psi), 1e-10);
    assert_eq!(*out.canonical_form(), CanonicalForm::Mixed { center: 0 });
    for j in 1..out.len() {
        assert!(
            is_right_canonical(out.site(j), 1e-10),
            "site {j} must stay an isometry through the deficient growth round"
        );
    }
    // The leftmost cut's growth is stopped by the rank-deficient outcome
    // rather than by its cap of 6: an implementation that ignored the
    // outcome would keep sketching to that cap.
    assert_eq!(
        out.bond_dim(0),
        4,
        "growth must stop at the rank-deficient append, not at the cap"
    );
}

#[test]
fn src_zero_product_stops_immediately() {
    let backend = NativeBackend::new();
    let psi: Mps<DenseStorage<f64>, DenseLayout> = Mps::from_sites(vec![
        cm_dense_tensor(vec![0.0; 4], vec![1, 2, 2]),
        cm_dense_tensor(vec![0.0; 8], vec![2, 2, 2]),
        cm_dense_tensor(vec![0.0; 4], vec![2, 2, 1]),
    ]);
    let op = make_identity_mpo(3, 2);

    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: Some(1e-8),
        seed: 9,
        ..Default::default()
    });
    let out = mps::apply_with_method(&backend, &op, &psi, None, method);

    let v = mps_to_dense(&out);
    assert!(v.norm() < 1e-14, "zero product must stay zero");
}

// ===========================================================================
// Output structure
// ===========================================================================

#[test]
fn src_canonical_form_and_isometry() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let out = mps::apply_with_method(&backend, &op, &psi, None, exact_rank_method(13));

    assert_eq!(*out.canonical_form(), CanonicalForm::Mixed { center: 0 });
    for j in 1..out.len() {
        assert!(
            is_right_canonical(out.site(j), 1e-10),
            "site {j} is not right-canonical"
        );
    }
}

#[test]
fn src_same_seed_is_reproducible() {
    let backend = NativeBackend::new();
    let psi = make_4site_mps();
    let op = make_total_n_dense_mpo(4);
    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: Some(1e-6),
        seed: 21,
        ..Default::default()
    });

    let a = mps::apply_with_method(&backend, &op, &psi, None, method);
    let b = mps::apply_with_method(&backend, &op, &psi, None, method);

    for j in 0..a.len() {
        assert_eq!(a.site(j).shape(), b.site(j).shape(), "site {j} shape");
        assert_eq!(
            a.site(j).data_slice(),
            b.site(j).data_slice(),
            "site {j} data not bit-identical"
        );
    }
}

// ===========================================================================
// Parameter branches
// ===========================================================================

#[test]
fn src_forced_growth_from_small_sketch() {
    let backend = NativeBackend::new();
    let psi = make_4site_mps();
    let op = make_total_n_dense_mpo(4);

    let cutoff = 1e-10;
    let exact = mps::apply(&backend, &op, &psi, None);
    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: Some(cutoff),
        sketch_dim: 1,
        sketch_increment: 1,
        seed: 17,
        ..Default::default()
    });
    let out = mps::apply_with_method(&backend, &op, &psi, None, method);

    // A tight cutoff on a genuinely entangled product cannot be met at
    // sketch size 1, so growth must have occurred for accuracy to hold.
    assert!(
        out.max_bond_dim() > 1,
        "sketch never grew beyond the initial size"
    );
    let rel = relative_error(&backend, &out, &exact);
    assert!(
        rel <= 100.0 * cutoff,
        "relative error {rel} exceeds 100x cutoff"
    );
}

#[test]
fn src_min_dim_floors_the_bond() {
    let backend = NativeBackend::new();
    // Identity on a bond-2 state: a loose cutoff is satisfied immediately,
    // so only min_dim forces the bond up to the floor (clamped to the cap).
    let psi = make_3site_test_mps();
    let op = make_identity_mpo(3, 2);

    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: Some(0.5),
        sketch_dim: 1,
        sketch_increment: 1,
        min_dim: 2,
        seed: 31,
        ..Default::default()
    });
    let out = mps::apply_with_method(&backend, &op, &psi, None, method);

    for j in 0..out.len() - 1 {
        let cap = op.site(j + 1).shape()[0] * psi.site(j + 1).shape()[0];
        assert!(
            out.bond_dim(j) >= 2.min(cap),
            "bond {j} = {} below the min_dim floor",
            out.bond_dim(j)
        );
    }
}

#[test]
fn src_fixed_rank_sets_every_bond_to_output_dim() {
    let backend = NativeBackend::new();
    let psi = make_4site_mps();
    let op = make_total_n_dense_mpo(4);

    // Every cut's representable rank exceeds 2 here (left MPO bond * left
    // MPS bond is at least 2 * 3), so an interior output_dim must land
    // exactly, not merely as an upper bound.
    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        output_dim: Some(2),
        seed: 37,
        ..Default::default()
    });
    let out = mps::apply_with_method(&backend, &op, &psi, None, method);

    for j in 0..out.len() - 1 {
        assert_eq!(out.bond_dim(j), 2, "bond {j} must equal output_dim");
    }
}

#[test]
fn src_max_dim_caps_every_bond() {
    let backend = NativeBackend::new();
    let psi = make_4site_mps();
    let op = make_total_n_dense_mpo(4);

    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: Some(1e-12),
        max_dim: Some(1),
        seed: 23,
        ..Default::default()
    });
    let out = mps::apply_with_method(&backend, &op, &psi, None, method);

    assert_eq!(out.max_bond_dim(), 1, "max_dim = 1 must cap every bond");
}

#[test]
fn src_finishing_pass_truncates() {
    let backend = NativeBackend::new();
    let psi = make_4site_mps();
    let op = make_total_n_dense_mpo(4);

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    let out = mps::apply_with_method(&backend, &op, &psi, Some(&params), exact_rank_method(19));

    assert_eq!(
        out.max_bond_dim(),
        1,
        "finishing pass must truncate to chi_max"
    );
}

// ===========================================================================
// Validation panics
// ===========================================================================

#[test]
#[should_panic(expected = "dense-only")]
fn src_panics_on_block_sparse() {
    use super::helpers::{make_2site_entangled_u1_mps, make_identity_u1_mpo};
    let backend = NativeBackend::new();
    let psi = make_2site_entangled_u1_mps();
    let op = make_identity_u1_mpo(2);

    let _ = mps::apply_with_method(
        &backend,
        &op,
        &psi,
        None,
        ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams::default()),
    );
}

#[test]
#[should_panic(expected = "requires a cutoff")]
fn src_panics_without_stopping_rule() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: None,
        ..Default::default()
    });
    let _ = mps::apply_with_method(&backend, &op, &psi, None, method);
}

#[test]
#[should_panic(expected = "finite and non-negative")]
fn src_fixed_mode_still_validates_ignored_fields() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    // Fixed mode ignores cutoff, but the documented contract still
    // validates it for configuration hygiene.
    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        output_dim: Some(2),
        cutoff: Some(f64::NAN),
        ..Default::default()
    });
    let _ = mps::apply_with_method(&backend, &op, &psi, None, method);
}

#[test]
#[should_panic(expected = "representable as a finite value")]
fn src_f32_rejects_unrepresentable_cutoff() {
    let backend = NativeBackend::new();
    let psi: Mps<DenseStorage<f32>, DenseLayout> = Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0_f32, 0.0, 0.5, 0.5], vec![1, 2, 2]),
        cm_dense_tensor(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2, 1]),
    ]);
    let op = mps::Mpo::from_sites(vec![
        cm_dense_tensor(vec![1.0_f32, 0.0, 0.0, 1.0], vec![1, 2, 2, 1]),
        cm_dense_tensor(vec![1.0_f32, 0.0, 0.0, 1.0], vec![1, 2, 2, 1]),
    ]);

    // 1e300 is finite in f64 (passes validation) but overflows f32; the
    // conversion gate must reject it instead of silently converging.
    let method = ApplyMethod::SuccessiveRandomized(SuccessiveRandomizedParams {
        cutoff: Some(1e300),
        ..Default::default()
    });
    let _ = mps::apply_with_method(&backend, &op, &psi, None, method);
}

/// Guard against a Y-assembly or QR-orientation error that a uniform-bond
/// fixture could hide: unequal adjacent bond dimensions make the (d, cap)
/// row fusion and the left-leg cap computation genuinely asymmetric.
#[test]
fn src_unequal_bonds_agree_with_streaming_naive() {
    let backend = NativeBackend::new();
    let psi = make_4site_mps(); // bonds (4, 4, 3): non-uniform
    let op = make_total_n_dense_mpo(4);

    let exact = mps::apply(&backend, &op, &psi, None);
    let out = mps::apply_with_method(&backend, &op, &psi, None, exact_rank_method(29));

    assert_dense_close(&mps_to_dense(&out), &mps_to_dense(&exact), 1e-9);
}
