//! Tests for the SRC linear-combination apply
//! (`apply_sum_successive_randomized`) and the SRC-based rounding
//! (`Mps::round_successive_randomized`).
//!
//! The assertion regimes follow the single-pair SRC suite: exactness with
//! probability one at (or clamped to) the exactly representable rank, and
//! a `100 * cutoff` acceptance factor for adaptive-mode relative errors.
//! References are built independently of the kernel under test: each
//! term's lossless streaming-naive product is densified and the dense
//! state vectors are combined.

use ariadnetor_core::{Complex, Scalar};
use ariadnetor_mps::{
    self as mps, ApplyMethod, CanonicalForm, Mpo, Mps, SuccessiveRandomizedParams, TensorChain,
    TruncSvdParams, TruncateParams, apply_sum_successive_randomized,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor, linear_combine};

use super::helpers::{
    apply_ok, cm_dense_tensor, densify, is_right_canonical, make_3site_test_mpo,
    make_3site_test_mps, make_4site_mps, make_total_n_dense_mpo, relative_frobenius,
    with_site_scaled,
};

type DenseMps<T> = Mps<DenseStorage<T>, DenseLayout>;
type DenseMpo<T> = Mpo<DenseStorage<T>, DenseLayout>;

/// Fixed-rank params clamped to the exactly representable rank at every
/// bond, where the sum is exact with probability one.
fn exact_rank_params(seed: u64) -> SuccessiveRandomizedParams {
    SuccessiveRandomizedParams {
        output_dim: Some(usize::MAX),
        seed,
        ..Default::default()
    }
}

/// Densified reference for `sum_t coeffs[t] * H_t psi_t`: each term goes
/// through the lossless streaming-naive apply independently, so the
/// reference never touches the sketching kernel under test.
fn lincomb_dense_reference<T: Scalar>(
    backend: &NativeBackend,
    terms: &[(&DenseMpo<T>, &DenseMps<T>)],
    coeffs: &[T],
) -> DenseTensor<T> {
    let denses: Vec<DenseTensor<T>> = terms
        .iter()
        .map(|(op, psi)| densify(backend, &mps::apply(backend, *op, *psi, None)))
        .collect();
    let refs: Vec<&DenseTensor<T>> = denses.iter().collect();
    linear_combine(&refs, coeffs).expect("reference terms share shape and order")
}

/// A second deterministic 3-site MPS, distinct from `make_3site_test_mps`
/// so two-term sums combine genuinely different states.
fn make_3site_alt_mps() -> DenseMps<f64> {
    Mps::from_sites(vec![
        cm_dense_tensor(vec![0.3, -0.4, 0.1, 0.9], vec![1, 2, 2]),
        cm_dense_tensor(
            (1..=8).map(|i| 0.9 - i as f64 * 0.07).collect(),
            vec![2, 2, 2],
        ),
        cm_dense_tensor(vec![0.2, 0.8, -0.5, 0.6], vec![2, 2, 1]),
    ])
}

/// Deterministic complex 3-site MPS/MPO pair (physical dim 2).
fn make_3site_c64_fixtures() -> (DenseMpo<Complex<f64>>, DenseMps<Complex<f64>>) {
    let c = |v: Vec<f64>| -> Vec<Complex<f64>> {
        v.iter()
            .enumerate()
            .map(|(i, &x)| Complex::new(x, 0.1 * i as f64 - 0.3))
            .collect()
    };
    let psi = Mps::from_sites(vec![
        cm_dense_tensor(c(vec![1.0, 0.2, 0.5, -0.5]), vec![1, 2, 2]),
        cm_dense_tensor(c((1..=8).map(|i| i as f64 * 0.1).collect()), vec![2, 2, 2]),
        cm_dense_tensor(c(vec![0.9, 0.1, -0.2, 1.0]), vec![2, 2, 1]),
    ]);
    let op = Mpo::from_sites(vec![
        cm_dense_tensor(
            c((1..=8).map(|i| i as f64 * 0.1).collect()),
            vec![1, 2, 2, 2],
        ),
        cm_dense_tensor(
            c((1..=16).map(|i| i as f64 * 0.05).collect()),
            vec![2, 2, 2, 2],
        ),
        cm_dense_tensor(
            c((1..=8).map(|i| 0.8 - i as f64 * 0.1).collect()),
            vec![2, 2, 2, 1],
        ),
    ]);
    (op, psi)
}

// ===========================================================================
// Single-term equivalence
// ===========================================================================

/// The sum entry over one coefficient-one term must be bit-identical to
/// the single-pair entry at the same seed: it pins both the RNG-stream
/// preservation of the shared-Gaussian refactor and the coefficient-one
/// arithmetic bypass (a `0 + 1 * x` pass through the combiner would break
/// exactness under signed zero and could flip adaptive decisions).
#[test]
fn src_sum_single_term_is_bit_identical_to_single_pair() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();
    let src = SuccessiveRandomizedParams {
        cutoff: Some(1e-8),
        sketch_dim: 1,
        sketch_increment: 1,
        seed: 42,
        ..Default::default()
    };

    let single = apply_ok(
        &backend,
        &op,
        &psi,
        None,
        ApplyMethod::SuccessiveRandomized(src),
    );
    let summed = apply_sum_successive_randomized(&backend, &[(&op, &psi)], &[1.0], None, src)
        .expect("finite inputs");

    assert_eq!(single.len(), summed.len());
    for j in 0..single.len() {
        assert_eq!(single.site(j).shape(), summed.site(j).shape(), "site {j}");
        assert_eq!(
            single.site(j).data_slice(),
            summed.site(j).data_slice(),
            "site {j} must be bit-identical"
        );
    }
    assert_eq!(single.canonical_form(), summed.canonical_form());
}

// ===========================================================================
// Sums against independent dense references
// ===========================================================================

#[test]
fn src_sum_exact_rank_matches_dense_reference() {
    let backend = NativeBackend::new();
    let psi1 = make_3site_test_mps();
    let psi2 = make_3site_alt_mps();
    let op1 = make_3site_test_mpo();
    let op2 = make_total_n_dense_mpo(3);
    let terms = [(&op1, &psi1), (&op2, &psi2)];
    let coeffs = [0.75, -1.5];

    let reference = lincomb_dense_reference(&backend, &terms, &coeffs);
    let out =
        apply_sum_successive_randomized(&backend, &terms, &coeffs, None, exact_rank_params(7))
            .expect("finite inputs");

    assert!(relative_frobenius(&densify(&backend, &out), &reference) < 1e-10);
    assert_eq!(out.canonical_form(), &CanonicalForm::Mixed { center: 0 });
    for j in 1..out.len() {
        assert!(is_right_canonical(out.site(j), 1e-10), "site {j}");
    }
}

/// Adaptive two-term sum on 4 sites with mixed per-term MPO bond
/// dimensions (a bond-2 operator and a bond-1 identity), forced through
/// sketch growth by a minimal initial sketch.
#[test]
fn src_sum_adaptive_meets_relative_tolerance() {
    let backend = NativeBackend::new();
    let psi1 = make_4site_mps();
    let psi2 = with_site_scaled(&psi1, 2, -0.6);
    let op1 = make_total_n_dense_mpo(4);
    let op2 = Mpo::identity(&[2, 2, 2, 2]);
    let terms = [(&op1, &psi1), (&op2, &psi2)];
    let coeffs = [1.0, 0.5];
    let cutoff = 1e-6;

    let reference = lincomb_dense_reference(&backend, &terms, &coeffs);
    let src = SuccessiveRandomizedParams {
        cutoff: Some(cutoff),
        sketch_dim: 1,
        sketch_increment: 1,
        seed: 42,
        ..Default::default()
    };
    let out = apply_sum_successive_randomized(&backend, &terms, &coeffs, None, src)
        .expect("finite inputs");

    let rel = relative_frobenius(&densify(&backend, &out), &reference);
    assert!(
        rel <= 100.0 * cutoff,
        "relative error {rel} vs cutoff {cutoff}"
    );
}

#[test]
fn src_sum_complex_coefficients_match_dense_reference() {
    let backend = NativeBackend::new();
    let (op1, psi1) = make_3site_c64_fixtures();
    let op2: DenseMpo<Complex<f64>> = Mpo::identity(&[2, 2, 2]);
    let psi2 = with_site_scaled(&psi1, 0, Complex::new(-0.3, 0.4));
    let terms = [(&op1, &psi1), (&op2, &psi2)];
    let coeffs = [Complex::new(0.5, -1.0), Complex::new(0.0, 2.0)];

    let reference = lincomb_dense_reference(&backend, &terms, &coeffs);
    let out =
        apply_sum_successive_randomized(&backend, &terms, &coeffs, None, exact_rank_params(11))
            .expect("finite inputs");

    assert!(relative_frobenius(&densify(&backend, &out), &reference) < 1e-10);
}

/// Exact cancellation between terms: with the Gaussian block shared
/// across terms the summed sketch is exactly zero, the zero-norm break
/// fires, and the result is the zero state. Independent per-term draws
/// would sketch this sum as nonzero — this is the case that makes the
/// sharing a correctness requirement rather than an optimization.
#[test]
fn src_sum_cancelling_terms_yield_zero_state() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();
    let terms = [(&op, &psi), (&op, &psi)];
    let coeffs = [1.0, -1.0];

    let src = SuccessiveRandomizedParams {
        cutoff: Some(1e-6),
        seed: 3,
        ..Default::default()
    };
    let out = apply_sum_successive_randomized(&backend, &terms, &coeffs, None, src)
        .expect("finite inputs");

    let dense = densify(&backend, &out);
    for (i, x) in dense.data_slice().iter().enumerate() {
        assert!(x.abs() <= 1e-14, "elem {i} of the zero state: {x}");
    }
    assert_eq!(out.canonical_form(), &CanonicalForm::Mixed { center: 0 });
}

/// A single term with a non-unit coefficient takes the non-bypass
/// combiner branch: the coefficient must land exactly once in the final
/// amplitude (sketch panels select the basis; the first-site center
/// carries the weight).
#[test]
fn src_sum_single_term_non_unit_coefficient_scales_once() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();
    let terms = [(&op, &psi)];
    let coeffs = [2.0];

    let reference = lincomb_dense_reference(&backend, &terms, &coeffs);
    let out =
        apply_sum_successive_randomized(&backend, &terms, &coeffs, None, exact_rank_params(17))
            .expect("finite inputs");

    assert!(relative_frobenius(&densify(&backend, &out), &reference) < 1e-10);
}

/// A zero coefficient disables its term entirely: even a non-finite
/// element in the disabled term must not poison the result (`0 * inf`
/// would be NaN if the term flowed through the combiner).
#[test]
fn src_sum_zero_coefficient_disables_poisoned_term() {
    let backend = NativeBackend::new();
    let psi1 = make_3site_test_mps();
    let psi2 = with_site_scaled(&psi1, 1, f64::INFINITY);
    let op = make_3site_test_mpo();
    let terms = [(&op, &psi1), (&op, &psi2)];
    let coeffs = [1.0, 0.0];

    let reference = lincomb_dense_reference(&backend, &[(&op, &psi1)], &[1.0]);
    let out =
        apply_sum_successive_randomized(&backend, &terms, &coeffs, None, exact_rank_params(19))
            .expect("the zero-weighted poisoned term must not surface as NonFinite");

    assert!(relative_frobenius(&densify(&backend, &out), &reference) < 1e-10);
}

/// All coefficients zero: the sum is the zero state at bond dimension 1.
#[test]
fn src_sum_all_zero_coefficients_yield_zero_state() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let out = apply_sum_successive_randomized(
        &backend,
        &[(&op, &psi)],
        &[0.0],
        None,
        SuccessiveRandomizedParams::default(),
    )
    .expect("finite inputs");

    assert_eq!(out.max_bond_dim(), 1);
    assert_eq!(out.canonical_form(), &CanonicalForm::Mixed { center: 0 });
    // The claimed Mixed form must hold structurally: off-center sites are
    // genuine right-isometries even though the state is zero.
    for j in 1..out.len() {
        assert!(is_right_canonical(out.site(j), 1e-14), "site {j}");
    }
    for x in densify(&backend, &out).data_slice() {
        assert_eq!(*x, 0.0);
    }
}

/// `params: Some(...)` routes the summed result through the standard
/// `canonicalize` + `truncate` finishing pass: bonds obey `chi_max` and
/// the requested center holds.
#[test]
fn src_sum_finishing_pass_truncates_summed_result() {
    let backend = NativeBackend::new();
    let psi1 = make_3site_test_mps();
    let psi2 = make_3site_alt_mps();
    let op1 = make_3site_test_mpo();
    let op2 = make_total_n_dense_mpo(3);
    let terms = [(&op1, &psi1), (&op2, &psi2)];
    let coeffs = [0.75, -1.5];

    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });
    let out = apply_sum_successive_randomized(
        &backend,
        &terms,
        &coeffs,
        Some(&params),
        exact_rank_params(23),
    )
    .expect("finite inputs");

    assert!(
        out.max_bond_dim() <= 2,
        "chi_max applied to the summed result"
    );
    assert_eq!(out.canonical_form(), &CanonicalForm::Mixed { center: 0 });
    for x in densify(&backend, &out).data_slice() {
        assert!(x.is_finite());
    }
}

// ===========================================================================
// Identity MPO and rounding
// ===========================================================================

#[test]
fn mpo_identity_lossless_apply_preserves_state() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let id = Mpo::identity(&[2, 2, 2]);

    let out = mps::apply(&backend, &id, &psi, None);
    assert!(relative_frobenius(&densify(&backend, &out), &densify(&backend, &psi)) < 1e-12);
}

/// Rounding an MPS whose stored bonds exceed its physical Schmidt ranks:
/// `make_4site_mps` stores bonds (4, 4, 3) while dimension counting caps
/// the first and last cuts at rank 2 (a 2 x 8 / 8 x 2 matricization).
/// Adaptive SRC detects the deficient rank one sketch increment past it,
/// so with unit growth the first cut lands at most one above the rank
/// (strictly below the stored bond) and the last cut is capped by its
/// row count.
#[test]
fn src_rounding_compresses_redundant_bonds() {
    let backend = NativeBackend::new();
    let psi = make_4site_mps();
    let mut rounded = psi.clone();
    rounded
        .round_successive_randomized(
            &backend,
            SuccessiveRandomizedParams {
                cutoff: Some(1e-10),
                sketch_dim: 1,
                sketch_increment: 1,
                seed: 5,
                ..Default::default()
            },
        )
        .expect("finite input");

    let rel = relative_frobenius(&densify(&backend, &rounded), &densify(&backend, &psi));
    assert!(rel < 1e-8, "rounding must preserve the state: {rel}");
    assert!(
        rounded.site(1).shape()[0] <= 3,
        "first cut within one increment of its dimension-counting rank"
    );
    assert!(
        rounded.site(3).shape()[0] <= 2,
        "last cut clamped to its row count"
    );
    assert_eq!(
        rounded.canonical_form(),
        &CanonicalForm::Mixed { center: 0 }
    );
    for j in 1..rounded.len() {
        assert!(is_right_canonical(rounded.site(j), 1e-9), "site {j}");
    }
}

// ===========================================================================
// Validation panics
// ===========================================================================

#[test]
#[should_panic(expected = "dense-only")]
fn src_sum_panics_on_block_sparse() {
    use super::helpers::{make_2site_entangled_u1_mps, make_identity_u1_mpo};
    let backend = NativeBackend::new();
    let psi = make_2site_entangled_u1_mps();
    let op = make_identity_u1_mpo(2);

    let _ = apply_sum_successive_randomized(
        &backend,
        &[(&op, &psi)],
        &[1.0],
        None,
        SuccessiveRandomizedParams::default(),
    );
}

#[test]
#[should_panic(expected = "dense-only")]
fn src_rounding_panics_on_block_sparse() {
    use super::helpers::make_2site_entangled_u1_mps;
    let backend = NativeBackend::new();
    let mut psi = make_2site_entangled_u1_mps();

    let _ = psi.round_successive_randomized(&backend, SuccessiveRandomizedParams::default());
}

#[test]
#[should_panic(expected = "chain must have at least one site")]
fn src_rounding_panics_on_empty_chain() {
    let backend = NativeBackend::new();
    let mut psi: DenseMps<f64> = Mps::empty();

    let _ = psi.round_successive_randomized(&backend, SuccessiveRandomizedParams::default());
}

#[test]
#[should_panic(expected = "terms must be non-empty")]
fn src_sum_panics_on_empty_terms() {
    let backend = NativeBackend::new();
    let terms: [(&DenseMpo<f64>, &DenseMps<f64>); 0] = [];

    let _ = apply_sum_successive_randomized(
        &backend,
        &terms,
        &[],
        None,
        SuccessiveRandomizedParams::default(),
    );
}

#[test]
#[should_panic(expected = "one coefficient per term")]
fn src_sum_panics_on_coefficient_count_mismatch() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let _ = apply_sum_successive_randomized(
        &backend,
        &[(&op, &psi), (&op, &psi)],
        &[1.0],
        None,
        SuccessiveRandomizedParams::default(),
    );
}

#[test]
#[should_panic(expected = "coefficients must be finite")]
fn src_sum_panics_on_non_finite_coefficient() {
    let backend = NativeBackend::new();
    let psi = make_3site_test_mps();
    let op = make_3site_test_mpo();

    let _ = apply_sum_successive_randomized(
        &backend,
        &[(&op, &psi)],
        &[f64::NAN],
        None,
        SuccessiveRandomizedParams::default(),
    );
}

#[test]
#[should_panic(expected = "equal length")]
fn src_sum_panics_on_length_mismatch() {
    let backend = NativeBackend::new();
    let psi1 = make_3site_test_mps();
    let op1 = make_3site_test_mpo();
    let psi2: DenseMps<f64> = Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0, 0.0], vec![1, 2, 1]),
        cm_dense_tensor(vec![0.0, 1.0], vec![1, 2, 1]),
    ]);
    let op2 = Mpo::identity(&[2, 2]);

    let _ = apply_sum_successive_randomized(
        &backend,
        &[(&op1, &psi1), (&op2, &psi2)],
        &[1.0, 1.0],
        None,
        SuccessiveRandomizedParams::default(),
    );
}

/// Two single-site chains take the bond-free exact path; the weighted sum
/// of local products must still honor the coefficients.
#[test]
fn src_sum_single_site_chains_match_dense_reference() {
    let backend = NativeBackend::new();
    let psi1: DenseMps<f64> =
        Mps::from_sites(vec![cm_dense_tensor(vec![0.6, -0.8], vec![1, 2, 1])]);
    let psi2: DenseMps<f64> = Mps::from_sites(vec![cm_dense_tensor(vec![0.3, 0.7], vec![1, 2, 1])]);
    let op1 = Mpo::identity(&[2]);
    let op2: DenseMpo<f64> = Mpo::from_sites(vec![cm_dense_tensor(
        vec![0.2, 1.1, -0.4, 0.9],
        vec![1, 2, 2, 1],
    )]);
    let terms = [(&op1, &psi1), (&op2, &psi2)];
    let coeffs = [2.0, -0.5];

    let reference = lincomb_dense_reference(&backend, &terms, &coeffs);
    let out =
        apply_sum_successive_randomized(&backend, &terms, &coeffs, None, exact_rank_params(13))
            .expect("finite inputs");

    assert!(relative_frobenius(&densify(&backend, &out), &reference) < 1e-12);
}

/// A non-finite element in one term's network must surface as
/// `ApplyError::NonFinite` from the summed panel — the multi-term
/// combining path, which the single-term bypass never exercises.
#[test]
fn src_sum_surfaces_non_finite_term_as_error() {
    let backend = NativeBackend::new();
    let psi1 = make_3site_test_mps();
    let psi2 = with_site_scaled(&psi1, 1, f64::INFINITY);
    let op = make_3site_test_mpo();
    let terms = [(&op, &psi1), (&op, &psi2)];

    let result = apply_sum_successive_randomized(
        &backend,
        &terms,
        &[1.0, 1.0],
        None,
        SuccessiveRandomizedParams::default(),
    );
    assert!(
        matches!(result, Err(mps::ApplyError::NonFinite { .. })),
        "poisoned term must not flow on as Ok"
    );
}

#[test]
#[should_panic(expected = "at least 1")]
fn mpo_identity_panics_on_zero_dimension() {
    let _: DenseMpo<f64> = Mpo::identity(&[2, 0]);
}

#[test]
#[should_panic(expected = "match within a term")]
fn src_sum_panics_on_within_term_dimension_mismatch() {
    let backend = NativeBackend::new();
    // MPO ket dimension 2 against an MPS with physical dimension 3.
    let psi: DenseMps<f64> = Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0, 0.0, 0.2], vec![1, 3, 1]),
        cm_dense_tensor(vec![0.0, 1.0, -0.1], vec![1, 3, 1]),
        cm_dense_tensor(vec![0.5, 0.5, 0.3], vec![1, 3, 1]),
    ]);
    let op = make_3site_test_mpo();

    let _ = apply_sum_successive_randomized(
        &backend,
        &[(&op, &psi)],
        &[1.0],
        None,
        SuccessiveRandomizedParams::default(),
    );
}

#[test]
#[should_panic(expected = "agree across terms")]
fn src_sum_panics_on_output_dimension_mismatch() {
    let backend = NativeBackend::new();
    let psi1 = make_3site_test_mps();
    let op1 = make_3site_test_mpo();
    // Physical dimension 3 throughout: within-term consistent, but the
    // output legs cannot line up with the physical-dimension-2 first term.
    let psi2: DenseMps<f64> = Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0, 0.0, 0.2], vec![1, 3, 1]),
        cm_dense_tensor(vec![0.0, 1.0, -0.1], vec![1, 3, 1]),
        cm_dense_tensor(vec![0.5, 0.5, 0.3], vec![1, 3, 1]),
    ]);
    let op2 = Mpo::identity(&[3, 3, 3]);

    let _ = apply_sum_successive_randomized(
        &backend,
        &[(&op1, &psi1), (&op2, &psi2)],
        &[1.0, 1.0],
        None,
        SuccessiveRandomizedParams::default(),
    );
}
