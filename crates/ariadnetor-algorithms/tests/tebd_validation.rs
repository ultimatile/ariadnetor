//! Error paths of the Dense TEBD driver. Every rejected input returns
//! before touching the state; a state whose norm breaks down is left as the
//! failing step produced it.

use algorithms_fixtures::dense_fixtures::{
    random_bond_terms_c64, random_mps_c64, random_mps_unknown_f64,
};
use ariadnetor_algorithms::tebd::{TebdError, TebdParams, TimeStep, tebd};
use ariadnetor_core::Scalar;
use ariadnetor_mps::{CanonicalForm, Mps, TensorChain, TruncSvdParams};
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor};
use num_complex::Complex;

type C64 = Complex<f64>;
type DenseMps<T> = Mps<DenseStorage<T>, DenseLayout>;

const PHYS: [usize; 5] = [2, 3, 2, 3, 2];
const BONDS: [usize; 6] = [1, 2, 4, 4, 3, 1];

const NO_TRUNC: TruncSvdParams = TruncSvdParams {
    chi_max: None,
    target_trunc_err: None,
};

const STEP_DETAIL: &str = "time step must be finite and positive";
const HALF_STEP_DETAIL: &str =
    "half time step is not a positive finite value in the storage's real scalar type";

fn params(time_step: TimeStep, trunc: TruncSvdParams) -> TebdParams {
    TebdParams {
        time_step,
        n_steps: 2,
        trunc,
    }
}

fn fixture() -> (DenseMps<C64>, Vec<DenseTensor<C64>>) {
    (
        random_mps_c64(&PHYS, &BONDS, 3),
        random_bond_terms_c64(&PHYS, 50),
    )
}

fn assert_unchanged<T: Scalar + PartialEq>(out: &DenseMps<T>, psi: &DenseMps<T>) {
    assert_eq!(out.canonical_form(), psi.canonical_form());
    assert_eq!(out.len(), psi.len());
    for j in 0..psi.len() {
        let (a, b) = (out.site(j), psi.site(j));
        assert!(
            a.shape() == b.shape() && a.order() == b.order() && a.data_slice() == b.data_slice(),
            "site {j} changed"
        );
    }
}

/// Run `tebd` on a copy of `psi`, expecting an error, and check that the
/// copy is left exactly as it was.
fn rejected<T: Scalar + PartialEq>(
    psi: &DenseMps<T>,
    terms: &[DenseTensor<T>],
    params: &TebdParams,
) -> TebdError {
    let mut out = psi.clone();
    let Err(err) = tebd(&mut out, terms, params) else {
        panic!("must be rejected");
    };
    assert_unchanged(&out, psi);
    err
}

fn invalid_detail(err: TebdError) -> &'static str {
    match err {
        TebdError::InvalidParams { detail } => detail,
        other => panic!("expected InvalidParams, got {other:?}"),
    }
}

#[test]
fn rejects_a_chain_shorter_than_two_sites() {
    let p = params(TimeStep::Real(0.1), NO_TRUNC);
    let one = random_mps_c64(&PHYS[..1], &BONDS[..2], 3);
    let err = rejected(&one, &[], &p);
    assert!(
        matches!(err, TebdError::TooFewSites { n_sites: 1 }),
        "{err:?}"
    );
    let err = rejected(&DenseMps::<C64>::empty(), &[], &p);
    assert!(
        matches!(err, TebdError::TooFewSites { n_sites: 0 }),
        "{err:?}"
    );
}

#[test]
fn rejects_a_bond_term_count_that_is_not_the_bond_count() {
    let (psi, terms) = fixture();
    let p = params(TimeStep::Real(0.1), NO_TRUNC);
    let err = rejected(&psi, &terms[..3], &p);
    assert!(
        matches!(
            err,
            TebdError::LengthMismatch {
                bonds: 4,
                bond_terms: 3
            }
        ),
        "{err:?}"
    );
    let mut extra = terms.clone();
    extra.push(terms[0].clone());
    let err = rejected(&psi, &extra, &p);
    assert!(
        matches!(
            err,
            TebdError::LengthMismatch {
                bonds: 4,
                bond_terms: 5
            }
        ),
        "{err:?}"
    );
}

#[test]
fn rejects_a_bond_term_of_the_wrong_shape() {
    let (psi, terms) = fixture();
    let p = params(TimeStep::Real(0.1), NO_TRUNC);

    let mut wrong_rank = terms.clone();
    wrong_rank[1] = terms[1].reshape_logical(vec![6, 6]);
    match rejected(&psi, &wrong_rank, &p) {
        TebdError::BondTermShape {
            bond: 1,
            expected: [3, 2, 3, 2],
            found,
        } => assert_eq!(found, [6, 6]),
        other => panic!("{other:?}"),
    }

    // Bond 0 joins dimensions (2, 3); the term of bond 1 is (3, 2).
    let mut swapped = terms.clone();
    swapped[0] = terms[1].clone();
    match rejected(&psi, &swapped, &p) {
        TebdError::BondTermShape {
            bond: 0,
            expected: [2, 3, 2, 3],
            found,
        } => assert_eq!(found, [3, 2, 3, 2]),
        other => panic!("{other:?}"),
    }
}

#[test]
fn rejects_a_step_that_is_not_finite_and_positive() {
    let (psi, terms) = fixture();
    for len in [f64::NAN, f64::INFINITY, 0.0, -0.1] {
        for time_step in [TimeStep::Real(len), TimeStep::Imaginary(len)] {
            let err = rejected(&psi, &terms, &params(time_step, NO_TRUNC));
            assert_eq!(invalid_detail(err), STEP_DETAIL, "{time_step:?}");
        }
    }
}

/// An `f64` step whose half overflows `f32`, and one whose half underflows
/// to zero in `f32` and would make every gate the identity.
#[test]
fn rejects_a_half_step_outside_the_real_scalar_range() {
    let psi: DenseMps<f32> = Mps::from_sites(vec![DenseTensor::ones(vec![1, 2, 1]); 3]);
    let terms = vec![DenseTensor::<f32>::zeros(vec![2, 2, 2, 2]); 2];
    for len in [1e300, 1e-100] {
        let err = rejected(&psi, &terms, &params(TimeStep::Imaginary(len), NO_TRUNC));
        assert_eq!(invalid_detail(err), HALF_STEP_DETAIL, "{len}");
    }
}

#[test]
fn rejects_truncation_params_the_svd_cannot_honor() {
    let (psi, terms) = fixture();
    let cases = [
        (Some(0), None, "trunc.chi_max must be > 0 if Some"),
        (
            None,
            Some(f64::NAN),
            "trunc.target_trunc_err must be finite",
        ),
        (
            None,
            Some(f64::INFINITY),
            "trunc.target_trunc_err must be finite",
        ),
        (
            None,
            Some(-0.1),
            "trunc.target_trunc_err must be non-negative",
        ),
    ];
    for (chi_max, target_trunc_err, detail) in cases {
        let trunc = TruncSvdParams {
            chi_max,
            target_trunc_err,
        };
        let err = rejected(&psi, &terms, &params(TimeStep::Real(0.1), trunc));
        assert_eq!(invalid_detail(err), detail);
    }
}

#[test]
fn rejects_real_time_on_a_real_scalar() {
    let psi = random_mps_unknown_f64(5, 2, 2, 3);
    let terms = vec![DenseTensor::<f64>::zeros(vec![2, 2, 2, 2]); 4];
    let err = rejected(&psi, &terms, &params(TimeStep::Real(0.1), NO_TRUNC));
    assert!(
        matches!(err, TebdError::GateConstruction { bond: 0, .. }),
        "{err:?}"
    );
}

#[test]
fn rejects_a_bond_term_that_is_not_hermitian() {
    let (psi, mut terms) = fixture();
    let skewed = terms[2].get([0, 0, 0, 1]) + C64::new(1.0, 0.0);
    terms[2].set([0, 0, 0, 1], skewed);
    for time_step in [TimeStep::Real(0.1), TimeStep::Imaginary(0.1)] {
        let err = rejected(&psi, &terms, &params(time_step, NO_TRUNC));
        assert!(
            matches!(err, TebdError::GateConstruction { bond: 2, .. }),
            "{time_step:?}: {err:?}"
        );
    }
}

/// Zero steps return no records and leave the state, but still validate.
#[test]
fn zero_steps_validate_and_leave_the_state() {
    let (psi, terms) = fixture();
    let zero_steps = |time_step| TebdParams {
        time_step,
        n_steps: 0,
        trunc: NO_TRUNC,
    };
    let mut out = psi.clone();
    let records = tebd(&mut out, &terms, &zero_steps(TimeStep::Real(0.1))).expect("valid call");
    assert!(records.is_empty());
    assert_unchanged(&out, &psi);
    let err = rejected(&psi, &terms, &zero_steps(TimeStep::Real(-0.1)));
    assert_eq!(invalid_detail(err), STEP_DETAIL);
}

#[test]
fn an_all_zero_state_breaks_down_in_the_first_step() {
    let (psi, terms) = fixture();
    let zero = Mps::from_sites(
        (0..psi.len())
            .map(|j| DenseTensor::<C64>::zeros(psi.site(j).shape().to_vec()))
            .collect(),
    );
    let mut out = zero.clone();
    let err = tebd(&mut out, &terms, &params(TimeStep::Real(0.1), NO_TRUNC))
        .expect_err("a zero state has no direction to normalize");
    assert!(
        matches!(err, TebdError::NormBreakdown { step: 0 }),
        "{err:?}"
    );
    // The step's gates ran, leaving the center on site 0 with zero norm; the
    // driver returned before normalizing it.
    assert_eq!(*out.canonical_form(), CanonicalForm::Mixed { center: 0 });
    assert_eq!(out.site(0).norm(), 0.0);
}
