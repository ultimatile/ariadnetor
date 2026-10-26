//! Tests for the Dense TEBD driver against dense references.
//!
//! The chain is contracted to its state vector, each bond term is embedded
//! into the full space, and the dense Trotter steps and propagators the
//! driver is compared against come from the general (Padé) matrix
//! exponential rather than the eigendecomposition the driver uses. Unless a
//! test says otherwise, the bond terms are random complex
//! Hermitian draws on neighboring physical dimensions that differ, so for a
//! generic draw neighboring terms do not commute and no term is symmetric under
//! transposition: a swapped leg, a reversed pass, or a missing half step
//! changes the result.

use algorithms_fixtures::dense_fixtures::{
    NO_TRUNC, random_bond_terms_c64, random_mps_c64, random_mps_unknown_f64, smallest_eigenvalue,
};
use ariadnetor_algorithms::tebd::{TebdParams, TebdStepRecord, TimeStep, tebd};
use ariadnetor_core::Scalar;
use ariadnetor_linalg::{contract, expm_antihermitian_with_backend, expm_with_backend, tensordot};
use ariadnetor_mps::{
    CanonicalForm, GateCenter, Mps, SpinHalf, TensorChain, TruncSvdParams, TruncateParams, inner,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor, add_all};
use num_complex::Complex;

type C64 = Complex<f64>;
type DenseMps<T> = Mps<DenseStorage<T>, DenseLayout>;

/// Physical dimensions of the 5-site fixture; neighbors always differ.
const PHYS: [usize; 5] = [2, 3, 2, 3, 2];
/// Bond dimensions of the 5-site fixture, boundaries included.
const BONDS: [usize; 6] = [1, 2, 4, 4, 3, 1];

/// The bond terms and initial state most tests share.
fn fixture() -> (Vec<DenseTensor<C64>>, DenseMps<C64>) {
    (
        random_bond_terms_c64(&PHYS, 100),
        random_mps_c64(&PHYS, &BONDS, 7),
    )
}

fn cap(chi_max: usize) -> TruncSvdParams {
    TruncSvdParams {
        chi_max: Some(chi_max),
        target_trunc_err: None,
    }
}

fn run<T: Scalar>(
    psi: &mut DenseMps<T>,
    terms: &[DenseTensor<T>],
    time_step: TimeStep,
    n_steps: usize,
    trunc: TruncSvdParams,
) -> Vec<TebdStepRecord<T::Real>> {
    let params = TebdParams {
        time_step,
        n_steps,
        trunc,
    };
    tebd(psi, terms, &params).expect("tebd")
}

/// State vector of the chain, flattened row-major over `(d_0, …, d_{n-1})`.
fn state_vector<T: Scalar>(psi: &DenseMps<T>) -> DenseTensor<T> {
    let backend = NativeBackend::new();
    let mut acc = psi.site(0).clone();
    for j in 1..psi.len() {
        let last = acc.rank() - 1;
        acc = tensordot(&backend, &acc, psi.site(j), &[last], &[0]).expect("chain contraction");
    }
    let dim = acc.shape().iter().product();
    acc.reshape_logical(vec![dim])
}

/// `term` at `bond` embedded into the full space as `1 ⊗ h ⊗ 1`, a matrix
/// over the row-major flattening of `(d_0, …, d_{n-1})`.
fn embed<T: Scalar>(term: &DenseTensor<T>, phys: &[usize], bond: usize) -> DenseTensor<T> {
    let left: usize = phys[..bond].iter().product();
    let right: usize = phys[bond + 2..].iter().product();
    let pair = phys[bond] * phys[bond + 1];
    let h = term.reshape_logical(vec![pair, pair]);
    let dim = left * pair * right;
    let mut out = DenseTensor::<T>::zeros(vec![dim, dim]);
    for l in 0..left {
        for r in 0..right {
            for a in 0..pair {
                for b in 0..pair {
                    out.set(
                        [(l * pair + a) * right + r, (l * pair + b) * right + r],
                        h.get([a, b]),
                    );
                }
            }
        }
    }
    out
}

/// `H = Σ_i h_i` as a dense matrix.
fn hamiltonian<T: Scalar>(terms: &[DenseTensor<T>], phys: &[usize]) -> DenseTensor<T> {
    let embedded: Vec<_> = terms
        .iter()
        .enumerate()
        .map(|(bond, term)| embed(term, phys, bond))
        .collect();
    add_all(&embedded.iter().collect::<Vec<_>>()).expect("terms share one shape and order")
}

fn matmul<T: Scalar>(a: &DenseTensor<T>, b: &DenseTensor<T>) -> DenseTensor<T> {
    contract(&NativeBackend::new(), a, b, "ij,jk->ik").expect("matmul")
}

fn matvec<T: Scalar>(a: &DenseTensor<T>, v: &DenseTensor<T>) -> DenseTensor<T> {
    contract(&NativeBackend::new(), a, v, "ij,j->i").expect("matvec")
}

/// `exp(-w A)` for `w` the half step: `i dt / 2` in real time, `tau / 2` in
/// imaginary time.
fn half_step_exp<T: Scalar<Real = f64>>(a: &DenseTensor<T>, time_step: TimeStep) -> DenseTensor<T> {
    let factor = match time_step {
        TimeStep::Real(dt) => T::from_real_imag(0.0, -dt / 2.0),
        TimeStep::Imaginary(tau) => T::from_real_imag(-tau / 2.0, 0.0),
    };
    expm_with_backend(&NativeBackend::new(), &a.scaled(factor), 1).expect("expm")
}

/// One step as a dense matrix, `S = G_0 ⋯ G_{n-2} · G_{n-2} ⋯ G_0` with
/// `G_i = exp(-w h_i)`: the rightmost factor acts first.
fn dense_step<T: Scalar<Real = f64>>(
    terms: &[DenseTensor<T>],
    phys: &[usize],
    time_step: TimeStep,
) -> DenseTensor<T> {
    let gates: Vec<_> = terms
        .iter()
        .enumerate()
        .map(|(bond, term)| half_step_exp(&embed(term, phys, bond), time_step))
        .collect();
    let mut s = gates[0].clone();
    for g in gates.iter().skip(1).chain(gates.iter().rev()) {
        s = matmul(&s, g);
    }
    s
}

/// `v` scaled to unit norm, computed apart from `Tensor::normalize`, which
/// the driver itself uses.
fn normalized<T: Scalar<Real = f64>>(v: &DenseTensor<T>) -> DenseTensor<T> {
    v.scaled(T::from_real_imag(1.0 / v.norm(), 0.0))
}

/// `S^n_steps psi0`, normalized.
fn dense_evolved<T: Scalar<Real = f64>>(
    psi0: &DenseMps<T>,
    terms: &[DenseTensor<T>],
    phys: &[usize],
    time_step: TimeStep,
    n_steps: usize,
) -> DenseTensor<T> {
    let s = dense_step(terms, phys, time_step);
    let mut v = state_vector(psi0);
    for _ in 0..n_steps {
        v = matvec(&s, &v);
    }
    normalized(&v)
}

/// `‖a − b‖` over two vectors, through logical indices.
fn distance<T: Scalar<Real = f64>>(a: &DenseTensor<T>, b: &DenseTensor<T>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    (0..a.shape()[0])
        .map(|i| (a.get([i]) + b.get([i]).scale_real(-1.0)).abs().powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Without truncation, three steps reproduce the dense `S^3 psi0`,
/// normalized, with consecutive step indices and no truncation error.
fn check_schedule(phys: &[usize], bonds: &[usize], time_step: TimeStep) {
    let terms = random_bond_terms_c64(phys, 100);
    let psi0 = random_mps_c64(phys, bonds, 7);
    let expected = dense_evolved(&psi0, &terms, phys, time_step, 3);
    let mut psi = psi0.clone();
    let records = run(&mut psi, &terms, time_step, 3, NO_TRUNC);
    let err = distance(&state_vector(&psi), &expected);
    assert!(err < 1e-12, "{time_step:?}: state error {err}");
    assert_eq!(records.len(), 3);
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record.step, i);
        assert_eq!(record.trunc_err_sum, 0.0, "step {i}");
    }
    assert_eq!(records[2].max_bond, psi.max_bond_dim());
}

#[test]
fn real_time_schedule_matches_the_dense_product() {
    check_schedule(&PHYS, &BONDS, TimeStep::Real(0.1));
}

#[test]
fn imaginary_time_schedule_matches_the_dense_product() {
    check_schedule(&PHYS, &BONDS, TimeStep::Imaginary(0.1));
}

#[test]
fn two_site_chain_matches_the_dense_product() {
    check_schedule(&PHYS[..2], &[1, 2, 1], TimeStep::Real(0.1));
}

/// Halving the step at fixed total time divides the error by about 4.
#[test]
fn error_is_second_order_in_the_step() {
    let (terms, psi0) = fixture();
    let h = hamiltonian(&terms, &PHYS);
    let exact_propagator =
        expm_with_backend(&NativeBackend::new(), &h.scaled(C64::new(0.0, -1.0)), 1).expect("expm");
    let exact = normalized(&matvec(&exact_propagator, &state_vector(&psi0)));
    let error = |n_steps: usize| {
        let mut psi = psi0.clone();
        let dt = 1.0 / n_steps as f64;
        run(&mut psi, &terms, TimeStep::Real(dt), n_steps, NO_TRUNC);
        distance(&state_vector(&psi), &exact)
    };
    let (coarse, fine) = (error(20), error(40));
    let ratio = coarse / fine;
    assert!(
        (3.5..=4.5).contains(&ratio),
        "errors {coarse} and {fine}, ratio {ratio}"
    );
}

/// Without truncation in real time, the first step keeps the input norm
/// and every later step starts from the renormalized state.
#[test]
fn records_the_norm_before_renormalization() {
    let (terms, mut psi) = fixture();
    let norm0 = 3.0;
    let rescale = norm0 / state_vector(&psi).norm();
    psi.site_mut(0).scale(C64::new(rescale, 0.0));
    let records = run(&mut psi, &terms, TimeStep::Real(0.1), 3, NO_TRUNC);
    assert!((records[0].norm - norm0).abs() < 1e-12 * norm0);
    for record in &records[1..] {
        assert!((record.norm - 1.0).abs() < 1e-12, "{record:?}");
    }
}

/// With `chi_max = 2`, one real-time step lands within `trunc_err_sum` of
/// the untruncated step: `‖S psi0 − norm · psi‖ ≤ trunc_err_sum`.
#[test]
fn truncation_error_bounds_the_state_error() {
    let (terms, psi0) = fixture();
    let time_step = TimeStep::Real(0.1);
    let untruncated = matvec(&dense_step(&terms, &PHYS, time_step), &state_vector(&psi0));
    let mut psi = psi0.clone();
    let record = run(&mut psi, &terms, time_step, 1, cap(2)).remove(0);
    assert!(
        record.trunc_err_sum > 1e-3,
        "fixture must actually truncate, got {}",
        record.trunc_err_sum
    );
    assert!(record.max_bond <= 2, "{record:?}");
    assert_eq!(record.max_bond, psi.max_bond_dim());
    let rescaled = state_vector(&psi).scaled(C64::new(record.norm, 0.0));
    let err = distance(&rescaled, &untruncated);
    assert!(
        err <= record.trunc_err_sum + 1e-12,
        "state error {err} exceeds trunc_err_sum {}",
        record.trunc_err_sum
    );
}

/// The truncation target is relative to the state norm: the same state at
/// 1000 times the norm keeps the same bonds and the same normalized result,
/// with every discarded weight scaled by the same factor.
#[test]
fn truncation_target_is_relative_to_the_state_norm() {
    let (terms, small) = fixture();
    let mut large = small.clone();
    large.site_mut(0).scale(C64::new(1000.0, 0.0));
    let time_step = TimeStep::Real(0.1);
    let target = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(1e-2),
    };

    let mut untruncated = small.clone();
    run(&mut untruncated, &terms, time_step, 1, NO_TRUNC);
    let (mut a, mut b) = (small, large);
    let ra = run(&mut a, &terms, time_step, 1, target.clone()).remove(0);
    let rb = run(&mut b, &terms, time_step, 1, target).remove(0);
    assert!(
        a.bond_dims() != untruncated.bond_dims(),
        "fixture must actually truncate: {:?}",
        a.bond_dims()
    );
    assert_eq!(a.bond_dims(), b.bond_dims());
    assert!(
        (rb.trunc_err_sum - 1000.0 * ra.trunc_err_sum).abs() < 1e-9 * rb.trunc_err_sum,
        "{ra:?} vs {rb:?}"
    );
    assert!(distance(&state_vector(&a), &state_vector(&b)) < 1e-12);
}

/// A truncating two-step call records the same steps as two one-step calls:
/// each record covers its own step's gates only, and the second step's
/// bound holds against the untruncated step from the first step's result.
#[test]
fn each_record_covers_its_own_step() {
    let (terms, psi0) = fixture();
    let time_step = TimeStep::Real(0.1);

    let mut stepwise = psi0.clone();
    let first = run(&mut stepwise, &terms, time_step, 1, cap(2)).remove(0);
    let untruncated = matvec(
        &dense_step(&terms, &PHYS, time_step),
        &state_vector(&stepwise),
    );
    let second = run(&mut stepwise, &terms, time_step, 1, cap(2)).remove(0);

    let mut psi = psi0.clone();
    let records = run(&mut psi, &terms, time_step, 2, cap(2));
    assert!(second.trunc_err_sum > 1e-3, "second step must truncate too");
    for (record, expected) in records.iter().zip([&first, &second]) {
        assert!(
            (record.trunc_err_sum - expected.trunc_err_sum).abs() < 1e-12
                && (record.norm - expected.norm).abs() < 1e-12,
            "{record:?} vs {expected:?}"
        );
    }
    assert_eq!(records[1].step, 1);
    assert!(distance(&state_vector(&psi), &state_vector(&stepwise)) < 1e-12);

    let rescaled = state_vector(&psi).scaled(C64::new(records[1].norm, 0.0));
    let err = distance(&rescaled, &untruncated);
    assert!(
        err <= records[1].trunc_err_sum + 1e-12,
        "state error {err} exceeds trunc_err_sum {}",
        records[1].trunc_err_sum
    );
}

/// A 2-site `f32` chain with every dimension 1 and bond term `h = -c`:
/// every tensor is a scalar, so each imaginary-time step multiplies the norm
/// by `exp(c tau)` up to rounding. A step whose factor underflows `f32` leaves a
/// zero state.
#[test]
fn f32_imaginary_time_tracks_scalar_growth_until_it_underflows() {
    let scalar_run = |c: f32, tau: f64| {
        let mut psi: DenseMps<f32> = Mps::from_sites(vec![DenseTensor::ones(vec![1, 1, 1]); 2]);
        let mut h = DenseTensor::<f32>::zeros(vec![1, 1, 1, 1]);
        h.set([0, 0, 0, 0], -c);
        let params = TebdParams {
            time_step: TimeStep::Imaginary(tau),
            n_steps: 2,
            trunc: NO_TRUNC,
        };
        tebd(&mut psi, &[h], &params)
    };

    let records = scalar_run(1.0, 0.5).expect("finite growth");
    for record in &records {
        let expected = 0.5_f32.exp();
        assert!(
            (record.norm - expected).abs() < 1e-6 * expected,
            "{record:?}"
        );
    }

    let err = scalar_run(-200.0, 1.0).expect_err("exp(-200) underflows f32");
    assert!(
        matches!(
            err,
            ariadnetor_algorithms::tebd::TebdError::NormBreakdown { step: 0 }
        ),
        "{err:?}"
    );
}

/// `S_i · S_{i+1} = Sz Sz + (S+ S- + S- S+) / 2` with legs
/// `(out_i, out_i+1, in_i, in_i+1)`.
fn heisenberg_term() -> DenseTensor<f64> {
    let s = SpinHalf;
    let products: [(DenseTensor<f64>, DenseTensor<f64>, f64); 3] = [
        (s.sz(), s.sz(), 1.0),
        (s.sp(), s.sm(), 0.5),
        (s.sm(), s.sp(), 0.5),
    ];
    let mut term = DenseTensor::<f64>::zeros(vec![2, 2, 2, 2]);
    for (a, b, coeff) in &products {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        let v = term.get([i, j, k, l]) + coeff * a.get([i, k]) * b.get([j, l]);
                        term.set([i, j, k, l], v);
                    }
                }
            }
        }
    }
    term
}

/// Imaginary time on the open 6-site spin-1/2 Heisenberg chain reaches the
/// exact ground-state energy, and `-ln(norm) / tau` estimates it.
#[test]
fn imaginary_time_converges_to_the_heisenberg_ground_state() {
    let n = 6;
    let phys = [2; 6];
    let terms = vec![heisenberg_term(); n - 1];
    let h = hamiltonian(&terms, &phys);
    let e0 = smallest_eigenvalue(&h);
    assert!((e0 + 2.4935771339).abs() < 1e-9, "fixture E0 = {e0}");

    let tau = 0.1;
    let mut psi = random_mps_unknown_f64(n, 2, 2, 5);
    let records = run(&mut psi, &terms, TimeStep::Imaginary(tau), 400, NO_TRUNC);
    let v = state_vector(&psi);
    let hv = matvec(&h, &v);
    let energy: f64 = (0..v.shape()[0]).map(|i| v.get([i]) * hv.get([i])).sum();
    assert!((energy - e0).abs() < 1e-6, "<H> = {energy}, E0 = {e0}");
    let estimate = -records[399].norm.ln() / tau;
    assert!(
        (estimate - e0).abs() < 1e-3,
        "-ln(norm)/tau = {estimate}, E0 = {e0}"
    );
}

/// `base` canonicalized to `center`, with that site scaled to unit norm and
/// tagged `form`, which must be true of the result.
fn prepared(base: &DenseMps<C64>, center: usize, form: CanonicalForm) -> DenseMps<C64> {
    let mut psi = base.clone();
    psi.canonicalize(&NativeBackend::new(), center);
    let nrm = psi.site(center).norm();
    psi.site_mut(center).scale(C64::new(1.0 / nrm, 0.0));
    psi.set_canonical_form(form);
    psi
}

/// Every input form gives the same result, which ends in `Mixed { center: 0 }`.
#[test]
fn result_is_independent_of_the_input_form() {
    let n = PHYS.len();
    let terms = random_bond_terms_c64(&PHYS, 100);
    let base = random_mps_c64(&PHYS, &BONDS, 13);
    let time_step = TimeStep::Real(0.1);
    let expected = dense_evolved(&base, &terms, &PHYS, time_step, 3);
    let inputs = [
        ("Unknown", base.clone()),
        ("Left", prepared(&base, n - 1, CanonicalForm::Left)),
        ("Right", prepared(&base, 0, CanonicalForm::Right)),
        (
            "Mixed{3}",
            prepared(&base, 3, CanonicalForm::Mixed { center: 3 }),
        ),
        (
            "Partial{1,4}",
            prepared(
                &base,
                2,
                CanonicalForm::Partial {
                    left_end: 1,
                    right_start: 4,
                },
            ),
        ),
    ];
    for (label, mut psi) in inputs {
        run(&mut psi, &terms, time_step, 3, NO_TRUNC);
        let err = distance(&state_vector(&psi), &expected);
        assert!(err < 1e-12, "{label}: state error {err}");
        assert_eq!(
            *psi.canonical_form(),
            CanonicalForm::Mixed { center: 0 },
            "{label}"
        );
    }
}

/// `norm` and `truncate` trust the form the driver leaves, and a further
/// gate applies correctly from it.
#[test]
fn result_works_with_the_other_mps_operations() {
    let backend = NativeBackend::new();
    let terms = random_bond_terms_c64(&PHYS, 100);
    let mut psi = random_mps_c64(&PHYS, &BONDS, 23);
    run(&mut psi, &terms, TimeStep::Real(0.1), 3, NO_TRUNC);

    let nrm = psi.norm::<C64, _>(&backend);
    let full: C64 = inner(&backend, &psi, &psi);
    assert!((nrm - 1.0).abs() < 1e-12, "norm {nrm}");
    assert!(
        (full.re.sqrt() - 1.0).abs() < 1e-12 && full.im.abs() < 1e-12,
        "inner {full}"
    );

    let mut truncated = psi.clone();
    let res = truncated.truncate(&backend, &TruncateParams::from(cap(2)));
    let actual = distance(&state_vector(&truncated), &state_vector(&psi));
    assert!(res.error > 1e-3, "fixture must actually truncate");
    assert!(
        (res.error - actual).abs() < 1e-10 * (1.0 + actual),
        "returned {}, state error {actual}",
        res.error
    );

    let bond = 2;
    let pair = PHYS[bond] * PHYS[bond + 1];
    let anti = terms[bond]
        .reshape_logical(vec![pair, pair])
        .scaled(C64::new(0.0, -0.3));
    let gate = expm_antihermitian_with_backend(&backend, &anti, 1)
        .expect("expm")
        .reshape_logical(vec![PHYS[bond], PHYS[bond + 1], PHYS[bond], PHYS[bond + 1]]);
    let expected = matvec(&embed(&gate, &PHYS, bond), &state_vector(&psi));
    psi.apply_two_site_gate(&backend, bond, &gate, &NO_TRUNC, GateCenter::Right);
    let err = distance(&state_vector(&psi), &expected);
    assert!(err < 1e-12, "gate after tebd: state error {err}");
}
