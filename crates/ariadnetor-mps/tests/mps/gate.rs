//! Tests for `Mps::apply_two_site_gate`.
//!
//! References are dense: the chain is contracted to its full state tensor
//! and the gate is applied to the two physical axes directly. Gates are
//! deliberately not transpose-symmetric and the pair's physical dimensions
//! differ, so swapped input / output legs or swapped sites change the
//! result instead of cancelling out.

use ariadnetor_core::{Complex, Scalar};
use ariadnetor_linalg::{
    contract, diagonal_scale, expm_antihermitian_with_backend, expm_with_backend,
    permute_with_backend, trunc_svd,
};
use ariadnetor_mps::{
    CanonicalForm, GateCenter, Mpo, Mps, TensorChain, TruncSvdParams, TruncateParams, apply, inner,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor, MemoryOrder};
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::helpers::{densify, relative_frobenius};

type DenseMps<T> = Mps<DenseStorage<T>, DenseLayout>;

/// Physical dimensions of the 5-site fixture; neighbors always differ.
const PHYS: [usize; 5] = [2, 3, 2, 3, 2];
/// Bond dimensions of the 5-site fixture, boundaries included.
const BONDS: [usize; 6] = [1, 2, 4, 4, 3, 1];

const NO_TRUNC: TruncSvdParams = TruncSvdParams {
    chi_max: None,
    target_trunc_err: None,
};

fn random_scalar<T: Scalar<Real = f64>>(rng: &mut StdRng) -> T {
    let re = rand::RngExt::random_range(rng, -0.5_f64..0.5);
    let im = rand::RngExt::random_range(rng, -0.5_f64..0.5);
    T::from_real_imag(re, im)
}

/// Dense tensor with independent random entries, filled through logical
/// indices so the construction names no memory order.
fn random_tensor<T: Scalar<Real = f64>>(shape: Vec<usize>, rng: &mut StdRng) -> DenseTensor<T> {
    let mut t = DenseTensor::<T>::zeros(shape.clone());
    let mut idx = vec![0; shape.len()];
    for _ in 0..shape.iter().product::<usize>() {
        t.set(&idx, random_scalar(rng));
        for axis in (0..shape.len()).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    t
}

fn random_mps<T: Scalar<Real = f64>>(phys: &[usize], bonds: &[usize], seed: u64) -> DenseMps<T> {
    let mut rng = StdRng::seed_from_u64(seed);
    let sites = (0..phys.len())
        .map(|j| random_tensor(vec![bonds[j], phys[j], bonds[j + 1]], &mut rng))
        .collect();
    Mps::from_sites(sites)
}

/// `exp(-0.3 h)` for a random real `h` with no symmetry: real and neither
/// unitary nor transpose-symmetric.
fn real_gate(d_l: usize, d_r: usize, seed: u64) -> DenseTensor<f64> {
    let dim = d_l * d_r;
    let mut rng = StdRng::seed_from_u64(seed);
    let h: DenseTensor<f64> = random_tensor(vec![dim, dim], &mut rng);
    let g = expm_with_backend(&NativeBackend::new(), &h.scaled(-0.3), 1).expect("expm");
    g.reshape_logical(vec![d_l, d_r, d_l, d_r])
}

/// `exp(-0.3 i h)` for a random complex Hermitian `h`: unitary but not
/// transpose-symmetric.
fn unitary_gate(d_l: usize, d_r: usize, seed: u64) -> DenseTensor<Complex<f64>> {
    let dim = d_l * d_r;
    let mut rng = StdRng::seed_from_u64(seed);
    let a: DenseTensor<Complex<f64>> = random_tensor(vec![dim, dim], &mut rng);
    let mut anti = DenseTensor::<Complex<f64>>::zeros(vec![dim, dim]);
    for i in 0..dim {
        for j in 0..dim {
            // -0.3 i (a + a^H) / 2 is anti-Hermitian.
            let herm = (a.get([i, j]) + a.get([j, i]).conj()) * 0.5;
            anti.set([i, j], herm * Complex::new(0.0, -0.3));
        }
    }
    let g = expm_antihermitian_with_backend(&NativeBackend::new(), &anti, 1).expect("expm");
    g.reshape_logical(vec![d_l, d_r, d_l, d_r])
}

/// The full state tensor `(1, d_0, ..., d_{N-1}, 1)` in column-major order.
fn dense_state<T: Scalar>(mps: &DenseMps<T>) -> DenseTensor<T> {
    densify(&NativeBackend::new(), mps).reordered(MemoryOrder::ColumnMajor)
}

/// Apply `gate` to physical axes `site + 1`, `site + 2` of a dense state.
fn apply_gate_dense<T: Scalar>(
    state: &DenseTensor<T>,
    gate: &DenseTensor<T>,
    site: usize,
) -> DenseTensor<T> {
    let labels: Vec<char> = ('a'..='w').take(state.rank()).collect();
    let input: String = labels.iter().collect();
    let output: String = labels
        .iter()
        .enumerate()
        .map(|(k, &c)| match k {
            k if k == site + 1 => 'y',
            k if k == site + 2 => 'z',
            _ => c,
        })
        .collect();
    let spec = format!(
        "yz{}{},{input}->{output}",
        labels[site + 1],
        labels[site + 2]
    );
    contract(&NativeBackend::new(), gate, state, &spec)
        .expect("dense gate application")
        .reordered(MemoryOrder::ColumnMajor)
}

/// Absolute Frobenius distance between two column-major tensors.
fn distance<T: Scalar>(a: &DenseTensor<T>, e: &DenseTensor<T>) -> f64 {
    relative_frobenius(a, e) * frobenius(e)
}

fn frobenius<T: Scalar>(t: &DenseTensor<T>) -> f64 {
    use num_traits::ToPrimitive;
    t.data_slice()
        .iter()
        .map(|v| v.abs().to_f64().expect("fits f64").powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Whether `site` is left-isometric (`left`) or right-isometric, with the
/// conjugate taken so the check holds for complex sites too.
fn is_isometric<T: Scalar<Real = f64>>(site: &DenseTensor<T>, left: bool) -> bool {
    let spec = if left { "abc,abd->cd" } else { "cab,dab->cd" };
    let gram = contract(&NativeBackend::new(), &site.conj(), site, spec).expect("gram");
    let k = gram.shape()[0];
    (0..k).all(|i| {
        (0..k).all(|j| {
            let expected = if i == j { 1.0 } else { 0.0 };
            (gram.get([i, j]) + T::from_real_imag(-expected, 0.0)).abs() < 1e-10
        })
    })
}

fn centered<T: Scalar<Real = f64>>(mut mps: DenseMps<T>, center: usize) -> DenseMps<T> {
    mps.canonicalize(&NativeBackend::new(), center);
    mps
}

/// Without truncation, every bond and both center choices reproduce the
/// dense application exactly, with zero error and the requested center.
fn check_exact<T: Scalar<Real = f64>>(gate_for: impl Fn(usize, usize, u64) -> DenseTensor<T>) {
    let backend = NativeBackend::new();
    for site in 0..PHYS.len() - 1 {
        let gate = gate_for(PHYS[site], PHYS[site + 1], 100 + site as u64);
        let psi = centered(random_mps::<T>(&PHYS, &BONDS, 7), site);
        let expected = apply_gate_dense(&dense_state(&psi), &gate, site);
        for (center, center_site) in [(GateCenter::Left, site), (GateCenter::Right, site + 1)] {
            let mut out = psi.clone();
            let res = out.apply_two_site_gate(&backend, site, &gate, &NO_TRUNC, center);
            assert_eq!(res.error, 0.0, "site {site}: untruncated error");
            let rel = relative_frobenius(&dense_state(&out), &expected);
            assert!(rel < 1e-12, "site {site} {center:?}: relative error {rel}");
            assert_eq!(
                *out.canonical_form(),
                CanonicalForm::Mixed {
                    center: center_site
                }
            );
        }
    }
}

#[test]
fn untruncated_application_matches_dense_f64() {
    check_exact(real_gate);
}

#[test]
fn untruncated_application_matches_dense_c64() {
    check_exact(unitary_gate);
}

/// With truncation, the returned error is the state error, and the pair's
/// non-center site is isometric on the side `center` leaves bare.
fn check_truncated<T: Scalar<Real = f64>>(
    label: &str,
    gate: &DenseTensor<T>,
    psi: &DenseMps<T>,
    site: usize,
) {
    let backend = NativeBackend::new();
    let expected = apply_gate_dense(&dense_state(psi), gate, site);
    let svd = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    for center in [GateCenter::Left, GateCenter::Right] {
        let mut out = psi.clone();
        let err = out
            .apply_two_site_gate(&backend, site, gate, &svd, center)
            .error;
        let actual = distance(&dense_state(&out), &expected);
        assert!(
            err > 1e-3,
            "{label}: fixture must actually truncate, got {err}"
        );
        assert!(
            (err - actual).abs() < 1e-10 * (1.0 + actual),
            "{label} {center:?}: returned {err}, state error {actual}"
        );
        assert_eq!(out.bond_dim(site), 1, "{label}");
        let (form_center, bare, left) = match center {
            GateCenter::Left => (site, site + 1, false),
            GateCenter::Right => (site + 1, site, true),
        };
        assert_eq!(
            *out.canonical_form(),
            CanonicalForm::Mixed {
                center: form_center
            },
            "{label} {center:?}"
        );
        assert!(
            is_isometric(out.site(bare), left),
            "{label} {center:?}: site {bare}"
        );
    }
}

#[test]
fn truncation_error_equals_state_error() {
    let psi = centered(random_mps::<f64>(&PHYS, &BONDS, 11), 2);
    check_truncated("f64", &real_gate(PHYS[1], PHYS[2], 3), &psi, 1);
    let psi = centered(random_mps::<Complex<f64>>(&PHYS, &BONDS, 11), 2);
    check_truncated("c64", &unitary_gate(PHYS[1], PHYS[2], 3), &psi, 1);
}

/// Forms whose center is not on the pair are canonicalized first, so the
/// error identity still holds; forms whose center is on the pair are used
/// as they are. Off-by-one centers on either side fall in the first group.
#[test]
fn input_forms_with_center_off_the_pair_are_canonicalized() {
    let n = PHYS.len();
    let gate = unitary_gate(PHYS[2], PHYS[3], 5);
    let base = random_mps::<Complex<f64>>(&PHYS, &BONDS, 13);
    let mixed = |c| centered(base.clone(), c);
    let mut partial = mixed(2);
    partial.set_canonical_form(CanonicalForm::Partial {
        left_end: 2,
        right_start: 4,
    });
    let inputs = [
        ("Unknown", base.clone()),
        ("Mixed{0}", mixed(0)),
        ("Mixed{1}", mixed(1)),
        ("Mixed{2}", mixed(2)),
        ("Mixed{3}", mixed(3)),
        ("Mixed{4}", mixed(4)),
        ("Partial{2,4}", partial),
    ];
    for (label, psi) in &inputs {
        check_truncated(label, &gate, psi, 2);
    }

    // `Left` / `Right` are normalized forms; build them from a normalized
    // state and test both the on-pair and the off-pair boundary site.
    let normalized = |c: usize| {
        let mut psi = mixed(c);
        let nrm = psi.norm::<Complex<f64>, _>(&NativeBackend::new());
        let scaled = psi.site(c).scaled(Complex::new(1.0 / nrm, 0.0));
        *psi.site_mut(c) = scaled;
        psi
    };
    let mut left = normalized(n - 1);
    left.set_canonical_form(CanonicalForm::Left);
    let mut right = normalized(0);
    right.set_canonical_form(CanonicalForm::Right);
    for site in [0, n - 2] {
        let g = unitary_gate(PHYS[site], PHYS[site + 1], 9);
        check_truncated(&format!("Left, site {site}"), &g, &left, site);
        check_truncated(&format!("Right, site {site}"), &g, &right, site);
    }
}

/// The gate split into a two-site MPO in the `(chi_L, ket, bra, chi_R)`
/// convention and applied with `apply` reaches the same state, so the
/// gate's leg convention agrees with the MPO route.
#[test]
fn matches_the_two_site_mpo_route() {
    let backend = NativeBackend::new();
    let site = 1;
    let (d_l, d_r) = (PHYS[site], PHYS[site + 1]);
    let gate = unitary_gate(d_l, d_r, 21);
    let psi = random_mps::<Complex<f64>>(&PHYS, &BONDS, 17);

    // (out_l, out_r, in_l, in_r) -> (in_l, out_l, in_r, out_r), then an
    // operator-Schmidt split between the two sites.
    let grouped = permute_with_backend(&backend, &gate, &[2, 0, 3, 1]).expect("permute");
    let (u, s, vt, _) = trunc_svd(&backend, &grouped, 2, &NO_TRUNC).expect("svd");
    let k = s.shape()[0];
    let w_l = diagonal_scale(&backend, &u, s.data_slice(), 1)
        .expect("scale")
        .reshape_logical(vec![1, d_l, d_l, k]);
    let w_r = vt.reshape_logical(vec![k, d_r, d_r, 1]);
    let mut sites = Mpo::<DenseStorage<Complex<f64>>, DenseLayout>::identity(&PHYS)
        .sites()
        .to_vec();
    sites[site] = w_l;
    sites[site + 1] = w_r;
    let mpo = Mpo::from_sites(sites);

    let via_mpo = apply(&backend, &mpo, &psi, None);
    let mut via_gate = psi.clone();
    via_gate.apply_two_site_gate(&backend, site, &gate, &NO_TRUNC, GateCenter::Right);
    let rel = relative_frobenius(&dense_state(&via_gate), &dense_state(&via_mpo));
    assert!(rel < 1e-12, "relative error {rel}");
}

/// `norm` reads only the center tensor the gate tagged, and `truncate`
/// starts from that tag; both must agree with dense evaluation.
#[test]
fn norm_and_truncate_trust_the_resulting_form() {
    let backend = NativeBackend::new();
    let gate = unitary_gate(PHYS[2], PHYS[3], 31);
    let mut psi = centered(random_mps::<Complex<f64>>(&PHYS, &BONDS, 23), 0);
    let svd = TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    };
    psi.apply_two_site_gate(&backend, 2, &gate, &svd, GateCenter::Right);

    let full: Complex<f64> = inner(&backend, &psi, &psi);
    let nrm = psi.norm::<Complex<f64>, _>(&backend);
    assert!(
        (nrm - full.re.sqrt()).abs() < 1e-10 * nrm,
        "{nrm} vs {}",
        full.re.sqrt()
    );

    let before = dense_state(&psi);
    let res = psi.truncate(&backend, &TruncateParams::from(svd));
    let actual = distance(&dense_state(&psi), &before);
    assert!(res.error > 1e-3, "fixture must actually truncate");
    assert!((res.error - actual).abs() < 1e-10 * (1.0 + actual));
}

#[test]
fn unitary_gate_without_truncation_preserves_norm() {
    let backend = NativeBackend::new();
    let mut psi = centered(random_mps::<Complex<f64>>(&PHYS, &BONDS, 29), 3);
    let nrm = psi.norm::<Complex<f64>, _>(&backend);
    let scaled = psi.site(3).scaled(Complex::new(1.0 / nrm, 0.0));
    *psi.site_mut(3) = scaled;
    psi.set_canonical_form(CanonicalForm::Mixed { center: 3 });
    for site in 0..PHYS.len() - 1 {
        let gate = unitary_gate(PHYS[site], PHYS[site + 1], 40 + site as u64);
        psi.apply_two_site_gate(&backend, site, &gate, &NO_TRUNC, GateCenter::Right);
    }
    let full: Complex<f64> = inner(&backend, &psi, &psi);
    assert!(
        (full.re - 1.0).abs() < 1e-12 && full.im.abs() < 1e-12,
        "{full}"
    );
}

fn panic_fixture() -> DenseMps<f64> {
    random_mps::<f64>(&PHYS, &BONDS, 1)
}

#[test]
#[should_panic(expected = "apply_two_site_gate: site 4 has no right neighbor")]
fn panics_on_last_site() {
    let gate = real_gate(2, 2, 1);
    panic_fixture().apply_two_site_gate(
        &NativeBackend::new(),
        4,
        &gate,
        &NO_TRUNC,
        GateCenter::Left,
    );
}

#[test]
#[should_panic(expected = "has no right neighbor")]
fn panics_on_overflowing_site() {
    let gate = real_gate(2, 2, 1);
    panic_fixture().apply_two_site_gate(
        &NativeBackend::new(),
        usize::MAX,
        &gate,
        &NO_TRUNC,
        GateCenter::Left,
    );
}

#[test]
#[should_panic(expected = "apply_two_site_gate: gate shape must be")]
fn panics_on_wrong_rank_gate() {
    let gate = real_gate(2, 3, 1).reshape_logical(vec![6, 6]);
    panic_fixture().apply_two_site_gate(
        &NativeBackend::new(),
        0,
        &gate,
        &NO_TRUNC,
        GateCenter::Left,
    );
}

#[test]
#[should_panic(expected = "apply_two_site_gate: gate shape must be")]
fn panics_on_swapped_gate_dimensions() {
    // Sites 0 and 1 have dimensions (2, 3); a (3, 2) gate is rejected.
    let gate = real_gate(3, 2, 1);
    panic_fixture().apply_two_site_gate(
        &NativeBackend::new(),
        0,
        &gate,
        &NO_TRUNC,
        GateCenter::Left,
    );
}

#[test]
#[should_panic(expected = "apply_two_site_gate: chi_max must be at least 1")]
fn panics_on_zero_chi_max() {
    let gate = real_gate(2, 3, 1);
    let svd = TruncSvdParams {
        chi_max: Some(0),
        target_trunc_err: None,
    };
    panic_fixture().apply_two_site_gate(&NativeBackend::new(), 0, &gate, &svd, GateCenter::Left);
}
