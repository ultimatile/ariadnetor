use std::sync::OnceLock;

use algorithms_fixtures::dense_fixtures::bond_term_hamiltonian;
use ariadnetor_linalg::eigh_with_backend;

use super::*;

/// Known ground energy of `Σ_i S_i · S_{i+1}` on the 8-site open chain.
const E0_8_SITES: f64 = -3.374932598687898;

/// The 8-site chain with `η = 1` over `T = 10`, short enough for the test
/// suite. The bounds of the comparisons of `C` and `S` with exact
/// diagonalization are each about three times the deviation this fixture
/// shows.
fn fixture() -> Params {
    let n_sites = 8;
    Params {
        n_sites,
        j: 1.0,
        dt: 0.05,
        n_steps: 200,
        eta: 1.0,
        dmrg_chi_max: 64,
        dmrg_max_sweeps: 30,
        tebd_chi_max: None,
        trunc_budget: 1e-8,
        qs: (0..=4)
            .map(|k| 2.0 * PI * f64::from(k) / n_sites as f64)
            .collect(),
        omegas: (0..=60).map(|i| -1.0 + 0.1 * f64::from(i)).collect(),
    }
}

/// Exact energies of the fixture chain and the matrix elements
/// `⟨n|S^z_x|0⟩`.
struct Exact {
    energies: Vec<f64>,
    /// `[x][n]`.
    sz_elems: Vec<Vec<C64>>,
}

impl Exact {
    fn correlation(&self, x: usize, c: usize, t: f64) -> C64 {
        let e0 = self.energies[0];
        (0..self.energies.len())
            .map(|n| {
                self.sz_elems[x][n].conj()
                    * self.sz_elems[c][n]
                    * C64::from_polar(1.0, -(self.energies[n] - e0) * t)
            })
            .sum()
    }

    fn lehmann(&self, params: &Params) -> Vec<Vec<f64>> {
        let c = params.center();
        let e0 = self.energies[0];
        let eta = params.eta;
        params
            .qs
            .iter()
            .map(|&q| {
                let weights: Vec<C64> = (0..self.energies.len())
                    .map(|n| {
                        (0..params.n_sites)
                            .map(|x| {
                                self.sz_elems[x][n].conj()
                                    * self.sz_elems[c][n]
                                    * (q * (x as f64 - c as f64)).cos()
                            })
                            .sum()
                    })
                    .collect();
                params
                    .omegas
                    .iter()
                    .map(|&w| {
                        let total: C64 = weights
                            .iter()
                            .zip(&self.energies)
                            .map(|(a, e)| a * (eta / PI) / ((w - (e - e0)).powi(2) + eta * eta))
                            .sum();
                        total.re
                    })
                    .collect()
            })
            .collect()
    }
}

fn exact() -> &'static Exact {
    static EXACT: OnceLock<Exact> = OnceLock::new();
    EXACT.get_or_init(|| {
        let n = fixture().n_sites;
        let h = bond_term_hamiltonian(&vec![bond_term(1.0); n - 1], &vec![2; n]);
        let (vals, vecs) = eigh_with_backend(Host::shared().as_ref(), &h, 1).expect("eigh");
        let dim = 1 << n;
        // Basis index `s` flattens `(s_0, …, s_{n-1})` row-major, so site
        // `x` is bit `n - 1 - x`; bit value 0 is spin up.
        let sz_diag = |x: usize, s: usize| {
            if (s >> (n - 1 - x)) & 1 == 0 {
                0.5
            } else {
                -0.5
            }
        };
        let sz_elems = (0..n)
            .map(|x| {
                (0..dim)
                    .map(|m| {
                        (0..dim)
                            .map(|s| vecs.get([s, m]).conj() * sz_diag(x, s) * vecs.get([s, 0]))
                            .sum()
                    })
                    .collect()
            })
            .collect();
        Exact {
            energies: vals.data_slice().to_vec(),
            sz_elems,
        }
    })
}

fn pipeline() -> &'static (f64, Correlations) {
    static RUN: OnceLock<(f64, Correlations)> = OnceLock::new();
    RUN.get_or_init(|| {
        let params = fixture();
        let (e0, psi0) = ground_state(&params).expect("ground state");
        (e0, correlations(&params, e0, &psi0).expect("correlations"))
    })
}

fn max_deviation(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn bond_terms_reproduce_the_known_ground_energy() {
    let e0 = exact().energies[0];
    assert!(
        (e0 - E0_8_SITES).abs() < 1e-10,
        "bond_term(1.0) does not build J Σ S·S with J = 1: E0 = {e0}, expected {E0_8_SITES}"
    );
}

#[test]
fn ground_energy_matches_exact_diagonalization() {
    let (e0, _) = pipeline();
    let exact_e0 = exact().energies[0];
    assert!(
        (e0 - exact_e0).abs() < 1e-10,
        "heisenberg_mpo and bond_term disagree on the coupling: DMRG E0 = {e0}, exact {exact_e0}"
    );
}

#[test]
fn correlations_match_exact_diagonalization() {
    let params = fixture();
    let (_, corr) = pipeline();
    assert_eq!(corr.values.len(), params.n_steps + 1);
    assert!(
        corr.trunc_err_sum <= 1e-12,
        "untruncated run truncated {}",
        corr.trunc_err_sum
    );
    let mut deviation: f64 = 0.0;
    for (k, row) in corr.values.iter().enumerate() {
        assert_eq!(row.len(), params.n_sites);
        for (x, v) in row.iter().enumerate() {
            let t = k as f64 * params.dt;
            deviation = deviation.max((v - exact().correlation(x, params.center(), t)).norm());
        }
    }
    assert!(deviation <= 6e-4, "max |C_TEBD - C_exact| = {deviation:e}");
}

#[test]
fn transform_of_exact_correlations_matches_lehmann() {
    let params = fixture();
    let values: Vec<Vec<C64>> = (0..=params.n_steps)
        .map(|k| {
            let t = k as f64 * params.dt;
            (0..params.n_sites)
                .map(|x| exact().correlation(x, params.center(), t))
                .collect()
        })
        .collect();
    let deviation = max_deviation(
        &structure_factor(&params, &values),
        &exact().lehmann(&params),
    );
    assert!(deviation <= 2e-4, "max |S - S_Lehmann| = {deviation:e}");
}

#[test]
fn structure_factor_matches_lehmann() {
    let params = fixture();
    let s = structure_factor(&params, &pipeline().1.values);
    let deviation = max_deviation(&s, &exact().lehmann(&params));
    assert!(deviation <= 2.5e-4, "max |S - S_Lehmann| = {deviation:e}");
}

#[test]
fn rejects_an_exceeded_truncation_budget() {
    let params = Params {
        tebd_chi_max: Some(2),
        trunc_budget: 1e-6,
        ..fixture()
    };
    let (e0, psi0) = ground_state(&params).expect("ground state");
    let err = correlations(&params, e0, &psi0).expect_err("budget exceeded");
    assert!(
        matches!(err, RunError::TruncationBudget { sum, budget, step }
            if sum > budget && budget == 1e-6 && (1..=params.n_steps).contains(&step)),
        "{err:?}"
    );
}

#[test]
fn reports_the_truncation_error_of_a_truncated_run() {
    // η = 2 lets K = 100 pass the window check, which keeps this run short.
    let params = Params {
        tebd_chi_max: Some(4),
        trunc_budget: f64::INFINITY,
        eta: 2.0,
        n_steps: 100,
        ..fixture()
    };
    let (e0, psi0) = ground_state(&params).expect("ground state");
    let corr = correlations(&params, e0, &psi0).expect("an infinite budget is never exceeded");
    assert!(
        corr.trunc_err_sum > 0.0,
        "chi_max = 4 truncates the 8-site chain"
    );
}

#[test]
fn rejects_an_unconverged_dmrg() {
    let params = Params {
        dmrg_max_sweeps: 1,
        ..fixture()
    };
    let Err(err) = ground_state(&params) else {
        panic!("one sweep does not converge");
    };
    assert!(
        matches!(err, RunError::DmrgNotConverged { n_sweeps: 1 }),
        "{err:?}"
    );
}

#[test]
fn rejects_a_truncated_dmrg_ground_state() {
    let params = Params {
        dmrg_chi_max: 2,
        ..fixture()
    };
    let Err(err) = ground_state(&params) else {
        panic!("bond dimension 2 truncates");
    };
    assert!(
        matches!(err, RunError::DmrgTruncated { max_trunc_err, bound }
            if max_trunc_err > bound && bound == DMRG_TRUNC_BOUND),
        "{err:?}"
    );
}

#[test]
fn forwards_dmrg_and_tebd_errors() {
    let params = Params {
        dmrg_max_sweeps: 0,
        ..fixture()
    };
    let Err(err) = ground_state(&params) else {
        panic!("zero sweeps are invalid DMRG parameters");
    };
    assert!(matches!(err, RunError::Dmrg(_)), "{err:?}");

    let params = Params {
        tebd_chi_max: Some(0),
        ..fixture()
    };
    let psi0 = neel_state(params.n_sites);
    let err = correlations(&params, 0.0, &psi0).expect_err("chi_max = 0 is invalid");
    assert!(matches!(err, RunError::Tebd(_)), "{err:?}");
}

/// A call expected to panic.
type PanicCase = Box<dyn FnOnce() + std::panic::UnwindSafe>;

/// The message `f` panics with.
fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let payload = std::panic::catch_unwind(f).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .expect("a panic message")
}

#[test]
fn validate_rejects_each_violated_precondition() {
    let base = fixture();
    let step_rate = "the time step dt and the window rate eta must be finite and positive";
    let coupling = "the coupling J must be finite and positive";
    let cases = [
        (
            Params {
                n_sites: 1,
                ..base.clone()
            },
            "the chain needs at least 2 sites, got 1",
        ),
        (
            Params {
                j: 0.0,
                ..base.clone()
            },
            coupling,
        ),
        (
            Params {
                j: f64::INFINITY,
                ..base.clone()
            },
            coupling,
        ),
        (
            Params {
                dt: 0.0,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Params {
                dt: -0.05,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Params {
                dt: f64::INFINITY,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Params {
                eta: 0.0,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Params {
                eta: -1.0,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Params {
                eta: f64::INFINITY,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Params {
                n_steps: 100,
                ..base.clone()
            },
            "the window e^(-eta t) must decay to at most",
        ),
        (
            Params {
                omegas: vec![-1.01 * PI / base.dt],
                ..base.clone()
            },
            "every |omega| must lie below the Nyquist frequency",
        ),
    ];
    for (params, expected) in cases {
        let message = panic_message(|| params.validate());
        assert!(message.starts_with(expected), "{params:?}: {message}");
    }
}

#[test]
fn builders_and_transform_reject_malformed_inputs() {
    let params = fixture();
    let (n, k) = (params.n_sites, params.n_steps);
    let transform = "structure_factor needs C_xc(t_k) for k = 0..=K and every site";
    let short = vec![vec![C64::new(0.0, 0.0); n]; k];
    let narrow = vec![vec![C64::new(0.0, 0.0); n - 1]; k + 1];
    let cases: [(PanicCase, &str); 4] = [
        (
            Box::new(|| drop(heisenberg_mpo(1, 1.0))),
            "heisenberg_mpo needs at least 2 sites, got 1",
        ),
        (
            Box::new(move || drop(sz_mpo(n, n))),
            "sz_mpo: site 8 is outside a chain of 8 sites",
        ),
        (
            Box::new({
                let params = params.clone();
                move || drop(structure_factor(&params, &short))
            }),
            transform,
        ),
        (
            Box::new(move || drop(structure_factor(&params, &narrow))),
            transform,
        ),
    ];
    for (f, expected) in cases {
        let message = panic_message(f);
        assert!(message.starts_with(expected), "{message}");
    }
}

#[test]
fn exceeds_fails_closed_on_nan() {
    assert!(!exceeds(1.0, 1.0));
    assert!(exceeds(2.0, 1.0));
    assert!(exceeds(f64::NAN, 1.0));
    assert!(exceeds(0.0, f64::NAN));
}
