use std::sync::OnceLock;

use algorithms_fixtures::dense_fixtures::{bond_term_hamiltonian, smallest_eigenvalue};
use algorithms_fixtures::{PanicCase, panic_message};
use ariadnetor_linalg::eigh_with_backend;

use super::*;

/// Known ground energy of `Σ_i S_i · S_{i+1}` on the 8-site open chain.
const E0_8_SITES: f64 = -3.374932598687898;

/// The 8-site chain with `η = 1` over `T = 10`, short enough for the test
/// suite. The bounds of the comparisons of `C` and `S` with exact
/// diagonalization are each about three times the deviation this fixture
/// shows.
fn fixture() -> Inputs {
    let n_sites = 8;
    Inputs {
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

/// Ground-state energy and state of the fixture chain.
fn ground() -> &'static (f64, DenseMps) {
    static GROUND: OnceLock<(f64, DenseMps)> = OnceLock::new();
    GROUND.get_or_init(|| ground_state(&Params::new(fixture())).expect("ground state"))
}

fn pipeline() -> &'static (f64, Correlations) {
    static RUN: OnceLock<(f64, Correlations)> = OnceLock::new();
    RUN.get_or_init(|| {
        let (e0, psi0) = ground();
        (
            *e0,
            correlations(&Params::new(fixture()), *e0, psi0).expect("correlations"),
        )
    })
}

/// The error `ground_state` returns for `params`.
fn ground_state_error(params: &Params) -> RunError {
    match ground_state(params) {
        Ok((e0, _)) => panic!("ground_state succeeded with E0 = {e0}"),
        Err(err) => err,
    }
}

/// The larger of `a` and `b`, or NaN if either is NaN. `f64::max` returns the
/// other operand for a NaN, which would let a NaN deviation pass a bound.
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn max_deviation(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "the spectra have different momentum counts"
    );
    for (row_a, row_b) in a.iter().zip(b) {
        assert_eq!(
            row_a.len(),
            row_b.len(),
            "the spectra have different frequency counts"
        );
    }
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, nan_max)
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

/// `H(J) = J H(1)`, so for `J > 0` the ground energy is linear in `J`; a coupling other
/// than 1 shows whether both builders carry `J` with power one.
#[test]
fn ground_energy_scales_with_the_coupling() {
    let j = 0.7;
    let expected = j * E0_8_SITES;
    let n = fixture().n_sites;
    let h = bond_term_hamiltonian(&vec![bond_term(j); n - 1], &vec![2; n]);
    let exact_e0 = smallest_eigenvalue(&h);
    assert!(
        (exact_e0 - expected).abs() < 1e-10,
        "bond_term does not scale with J: E0 = {exact_e0}, expected {expected}"
    );
    let (e0, _) = ground_state(&Params::new(Inputs { j, ..fixture() })).expect("ground state");
    assert!(
        (e0 - expected).abs() < 1e-10,
        "heisenberg_mpo does not scale with J: E0 = {e0}, expected {expected}"
    );
}

#[test]
fn correlations_match_exact_diagonalization() {
    let params = Params::new(fixture());
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
            deviation = nan_max(
                deviation,
                (v - exact().correlation(x, params.center(), t)).norm(),
            );
        }
    }
    assert!(deviation <= 6e-4, "max |C_TEBD - C_exact| = {deviation:e}");
}

#[test]
fn transform_of_exact_correlations_matches_lehmann() {
    let params = Params::new(fixture());
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
    let params = Params::new(fixture());
    let s = structure_factor(&params, &pipeline().1.values);
    let deviation = max_deviation(&s, &exact().lehmann(&params));
    assert!(deviation <= 2.5e-4, "max |S - S_Lehmann| = {deviation:e}");
}

#[test]
fn rejects_an_exceeded_truncation_budget() {
    let params = Params::new(Inputs {
        tebd_chi_max: Some(2),
        trunc_budget: 1e-6,
        ..fixture()
    });
    let (e0, psi0) = ground();
    let err = correlations(&params, *e0, psi0).expect_err("budget exceeded");
    assert!(
        matches!(err, RunError::TruncationBudget { sum, budget, sample }
            if sum > budget && budget == 1e-6 && (1..=params.n_steps).contains(&sample)),
        "{err:?}"
    );
}

#[test]
fn reports_the_truncation_error_of_a_truncated_run() {
    // η = 2 lets K = 100 pass the window check, which keeps this run short.
    let params = Params::new(Inputs {
        tebd_chi_max: Some(4),
        trunc_budget: f64::INFINITY,
        eta: 2.0,
        n_steps: 100,
        ..fixture()
    });
    let (e0, psi0) = ground();
    let corr =
        correlations(&params, *e0, psi0).expect("a finite sum does not exceed an infinite budget");
    assert!(
        corr.trunc_err_sum > 0.0,
        "chi_max = 4 truncates the 8-site chain"
    );

    // Doubling ψ₀ doubles the evolved state and keeps every truncation
    // decision, so the evolved state's truncation error doubles up to rounding.
    let mut doubled = psi0.clone();
    *doubled.site_mut(0) = doubled.site(0).scaled(c64(2.0));
    let doubled_corr = correlations(&params, *e0, &doubled).expect("infinite budget");
    let ratio = doubled_corr.trunc_err_sum / corr.trunc_err_sum;
    assert!((ratio - 2.0).abs() < 1e-9, "error ratio {ratio}");
}

#[test]
fn rejects_an_unconverged_dmrg() {
    let params = Params::new(Inputs {
        dmrg_max_sweeps: 1,
        ..fixture()
    });
    let err = ground_state_error(&params);
    assert!(
        matches!(err, RunError::DmrgNotConverged { n_sweeps: 1 }),
        "{err:?}"
    );
}

#[test]
fn rejects_a_truncated_dmrg_ground_state() {
    let params = Params::new(Inputs {
        dmrg_chi_max: 2,
        ..fixture()
    });
    let err = ground_state_error(&params);
    assert!(
        matches!(err, RunError::DmrgTruncated { max_trunc_err }
            if max_trunc_err > DMRG_TRUNC_BOUND),
        "{err:?}"
    );
}

#[test]
fn error_report_lists_the_error_and_its_source() {
    let err = RunError::Tebd {
        sample: 3,
        source: TebdError::NormBreakdown { step: 0 },
    };
    assert_eq!(
        error_report(&err),
        "error: the single-step TEBD call that produces sample 3 failed\n  \
         caused by: state norm is zero or not finite after step 0\n"
    );
}

#[test]
fn error_report_follows_every_level_of_the_source_chain() {
    #[derive(Debug, thiserror::Error)]
    #[error("inner")]
    struct Inner;
    #[derive(Debug, thiserror::Error)]
    #[error("middle")]
    struct Middle(#[source] Inner);
    #[derive(Debug, thiserror::Error)]
    #[error("outer")]
    struct Outer(#[source] Middle);
    assert_eq!(
        error_report(&Outer(Middle(Inner))),
        "error: outer\n  caused by: middle\n  caused by: inner\n"
    );
}

/// A zero state has no norm to renormalize, so the first TEBD step breaks
/// down at run time.
#[test]
fn forwards_a_tebd_breakdown() {
    let params = Params::new(fixture());
    let sites = (0..params.n_sites)
        .map(|_| DenseTensor::<C64>::zeros(vec![1, 2, 1]))
        .collect();
    let err = correlations(&params, 0.0, &Mps::from_sites(sites)).expect_err("breakdown");
    assert!(
        matches!(
            err,
            RunError::Tebd {
                sample: 1,
                source: TebdError::NormBreakdown { .. }
            }
        ),
        "{err:?}"
    );
}

#[test]
fn default_inputs_are_accepted() {
    let params = Params::new(default_inputs());
    assert_eq!(params.n_sites, 12);
}

#[test]
fn params_reject_each_violated_precondition() {
    let base = fixture();
    let step_rate = "the time step dt and the window rate eta must be finite and positive";
    let coupling = "the coupling J must be finite and positive";
    let cases = [
        (
            Inputs {
                n_sites: 1,
                ..base.clone()
            },
            "the chain needs at least 2 sites, got 1",
        ),
        (
            Inputs {
                j: 0.0,
                ..base.clone()
            },
            coupling,
        ),
        (
            Inputs {
                j: f64::INFINITY,
                ..base.clone()
            },
            coupling,
        ),
        (
            Inputs {
                dt: 0.0,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Inputs {
                dt: -0.05,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Inputs {
                dt: f64::INFINITY,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Inputs {
                eta: 0.0,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Inputs {
                eta: -1.0,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Inputs {
                eta: f64::INFINITY,
                ..base.clone()
            },
            step_rate,
        ),
        (
            Inputs {
                n_steps: 100,
                ..base.clone()
            },
            "the window e^(-eta t) must decay to at most",
        ),
        (
            Inputs {
                omegas: vec![-1.01 * PI / base.dt],
                ..base.clone()
            },
            "every |omega| must lie below the Nyquist frequency",
        ),
        (
            Inputs {
                qs: vec![f64::NAN],
                ..base.clone()
            },
            "every momentum q must be finite",
        ),
        (
            Inputs {
                omegas: vec![PI / base.dt],
                ..base.clone()
            },
            "every |omega| must lie below the Nyquist frequency",
        ),
        (
            Inputs {
                trunc_budget: -1.0,
                ..base.clone()
            },
            "the truncation budget must be non-negative",
        ),
        (
            Inputs {
                trunc_budget: f64::NAN,
                ..base.clone()
            },
            "the truncation budget must be non-negative",
        ),
        (
            Inputs {
                dmrg_chi_max: 0,
                ..base.clone()
            },
            "the DMRG bond-dimension and sweep caps must be at least 1",
        ),
        (
            Inputs {
                dmrg_max_sweeps: 0,
                ..base.clone()
            },
            "the DMRG bond-dimension and sweep caps must be at least 1",
        ),
        (
            Inputs {
                tebd_chi_max: Some(0),
                ..base.clone()
            },
            "the TEBD bond-dimension cap must be at least 1",
        ),
    ];
    for (inputs, expected) in cases {
        let shown = format!("{inputs:?}");
        let message = panic_message(|| drop(Params::new(inputs)));
        assert!(message.starts_with(expected), "{shown}: {message}");
    }
}

#[test]
fn entry_points_reject_malformed_inputs() {
    let params = Params::new(fixture());
    let (n, k) = (params.n_sites, params.n_steps);
    let transform = "structure_factor needs C_xc(t_k) for k = 0..=K and every site";
    let short = vec![vec![c64(0.0); n]; k];
    let narrow = vec![vec![c64(0.0); n - 1]; k + 1];
    let cases: [(PanicCase, &str); 5] = [
        (
            Box::new(|| drop(heisenberg_mpo(1, 1.0))),
            "heisenberg_mpo needs at least 2 sites, got 1",
        ),
        (
            Box::new({
                let params = params.clone();
                move || drop(correlations(&params, 0.0, &neel_state(n - 1)))
            }),
            "correlations: psi0 has 7 sites, but n_sites is 8",
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
        assert!(message.contains(expected), "{message}");
    }
}

/// Rows are frequencies and columns momenta, while `s` is indexed `[q][ω]`.
#[test]
fn format_table_puts_frequencies_in_rows() {
    let table = format_table(&[0.0, 1.0], &[0.5, 2.0], &[vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert_eq!(
        table,
        "# omega q=0.0000 q=1.0000\n  0.500  1.00000  3.00000\n  2.000  2.00000  4.00000\n"
    );
}

#[test]
fn not_within_fails_closed_on_nan() {
    assert!(!not_within(1.0, 1.0));
    assert!(not_within(2.0, 1.0));
    assert!(not_within(f64::NAN, 1.0));
    assert!(not_within(0.0, f64::NAN));
}

#[test]
fn max_deviation_keeps_a_nan() {
    let zeros = vec![vec![0.0; 2]];
    assert!(max_deviation(&[vec![f64::NAN, 0.0]], &zeros).is_nan());
    assert!(max_deviation(&[vec![0.0, f64::NAN]], &zeros).is_nan());
    assert_eq!(max_deviation(&[vec![0.5, -1.0]], &zeros), 1.0);
}

#[test]
fn max_deviation_rejects_mismatched_shapes() {
    let cases: [(PanicCase, &str); 2] = [
        (
            Box::new(|| {
                max_deviation(&[], &[vec![0.0]]);
            }),
            "different momentum counts",
        ),
        (
            Box::new(|| {
                max_deviation(&[vec![0.0; 2], vec![0.0]], &[vec![0.0; 2], vec![0.0; 2]]);
            }),
            "different frequency counts",
        ),
    ];
    for (f, expected) in cases {
        let message = panic_message(f);
        assert!(message.contains(expected), "{message}");
    }
}
