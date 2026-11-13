//! Dynamical structure factor `S^zz(q, ω)` of the spin-1/2 Heisenberg open
//! chain `H = J Σ_i S_i · S_{i+1}` by real-time evolution.
//!
//! For the center site `c = n / 2` the example measures
//!
//! ```text
//! C_xc(t) = ⟨ψ₀| S^z_x e^{-i(H - E₀)t} S^z_c |ψ₀⟩,
//! C(q, t) = Σ_x cos(q (x - c)) C_xc(t),
//! S(q, ω) = (1/2π) ∫ dt e^{iωt} e^{-η|t|} C(q, t)
//!         = (1/π) Re ∫_0^∞ dt e^{iωt} e^{-ηt} C(q, t).
//! ```
//!
//! The one-sided form holds because, for a real Hamiltonian, `C_xc = C_cx`,
//! so the cosine weights give `C(q, -t) = conj C(q, t)`. With the
//! exponential window the result is the Lehmann sum broadened by a
//! Lorentzian of half width `η`:
//!
//! ```text
//! S(q, ω) = Σ_n A_n(q) (η/π) / ((ω - ω_n)² + η²),
//! A_n(q) = Σ_x cos(q (x - c)) ⟨0|S^z_x|n⟩⟨n|S^z_c|0⟩,   ω_n = E_n - E₀.
//! ```
//!
//! The pipeline: the ground state `ψ₀` and `E₀` by two-site DMRG, `S^z_c ψ₀`
//! by an exact MPO application, real-time TEBD one step at a time, the
//! overlaps `⟨ψ₀|S^z_x|φ(t)⟩` for every `x` at every step, and a trapezoid
//! sum over the samples `t_k = k δt`, `k = 0, …, K`.
//!
//! Run it with
//!
//! ```text
//! cargo run --release -p ariadnetor-algorithms --example dynamical_structure_factor
//! ```

use std::error::Error;
use std::f64::consts::PI;
use std::process::ExitCode;

use ariadnetor_algorithms::dmrg::{DmrgError, DmrgSweepParams, LocalEigensolverParams, dmrg_2site};
use ariadnetor_algorithms::krylov::LanczosParams;
use ariadnetor_algorithms::tebd::{TebdError, TebdParams, TimeStep, tebd};
use ariadnetor_mps::{Mpo, Mps, SpinHalf, TensorChain, TruncSvdParams, apply, braket};
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor, Host};
use num_complex::Complex;

type C64 = Complex<f64>;
type DenseMps = Mps<DenseStorage<C64>, DenseLayout>;
type DenseMpo = Mpo<DenseStorage<C64>, DenseLayout>;

/// Largest DMRG truncation error, in the last sweep, accepted for the
/// ground state.
const DMRG_TRUNC_BOUND: f64 = 1e-8;

/// Largest value the window `e^{-ηt}` may keep at the last sample.
const WINDOW_TAIL_BOUND: f64 = 1e-4;

/// Inputs of one run; [`Params::new`] lists the checks they go through.
#[derive(Debug, Clone)]
struct Inputs {
    /// Number of sites.
    n_sites: usize,
    /// Exchange coupling `J`.
    j: f64,
    /// Time step `δt`.
    dt: f64,
    /// Number of time steps `K`; the samples are `t_k = k δt`, `k = 0, …, K`.
    n_steps: usize,
    /// Window decay rate `η`, the Lorentzian half width.
    eta: f64,
    /// Bond-dimension cap of the DMRG ground state.
    dmrg_chi_max: usize,
    /// Sweep cap of the DMRG ground state.
    dmrg_max_sweeps: usize,
    /// Bond-dimension cap of the time evolution; `None` keeps every
    /// singular value.
    tebd_chi_max: Option<usize>,
    /// Largest accepted truncation error of the evolved state
    /// `e^{-iHt} S^z_c ψ₀`, summed over the TEBD steps.
    trunc_budget: f64,
    /// Momenta `q`.
    qs: Vec<f64>,
    /// Frequencies `ω`.
    omegas: Vec<f64>,
}

mod params {
    use std::f64::consts::PI;
    use std::ops::Deref;

    use super::{Inputs, WINDOW_TAIL_BOUND};

    /// Inputs that [`Params::new`] accepted. The field is private to this
    /// module, so every `Params` went through those checks.
    #[derive(Debug, Clone)]
    pub(super) struct Params(Inputs);

    impl Params {
        /// Accept `inputs` for a run.
        ///
        /// # Panics
        ///
        /// - The chain has fewer than 2 sites.
        /// - `J` is not finite and positive: for `J ≤ 0` the ground state is a
        ///   multiplet whose states give different correlations.
        /// - `δt` or `η` is not finite and positive.
        /// - The window at the last sample is above [`WINDOW_TAIL_BOUND`], so
        ///   the cut-off tail biases `S`.
        /// - A frequency's magnitude is not below the Nyquist frequency
        ///   `π/δt` of the samples.
        /// - A momentum is not finite.
        /// - The truncation budget is negative or NaN.
        /// - A bond-dimension cap or the sweep cap is zero.
        pub(super) fn new(inputs: Inputs) -> Self {
            assert!(
                inputs.n_sites >= 2,
                "the chain needs at least 2 sites, got {}",
                inputs.n_sites
            );
            assert!(
                inputs.j.is_finite() && inputs.j > 0.0,
                "the coupling J must be finite and positive (antiferromagnetic), got {}",
                inputs.j
            );
            assert!(
                inputs.dt.is_finite()
                    && inputs.dt > 0.0
                    && inputs.eta.is_finite()
                    && inputs.eta > 0.0,
                "the time step dt and the window rate eta must be finite and positive, \
                 got dt = {}, eta = {}",
                inputs.dt,
                inputs.eta
            );
            let tail = (-inputs.eta * inputs.dt * inputs.n_steps as f64).exp();
            assert!(
                tail <= WINDOW_TAIL_BOUND,
                "the window e^(-eta t) must decay to at most {WINDOW_TAIL_BOUND:e} at the last sample, \
                 got e^(-eta K dt) = {tail:e}"
            );
            let nyquist = PI / inputs.dt;
            assert!(
                inputs.omegas.iter().all(|w| w.abs() < nyquist),
                "every |omega| must lie below the Nyquist frequency pi/dt = {nyquist}"
            );
            assert!(
                inputs.qs.iter().all(|q| q.is_finite()),
                "every momentum q must be finite"
            );
            assert!(
                inputs.trunc_budget >= 0.0,
                "the truncation budget must be non-negative, got {}",
                inputs.trunc_budget
            );
            assert!(
                inputs.dmrg_chi_max >= 1 && inputs.dmrg_max_sweeps >= 1,
                "the DMRG bond-dimension and sweep caps must be at least 1"
            );
            assert!(
                inputs.tebd_chi_max != Some(0),
                "the TEBD bond-dimension cap must be at least 1"
            );
            Self(inputs)
        }

        /// Center site `c`.
        pub(super) fn center(&self) -> usize {
            self.0.n_sites / 2
        }
    }

    impl Deref for Params {
        type Target = Inputs;

        fn deref(&self) -> &Inputs {
            &self.0
        }
    }
}

use params::Params;

/// Errors of a run.
#[derive(Debug, thiserror::Error)]
enum RunError {
    /// `dmrg_2site` failed.
    #[error("DMRG failed")]
    Dmrg(#[from] DmrgError),
    /// DMRG's truncation error is above [`DMRG_TRUNC_BOUND`] or NaN.
    #[error(
        "DMRG truncated the ground state: truncation error {max_trunc_err:e} exceeds {bound:e}"
    )]
    DmrgTruncated {
        /// The last sweep's largest truncation error.
        max_trunc_err: f64,
        /// [`DMRG_TRUNC_BOUND`].
        bound: f64,
    },
    /// DMRG stopped before convergence.
    #[error("DMRG did not converge in {n_sweeps} sweeps")]
    DmrgNotConverged {
        /// Sweeps run.
        n_sweeps: usize,
    },
    /// `tebd` failed.
    #[error("the single-step TEBD call that produces sample {sample} failed")]
    Tebd {
        /// Sample `k` whose step failed; `k = 1` is the first TEBD step.
        sample: usize,
        /// The `tebd` error.
        #[source]
        source: TebdError,
    },
    /// The summed TEBD truncation error is above the budget or NaN.
    #[error("TEBD truncation error {sum:e} exceeds the budget {budget:e} at sample {sample}")]
    TruncationBudget {
        /// Summed truncation error of the evolved state up to `sample`.
        sum: f64,
        /// [`Inputs::trunc_budget`].
        budget: f64,
        /// Sample `k` whose step exceeded the budget; `k = 1` is the first
        /// TEBD step.
        sample: usize,
    },
}

/// Whether `value` is above `bound` or either is NaN, so that a NaN error or
/// bound fails the check instead of passing it.
fn exceeds(value: f64, bound: f64) -> bool {
    value.partial_cmp(&bound).is_none_or(|o| o.is_gt())
}

fn c64(re: f64) -> C64 {
    C64::new(re, 0.0)
}

/// `J (S^z S^z + (S⁺S⁻ + S⁻S⁺)/2)` on two sites, with legs
/// `(out_i, out_i+1, in_i, in_i+1)`.
fn bond_term(j: f64) -> DenseTensor<C64> {
    let (sz, sp, sm) = (
        SpinHalf.sz::<C64>(),
        SpinHalf.sp::<C64>(),
        SpinHalf.sm::<C64>(),
    );
    let mut h = DenseTensor::<C64>::zeros(vec![2, 2, 2, 2]);
    for o1 in 0..2 {
        for o2 in 0..2 {
            for i1 in 0..2 {
                for i2 in 0..2 {
                    let zz = sz.get([o1, i1]) * sz.get([o2, i2]);
                    let flip =
                        sp.get([o1, i1]) * sm.get([o2, i2]) + sm.get([o1, i1]) * sp.get([o2, i2]);
                    h.set([o1, o2, i1, i2], (zz + flip * 0.5) * j);
                }
            }
        }
    }
    h
}

/// MPO of `J Σ_i S_i · S_{i+1}` on `n` sites, from the five-state automaton
/// whose last row starts a term and whose first column ends it.
///
/// # Panics
///
/// Panics if `n < 2`.
fn heisenberg_mpo(n: usize, j: f64) -> DenseMpo {
    assert!(n >= 2, "heisenberg_mpo needs at least 2 sites, got {n}");
    let (id, sz, sp, sm) = (
        SpinHalf.id::<C64>(),
        SpinHalf.sz::<C64>(),
        SpinHalf.sp::<C64>(),
        SpinHalf.sm::<C64>(),
    );
    // (row, column, operator, coefficient) of the 5 × 5 operator matrix.
    let cells = [
        (0, 0, &id, 1.0),
        (1, 0, &sp, 1.0),
        (2, 0, &sm, 1.0),
        (3, 0, &sz, 1.0),
        (4, 1, &sm, 0.5 * j),
        (4, 2, &sp, 0.5 * j),
        (4, 3, &sz, j),
        (4, 4, &id, 1.0),
    ];
    let sites = (0..n)
        .map(|site| {
            let (first, last) = (site == 0, site == n - 1);
            let shape = vec![if first { 1 } else { 5 }, 2, 2, if last { 1 } else { 5 }];
            let mut w = DenseTensor::<C64>::zeros(shape);
            for &(row, col, op, coeff) in &cells {
                if (first && row != 4) || (last && col != 0) {
                    continue;
                }
                let (l, r) = (if first { 0 } else { row }, if last { 0 } else { col });
                for k in 0..2 {
                    for b in 0..2 {
                        // Leg 1 is the input (ket) index, leg 2 the output.
                        w.set([l, k, b, r], op.get([b, k]) * coeff);
                    }
                }
            }
            w
        })
        .collect();
    Mpo::from_sites(sites)
}

/// `S^z` at site `x` of an `n`-site chain, identity elsewhere.
///
/// # Panics
///
/// Panics if `x >= n`.
fn sz_mpo(n: usize, x: usize) -> DenseMpo {
    assert!(x < n, "sz_mpo: site {x} is outside a chain of {n} sites");
    let mut w = DenseMpo::identity(&vec![2; n]);
    let sz = SpinHalf.sz::<C64>();
    for k in 0..2 {
        for b in 0..2 {
            w.site_mut(x).set([0, k, b, 0], sz.get([b, k]));
        }
    }
    w
}

/// Néel product state `|↑↓↑↓…⟩`.
fn neel_state(n: usize) -> DenseMps {
    let sites = (0..n)
        .map(|x| {
            let mut a = DenseTensor::<C64>::zeros(vec![1, 2, 1]);
            a.set([0, x % 2, 0], c64(1.0));
            a
        })
        .collect();
    Mps::from_sites(sites)
}

/// Ground-state energy and state by two-site DMRG from the Néel state.
fn ground_state(params: &Params) -> Result<(f64, DenseMps), RunError> {
    let dmrg = DmrgSweepParams {
        max_sweeps: params.dmrg_max_sweeps,
        min_sweeps: params.dmrg_max_sweeps.min(3),
        energy_tol: 1e-12,
        eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 200,
            tol: 1e-12,
            seed: Some(7),
        }),
        trunc: TruncSvdParams {
            chi_max: Some(params.dmrg_chi_max),
            target_trunc_err: None,
        },
    };
    let mpo = heisenberg_mpo(params.n_sites, params.j);
    let (result, psi) = dmrg_2site::<C64, _, _>(&mpo, &neel_state(params.n_sites), &dmrg)?;
    let max_trunc_err = result.sweeps.last().map_or(0.0, |s| s.max_trunc_err);
    if exceeds(max_trunc_err, DMRG_TRUNC_BOUND) {
        return Err(RunError::DmrgTruncated {
            max_trunc_err,
            bound: DMRG_TRUNC_BOUND,
        });
    }
    if !result.converged {
        return Err(RunError::DmrgNotConverged {
            n_sweeps: result.n_sweeps,
        });
    }
    Ok((result.energy, psi))
}

/// `C_xc(t_k)` indexed by `[k][x]`, with the summed truncation error of the
/// evolved state.
#[derive(Debug, Clone)]
struct Correlations {
    values: Vec<Vec<C64>>,
    trunc_err_sum: f64,
}

/// Measure `C_xc(t_k)` for every site `x` and sample `k`.
///
/// # Panics
///
/// Panics if `psi0` does not have
/// `n_sites` sites.
fn correlations(params: &Params, e0: f64, psi0: &DenseMps) -> Result<Correlations, RunError> {
    let n = params.n_sites;
    assert!(
        psi0.len() == n,
        "correlations: psi0 has {} sites, but n_sites is {n}",
        psi0.len()
    );
    let backend = Host::shared();
    let sz: Vec<DenseMpo> = (0..n).map(|x| sz_mpo(n, x)).collect();
    let terms = vec![bond_term(params.j); n - 1];
    let tebd_params = TebdParams {
        time_step: TimeStep::Real(params.dt),
        n_steps: 1,
        trunc: TruncSvdParams {
            chi_max: params.tebd_chi_max,
            target_trunc_err: None,
        },
    };
    let measure = |phi: &DenseMps, factor: C64| -> Vec<C64> {
        sz.iter()
            .map(|op| factor * braket::<C64, _, _, _>(backend.as_ref(), psi0, op, phi))
            .collect()
    };

    let mut phi = apply::<C64, _, _, _>(backend.as_ref(), &sz[params.center()], psi0, None);
    let mut values = vec![measure(&phi, c64(1.0))];
    // `tebd` renormalizes after every step; the product of the recorded
    // norms restores the norm the evolution carries, ‖S^z_c ψ₀‖ included.
    // Each step's truncation error is measured on the state `tebd` was
    // given, the evolved state divided by the product so far, so scaling it
    // by that product puts every step's error on the evolved state's scale.
    let mut norm_product = 1.0;
    let mut trunc_err_sum = 0.0;
    for sample in 1..=params.n_steps {
        let record = &tebd(&mut phi, &terms, &tebd_params)
            .map_err(|source| RunError::Tebd { sample, source })?[0];
        trunc_err_sum += norm_product * record.trunc_err_sum;
        norm_product *= record.norm;
        if exceeds(trunc_err_sum, params.trunc_budget) {
            return Err(RunError::TruncationBudget {
                sum: trunc_err_sum,
                budget: params.trunc_budget,
                sample,
            });
        }
        let t = sample as f64 * params.dt;
        values.push(measure(&phi, C64::from_polar(norm_product, e0 * t)));
    }
    Ok(Correlations {
        values,
        trunc_err_sum,
    })
}

/// `S(q, ω)` indexed by `[q][ω]` from `C_xc(t_k)` indexed by `[k][x]`.
///
/// # Panics
///
/// Panics if `values` is not
/// `(K + 1) × n_sites`.
fn structure_factor(params: &Params, values: &[Vec<C64>]) -> Vec<Vec<f64>> {
    assert!(
        values.len() == params.n_steps + 1 && values.iter().all(|row| row.len() == params.n_sites),
        "structure_factor needs C_xc(t_k) for k = 0..=K and every site"
    );
    let c = params.center() as f64;
    let last = values.len() - 1;
    // Trapezoid weight times the window, per sample.
    let windowed: Vec<f64> = (0..=last)
        .map(|k| {
            let weight = if k == 0 || k == last { 0.5 } else { 1.0 } * params.dt;
            weight * (-params.eta * k as f64 * params.dt).exp()
        })
        .collect();
    params
        .qs
        .iter()
        .map(|&q| {
            let cq: Vec<C64> = values
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(x, v)| v * (q * (x as f64 - c)).cos())
                        .sum()
                })
                .collect();
            params
                .omegas
                .iter()
                .map(|&w| {
                    let sum: C64 = cq
                        .iter()
                        .zip(&windowed)
                        .enumerate()
                        .map(|(k, (v, &weight))| {
                            v * C64::from_polar(weight, w * k as f64 * params.dt)
                        })
                        .sum();
                    sum.re / PI
                })
                .collect()
        })
        .collect()
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprint!("{}", error_report(&err));
            ExitCode::FAILURE
        }
    }
}

/// `err` and the chain of its sources, one per line.
fn error_report(err: &dyn Error) -> String {
    let mut report = format!("error: {err}\n");
    let mut source = err.source();
    while let Some(cause) = source {
        report.push_str(&format!("  caused by: {cause}\n"));
        source = cause.source();
    }
    report
}

/// Inputs of the run `main` performs.
fn default_inputs() -> Inputs {
    let n_sites = 12;
    Inputs {
        n_sites,
        j: 1.0,
        dt: 0.05,
        n_steps: 500,
        eta: 0.5,
        dmrg_chi_max: 64,
        dmrg_max_sweeps: 30,
        tebd_chi_max: Some(64),
        trunc_budget: 1e-3,
        qs: (0..=n_sites / 2)
            .map(|k| 2.0 * PI * k as f64 / n_sites as f64)
            .collect(),
        omegas: (0..=40).map(|i| 0.1 * f64::from(i)).collect(),
    }
}

/// Run the default parameters and print `S(q, ω)` as a table.
fn run() -> Result<(), RunError> {
    let params = Params::new(default_inputs());
    let (e0, psi0) = ground_state(&params)?;
    let corr = correlations(&params, e0, &psi0)?;
    let s = structure_factor(&params, &corr.values);

    println!("# n = {}, J = {}, E0 = {e0:.12}", params.n_sites, params.j);
    println!(
        "# dt = {}, K = {}, eta = {}, summed TEBD truncation error = {:e}",
        params.dt, params.n_steps, params.eta, corr.trunc_err_sum
    );
    print!("# omega");
    for q in &params.qs {
        print!(" q={q:.4}");
    }
    println!();
    for (i, w) in params.omegas.iter().enumerate() {
        print!("{w:7.3}");
        for row in &s {
            print!(" {:8.5}", row[i]);
        }
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests;
