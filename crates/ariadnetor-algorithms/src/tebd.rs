//! Time-evolving block decimation (TEBD) on a Dense MPS.
//!
//! [`tebd`] evolves an MPS under `H = Σ_i h_i`, where each bond term `h_i`
//! acts on sites `i` and `i + 1`, by repeated second-order Trotter steps.
//! On a chain of `n` sites, one step of length `z` (`i·dt` in real time,
//! `tau` in imaginary time) is
//!
//! ```text
//! S(z) = g_0(z/2) ⋯ g_{n-2}(z/2) · g_{n-2}(z/2) ⋯ g_0(z/2),   g_i(w) = exp(-w h_i)
//! ```
//!
//! applied right factor first: a left-to-right pass of half-step gates
//! followed by a right-to-left pass. Each `h_i` enters with total weight
//! `z` and `S(-z) S(z) = 1`, so `S(z) = exp(-zH) + O(z³)`.
//!
//! The gates follow the sweep order: the left-to-right pass leaves the
//! orthogonality center on the right site of each pair and the return pass
//! leaves it on the left site, so from the second gate of a call on, the
//! center is already on the gated pair. No gauge step runs between gates,
//! and, given a canonical-form tag that holds, each gate's truncation error
//! is the exact state error it introduces.

use ariadnetor_core::{Scalar, try_real_from_f64};
use ariadnetor_linalg::{
    LinalgError, TruncSvdParams, expm_antihermitian_with_backend, expm_hermitian_with_backend,
};
use ariadnetor_mps::{CanonicalForm, GateCenter, Mps, TensorChain};
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor, Host};
use num_traits::{Float, One, ToPrimitive, Zero};

use crate::trunc_params::validate_trunc_params;

/// Kind and length of one Trotter step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeStep {
    /// Real-time step `dt`: gates `exp(-i dt/2 h)`. Requires a complex
    /// scalar.
    Real(f64),
    /// Imaginary-time step `tau`: gates `exp(-tau/2 h)`.
    Imaginary(f64),
}

/// Caller-supplied parameters for [`tebd`].
#[derive(Debug, Clone)]
pub struct TebdParams {
    /// Kind and length of each step. The length must be finite and
    /// positive, and so must its half converted to the storage's real
    /// scalar type. Evolving backward in real time is expressed by negating
    /// the bond terms.
    pub time_step: TimeStep,
    /// Number of steps to run.
    pub n_steps: usize,
    /// Truncation applied to the new bond after every gate. `chi_max`
    /// applies as is; `target_trunc_err` is relative to the state's norm
    /// before the gate, so a gate may discard singular values whose
    /// Frobenius norm is up to that fraction of it.
    pub trunc: TruncSvdParams,
}

/// Diagnostics of one Trotter step.
#[derive(Debug, Clone)]
pub struct TebdStepRecord<R> {
    /// Index of the step, counted from 0 within the call.
    pub step: usize,
    /// Sum over this step's gates of the Frobenius norms of the discarded
    /// singular values. In real time, given a canonical-form tag that holds
    /// (see [`tebd`]), it bounds the distance between the untruncated step
    /// applied to the incoming state and `norm` times the renormalized
    /// outgoing state.
    pub trunc_err_sum: R,
    /// Norm of the state after this step's gates, before renormalization.
    /// Once an imaginary-time run has converged, `-ln(norm) / tau`
    /// estimates the ground-state energy.
    pub norm: R,
    /// Largest bond dimension after this step.
    pub max_bond: usize,
}

/// Errors returned by [`tebd`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TebdError {
    /// The MPS has fewer than 2 sites.
    #[error("TEBD requires at least 2 sites, got {n_sites}")]
    TooFewSites {
        /// The site count supplied.
        n_sites: usize,
    },
    /// The number of bond terms is not the number of bonds.
    #[error("{bonds} bonds but {bond_terms} bond terms")]
    LengthMismatch {
        /// Number of bonds of the MPS (`len() - 1`).
        bonds: usize,
        /// Number of bond terms supplied.
        bond_terms: usize,
    },
    /// A bond term's shape does not match the physical dimensions of its
    /// two sites.
    #[error("bond term {bond} has shape {found:?}, expected {expected:?}")]
    BondTermShape {
        /// Index of the bond.
        bond: usize,
        /// `[d_bond, d_bond+1, d_bond, d_bond+1]`.
        expected: [usize; 4],
        /// The shape supplied.
        found: Vec<usize>,
    },
    /// [`TebdParams`] failed validation. `detail` names the constraint
    /// that fired.
    #[error("invalid TebdParams: {detail}")]
    InvalidParams {
        /// Names the validation constraint that failed.
        detail: &'static str,
    },
    /// Exponentiating a bond term failed, for example for a term that is
    /// not Hermitian within the exponential's tolerance.
    #[error("building the gate of bond {bond} failed")]
    GateConstruction {
        /// Index of the bond.
        bond: usize,
        /// The underlying matrix-exponential error.
        #[source]
        source: LinalgError,
    },
    /// The norm after a step's gates is zero or not finite.
    #[error("state norm is zero or not finite after step {step}")]
    NormBreakdown {
        /// Index of the failing step.
        step: usize,
    },
}

/// Evolve `psi` in place by `params.n_steps` second-order Trotter steps of
/// `H = Σ_i bond_terms[i]`, renormalizing the state after every step.
///
/// `bond_terms[i]` is the Hermitian term acting on sites `i` and `i + 1`,
/// with legs `(out_i, out_i+1, in_i, in_i+1)`: it is the
/// `reshape_logical([d_i, d_i+1, d_i, d_i+1])` of a `d_i·d_i+1 × d_i·d_i+1`
/// matrix whose rows are outputs, the leg order
/// [`Mps::apply_two_site_gate`] takes. On-site terms are folded into the
/// bond terms by the caller.
///
/// `psi` may be in any canonical form. The form is trusted, not checked:
/// the sites it marks isometric must be, or the truncation errors stop
/// being state errors. After each step the form is
/// `Mixed { center: 0 }` and the state has unit norm; with `n_steps == 0`,
/// `psi` is left unchanged.
/// Returns one record per step.
///
/// The driver is host-pinned: every kernel runs on the substrate
/// [`Host::shared`] returns.
///
/// # Errors
///
/// - [`TebdError::TooFewSites`] if `psi` has fewer than 2 sites.
/// - [`TebdError::LengthMismatch`] if `bond_terms.len() != psi.len() - 1`.
/// - [`TebdError::BondTermShape`] if a term's shape does not match its
///   sites.
/// - [`TebdError::InvalidParams`] if the step length or its half in the
///   real scalar type is not finite and positive, the step is real time on
///   a real scalar type, `trunc.chi_max` is `Some(0)`, or
///   `trunc.target_trunc_err` is not finite or is negative.
/// - [`TebdError::GateConstruction`] if exponentiating a term fails.
/// - [`TebdError::NormBreakdown`] if the norm after a step is zero or not
///   finite.
///
/// Every error except `NormBreakdown` is returned before `psi` is
/// modified, whatever `n_steps` is. On `NormBreakdown`, `psi` holds the
/// failing step's unnormalized state.
///
/// # Panics
///
/// `psi` must be a well-formed chain (every site rank 3, adjacent bond
/// dimensions equal) whose canonical-form tag is consistent with its
/// length; the driver does not check this. Panics if a linear-algebra call
/// inside [`Mps::apply_two_site_gate`] fails or, with a truncation target,
/// if the state norm before a gate is not finite. In imaginary time the
/// state is renormalized only between steps, and within a step its norm
/// changes by about `exp(-tau E)`, which on a long chain can overflow the
/// real scalar type and end in one of these panics. A panic inside a step
/// can leave `psi` with that step's earlier gates applied.
pub fn tebd<T: Scalar>(
    psi: &mut Mps<DenseStorage<T>, DenseLayout>,
    bond_terms: &[DenseTensor<T>],
    params: &TebdParams,
) -> Result<Vec<TebdStepRecord<T::Real>>, TebdError> {
    let n_sites = psi.len();
    if n_sites < 2 {
        return Err(TebdError::TooFewSites { n_sites });
    }
    if bond_terms.len() != n_sites - 1 {
        return Err(TebdError::LengthMismatch {
            bonds: n_sites - 1,
            bond_terms: bond_terms.len(),
        });
    }
    for (bond, term) in bond_terms.iter().enumerate() {
        let d_l = psi.site(bond).shape()[1];
        let d_r = psi.site(bond + 1).shape()[1];
        let expected = [d_l, d_r, d_l, d_r];
        if term.shape() != expected {
            return Err(TebdError::BondTermShape {
                bond,
                expected,
                found: term.shape().to_vec(),
            });
        }
    }

    let half = validate_params::<T>(params)?;
    let zero = T::Real::zero();
    let backend = Host::shared();
    let gates = bond_terms
        .iter()
        .enumerate()
        .map(|(bond, term)| {
            let shape = term.shape().to_vec();
            let matrix = term.reshape_logical(vec![shape[0] * shape[1], shape[2] * shape[3]]);
            let exp = match params.time_step {
                TimeStep::Real(_) => expm_antihermitian_with_backend(
                    backend.as_ref(),
                    &matrix.scaled(T::from_real_imag(zero, -half)),
                    1,
                ),
                TimeStep::Imaginary(_) => expm_hermitian_with_backend(
                    backend.as_ref(),
                    &matrix.scaled(T::from_real_imag(-half, zero)),
                    1,
                ),
            };
            exp.map(|gate| gate.reshape_logical(shape))
                .map_err(|source| TebdError::GateConstruction { bond, source })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // The first gate would establish this gauge anyway; doing it here also
    // keeps the relative truncation target's `Mps::norm` reads, which
    // contract the whole chain for these tags, single-site.
    if params.n_steps > 0
        && matches!(
            psi.canonical_form(),
            CanonicalForm::Unknown | CanonicalForm::Partial { .. }
        )
    {
        psi.canonicalize::<T, _>(backend.as_ref(), 0);
    }

    let mut records = Vec::new();
    for step in 0..params.n_steps {
        let mut trunc_err_sum = zero;
        let passes = gates
            .iter()
            .enumerate()
            .map(|pair| (pair, GateCenter::Right))
            .chain(
                gates
                    .iter()
                    .enumerate()
                    .rev()
                    .map(|pair| (pair, GateCenter::Left)),
            );
        for ((bond, gate), center) in passes {
            let svd = TruncSvdParams {
                chi_max: params.trunc.chi_max,
                target_trunc_err: params.trunc.target_trunc_err.map(|target| {
                    let norm = psi.norm::<T, _>(backend.as_ref());
                    assert!(
                        norm.is_finite(),
                        "tebd: state norm is not finite before a gate of step {step}",
                    );
                    target * norm.to_f64().unwrap_or(f64::NAN)
                }),
            };
            let res = psi.apply_two_site_gate(backend.as_ref(), bond, gate, &svd, center);
            trunc_err_sum = trunc_err_sum + res.error;
        }

        // The return pass leaves sites 1.. right-isometric, so site 0
        // carries the whole norm.
        debug_assert_eq!(*psi.canonical_form(), CanonicalForm::Mixed { center: 0 });
        let norm = psi.site(0).norm();
        if !(norm.is_finite() && norm > zero) {
            return Err(TebdError::NormBreakdown { step });
        }
        psi.site_mut(0).normalize();
        // `site_mut` resets the form to `Unknown`; without the re-tag the
        // next step's first gate would re-canonicalize the whole chain.
        psi.set_canonical_form(CanonicalForm::Mixed { center: 0 });
        records.push(TebdStepRecord {
            step,
            trunc_err_sum,
            norm,
            max_bond: psi.max_bond_dim(),
        });
    }
    Ok(records)
}

/// Validate `params` and return the half step length in `T::Real`.
fn validate_params<T: Scalar>(params: &TebdParams) -> Result<T::Real, TebdError> {
    let invalid = |detail| TebdError::InvalidParams { detail };
    let (TimeStep::Real(len) | TimeStep::Imaginary(len)) = params.time_step;
    if !(len.is_finite() && len > 0.0) {
        return Err(invalid("time step must be finite and positive"));
    }
    let (zero, one) = (T::Real::zero(), T::Real::one());
    if matches!(params.time_step, TimeStep::Real(_)) && T::from_real_imag(zero, one).im() == zero {
        return Err(invalid("a real-time step requires a complex scalar type"));
    }
    // A step far below the real type's range converts to zero and would make
    // every gate the identity; one far above it converts to infinity.
    let half = try_real_from_f64::<T>(len / 2.0)
        .filter(|h| *h > T::Real::zero())
        .ok_or(invalid(
            "half time step is not a positive finite value in the storage's real scalar type",
        ))?;
    validate_trunc_params(&params.trunc).map_err(invalid)?;
    Ok(half)
}
