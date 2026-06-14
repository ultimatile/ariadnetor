//! ARPACK-NG-backed smallest-eigenpair solver, feature-gated under
//! `arpack`. Targets the same Hermitian-operator problem as
//! [`super::lanczos::lanczos_smallest`] but delegates the iteration to
//! ARPACK, which has stronger convergence behavior on hard spectra.
//!
//! Type dispatch is sealed to `f32` / `f64` / `Complex<f32>` /
//! `Complex<f64>`. The `nev = 1` choice is inherited from the
//! `arpack-rs` `smallest_eigenpair_*` entry points; multi-eigenvalue
//! extraction is tracked as future work and will land as a separate
//! entry point once `arpack-rs` exposes a `nev > 1` API.
//!
//! The matvec adapter copies the read-side slice into a fresh `Dense`
//! per call, applies the operator, and copies the result back into the
//! ARPACK-managed buffer. The two allocations per matvec are a known
//! cost — they will be eliminated once `LinearOp` grows a
//! slice-in-place variant.

use arnet_core::Scalar;
use arnet_tensor::{DenseTensor, linear_combine};
use num_complex::{Complex32, Complex64};
use num_traits::{NumCast, One, Zero};

use super::lanczos::LinearOp;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// Tunable parameters for the ARPACK-backed solver.
///
/// Field semantics mirror `arpack::symmetric::Options` and
/// `arpack::arnoldi::Options`; the wrapper picks the right one per
/// scalar type.
#[derive(Debug, Clone)]
pub struct ArpackParams {
    /// Convergence tolerance — must be **strictly positive**. Used
    /// both as ARPACK's relative stopping criterion and as the
    /// wrapper's absolute threshold for `ArpackResult::converged`.
    /// The "tol = 0 means machine-epsilon default" sentinel that
    /// `arpack-rs` forwards is rejected at this layer because it
    /// would silently break the `converged` divergence indicator
    /// (`residual <= 0` is unreachable). Pass an explicit value
    /// (e.g. `1e-12` for `f64` precision targets, `1e-5` for `f32`).
    pub tol: f64,
    /// Maximum number of restart iterations.
    pub max_iter: usize,
    /// Krylov-subspace dimension. `None` selects ARPACK's default
    /// (driver-specific; see `arpack-rs` docs).
    pub ncv: Option<usize>,
}

impl Default for ArpackParams {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            max_iter: 300,
            ncv: None,
        }
    }
}

/// Output of [`arpack_smallest`].
#[derive(Debug, Clone)]
pub struct ArpackResult<T: Scalar> {
    /// Smallest (algebraic / real-part) eigenvalue. The imaginary
    /// part is dropped — for Hermitian operators it is numerically
    /// zero by construction.
    pub eigenvalue: T::Real,
    /// Corresponding eigenvector of length `dim`, unit-normalized.
    pub eigenvector: DenseTensor<T>,
    /// Number of restart iterations performed (ARPACK's `iparam[2]`
    /// writeback).
    pub iters: usize,
    /// Number of operator applications performed by ARPACK
    /// (`iparam[8]` writeback). The dominant cost term in
    /// DMRG-scale workloads.
    pub n_matvec: usize,
    /// True residual `||H psi - lambda psi||_2` recomputed by the
    /// wrapper from the returned pair.
    pub residual: T::Real,
    /// `true` iff `residual <= params.tol` interpreted as an
    /// **absolute** bound (cast to `T::Real`).
    ///
    /// ARPACK's internal stopping criterion is *relative*
    /// (`residual <= tol * |lambda|`), so an `Ok` return means
    /// ARPACK accepted the pair in its own sense — but that may not
    /// match what the caller meant by `params.tol`. This flag is the
    /// divergence indicator: `Ok + converged = true` means both
    /// agree the pair meets the requested precision; `Ok + converged
    /// = false` means ARPACK delivered the relative-tol stopping
    /// guarantee but the absolute residual still exceeds
    /// `params.tol`. The caller can then decide whether to accept or
    /// retry with a tighter `tol`.
    pub converged: bool,
}

/// Errors from the ARPACK-backed solver. Mirrors `arpack::Error`
/// without information loss; the wrapper does not collapse distinct
/// upstream codes.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ArpackError {
    /// Wrapper-side parameter validation or upstream `InvalidParam`.
    #[error("invalid parameter: {0}")]
    InvalidParam(&'static str),
    /// `*aupd_c` returned a non-recoverable info code.
    #[error("ARPACK *aupd returned info = {0}")]
    AupdFailed(i32),
    /// `*eupd_c` returned a non-zero info code.
    #[error("ARPACK *eupd returned info = {0}")]
    EupdFailed(i32),
    /// `*aupd_c` requested an unsupported `ido` value.
    #[error("ARPACK requested unsupported ido = {0}")]
    UnexpectedIdo(i32),
    /// `*aupd_c` returned `info = 1`: the iteration hit `max_iter`
    /// before the requested eigenpair converged. The iparam
    /// diagnostic counters are preserved.
    #[error(
        "ARPACK hit max_iter without convergence: iters = {iters}, \
         nconv = {nconv}, n_matvec = {n_matvec}"
    )]
    MaxIterReached {
        iters: usize,
        nconv: usize,
        n_matvec: usize,
    },
}

impl From<arpack::Error> for ArpackError {
    fn from(e: arpack::Error) -> Self {
        match e {
            arpack::Error::InvalidParam(m) => ArpackError::InvalidParam(m),
            arpack::Error::AupdFailed { info, .. } => ArpackError::AupdFailed(info),
            arpack::Error::EupdFailed { info, .. } => ArpackError::EupdFailed(info),
            arpack::Error::UnexpectedIdo(i) => ArpackError::UnexpectedIdo(i),
            arpack::Error::MaxIterReached {
                iters,
                nconv,
                n_matvec,
            } => ArpackError::MaxIterReached {
                iters,
                nconv,
                n_matvec,
            },
            // `arpack::Error` is `#[non_exhaustive]`; future variants
            // round-trip through this catch-all until explicitly
            // handled.
            _ => ArpackError::InvalidParam("unrecognized arpack error variant"),
        }
    }
}

/// Sealed trait selecting the appropriate `arpack-rs` driver per
/// scalar type. Implemented for `f32`, `f64`, `Complex<f32>`,
/// `Complex<f64>`; not extensible by downstream crates.
pub trait ArpackScalar: Scalar + sealed::Sealed {
    /// Drive `arpack-rs`'s smallest-eigenpair entry point for this
    /// scalar type with the wrapper's `matvec` closure.
    fn solve(
        n: usize,
        matvec: &mut dyn FnMut(&[Self], &mut [Self]),
        params: &ArpackParams,
    ) -> Result<arpack::EigSolution<Self>, arpack::Error>;
}

impl ArpackScalar for f32 {
    fn solve(
        n: usize,
        matvec: &mut dyn FnMut(&[Self], &mut [Self]),
        params: &ArpackParams,
    ) -> Result<arpack::EigSolution<Self>, arpack::Error> {
        let opts = arpack::symmetric::Options {
            tol: params.tol,
            max_iter: params.max_iter,
            ncv: params.ncv,
        };
        arpack::symmetric::smallest_eigenpair_f32(n, matvec, &opts)
    }
}

impl ArpackScalar for f64 {
    fn solve(
        n: usize,
        matvec: &mut dyn FnMut(&[Self], &mut [Self]),
        params: &ArpackParams,
    ) -> Result<arpack::EigSolution<Self>, arpack::Error> {
        let opts = arpack::symmetric::Options {
            tol: params.tol,
            max_iter: params.max_iter,
            ncv: params.ncv,
        };
        arpack::symmetric::smallest_eigenpair_f64(n, matvec, &opts)
    }
}

impl ArpackScalar for Complex32 {
    fn solve(
        n: usize,
        matvec: &mut dyn FnMut(&[Self], &mut [Self]),
        params: &ArpackParams,
    ) -> Result<arpack::EigSolution<Self>, arpack::Error> {
        let opts = arpack::arnoldi::Options {
            tol: params.tol,
            max_iter: params.max_iter,
            ncv: params.ncv,
        };
        arpack::arnoldi::smallest_eigenpair_c32(n, matvec, &opts)
    }
}

impl ArpackScalar for Complex64 {
    fn solve(
        n: usize,
        matvec: &mut dyn FnMut(&[Self], &mut [Self]),
        params: &ArpackParams,
    ) -> Result<arpack::EigSolution<Self>, arpack::Error> {
        let opts = arpack::arnoldi::Options {
            tol: params.tol,
            max_iter: params.max_iter,
            ncv: params.ncv,
        };
        arpack::arnoldi::smallest_eigenpair_c64(n, matvec, &opts)
    }
}

/// Smallest eigenpair of a Hermitian linear operator via ARPACK.
///
/// `dim` is the length of the flat vector the operator acts on. The
/// closure / `LinearOp` is invoked many times per call (one per
/// ARPACK reverse-communication step).
///
/// # Numerical preconditions
///
/// - `params.tol` is honored down to `T::Real::epsilon()`. Asking for
///   a tolerance below working precision (e.g. `1e-10` for `f32`)
///   cannot be satisfied; `converged` will reflect achievable
///   precision rather than the requested one.
/// - The operator must be Hermitian. For complex operators ARPACK's
///   smallest-real-part selector returns the smallest real eigenvalue,
///   matching the symmetric driver's algebraic-smallest behavior.
pub fn arpack_smallest<T, Op>(
    op: &Op,
    dim: usize,
    params: &ArpackParams,
) -> Result<ArpackResult<T>, ArpackError>
where
    T: ArpackScalar,
    T::Real: Scalar<Real = T::Real>,
    Op: LinearOp<T>,
{
    assert!(dim >= 1, "arpack_smallest: dim must be >= 1");
    if !params.tol.is_finite() || params.tol <= 0.0 {
        return Err(ArpackError::InvalidParam(
            "params.tol must be finite and strictly positive",
        ));
    }

    // Drive ARPACK with a closure that adapts ARPACK's slice-in /
    // slice-out matvec interface to the `LinearOp` Dense-in / Dense-
    // out interface. Two `Vec` allocations per matvec is a known cost
    // and will be eliminated when `LinearOp` grows a slice variant.
    let solution = T::solve(
        dim,
        &mut |x_slice, y_slice| {
            let x_dense = DenseTensor::from_raw_parts(x_slice.to_vec(), vec![dim]);
            let y_dense = op.apply(&x_dense);
            assert_eq!(
                y_dense.shape(),
                &[dim],
                "LinearOp::apply must return a rank-1 tensor of shape [dim]",
            );
            y_slice.copy_from_slice(y_dense.data_slice());
        },
        params,
    )?;

    let eigenvalue = solution.eigenvalue.re();
    let mut eigenvector = DenseTensor::from_raw_parts(solution.eigenvector, vec![dim]);
    // ARPACK normalizes its output; pass through `normalize` as a
    // safety belt against precision drift in the down-cast.
    eigenvector.normalize();

    // True residual ||H psi - lambda psi||_2.
    let h_psi_raw = op.apply(&eigenvector);
    assert_eq!(
        h_psi_raw.shape(),
        &[dim],
        "LinearOp::apply must return a rank-1 tensor of shape [dim]",
    );
    // The user operator is not required to declare an output `order()`
    // matching ARPACK's eigenvector. Normalize against
    // `eigenvector.order()` so `linear_combine` does not reject mixed-
    // order inputs; for 1D data this is metadata-only.
    let h_psi = if h_psi_raw.order() == eigenvector.order() {
        h_psi_raw
    } else {
        h_psi_raw.reordered(eigenvector.order())
    };
    let lambda_t = T::from_real_imag(eigenvalue, T::Real::zero());
    let neg_lambda = lambda_t.scale_real(-T::Real::one());
    let residual_vec = linear_combine(&[&h_psi, &eigenvector], &[T::one(), neg_lambda])
        .expect("linear_combine on rank-1 tensors of matching shape");
    let residual = residual_vec.norm();

    let tol_real: T::Real = <T::Real as NumCast>::from(params.tol)
        .unwrap_or_else(|| panic!("tol {} not representable in T::Real", params.tol));
    let converged = residual <= tol_real;

    Ok(ArpackResult {
        eigenvalue,
        eigenvector,
        iters: solution.iters,
        n_matvec: solution.n_matvec,
        residual,
        converged,
    })
}
