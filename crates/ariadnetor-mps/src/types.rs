//! Core MPS/MPO data types.
//!
//! `Mps` / `Mpo` store a `Vec<Tensor<St, L>>` of joined-form site
//! tensors. They carry no compute backend: every operation receives the
//! backend explicitly at its call site (the call-site-supply design), so
//! the chain types are backend-agnostic data containers.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::TruncSvdParams;
use ariadnetor_tensor::{Storage, StorageFor, Tensor, TensorLayout};
use serde::{Deserialize, Serialize};

/// Canonical form of a tensor chain.
///
/// `Serialize` / `Deserialize` round-trip the form verbatim for MPS
/// serialization. No stricter-than-type validation is imposed on load: the
/// in-memory type does not enforce its documented positional invariants (the
/// setter stores arbitrary values), so rejecting an out-of-order `Partial`
/// would break the lossless round-trip of legally-representable states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanonicalForm {
    /// No canonicalization guarantees.
    Unknown,
    /// All sites are left-isometric (A-form). The state is normalized.
    Left,
    /// All sites are right-isometric (B-form). The state is normalized.
    Right,
    /// 0..left_end left-canonical, right_start..N right-canonical.
    /// Non-canonical region spans multiple sites (right_start - left_end > 1).
    Partial {
        /// One past the last left-canonical site: sites `0..left_end` are left-isometric.
        left_end: usize,
        /// First right-canonical site: sites `right_start..N` are right-isometric.
        right_start: usize,
    },
    /// Single-site orthogonality center at `center`.
    /// 0..center left-canonical, center+1..N right-canonical.
    Mixed {
        /// Index of the single orthogonality-center site.
        center: usize,
    },
}

/// How singular values are distributed during truncation sweeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvdAbsorb {
    /// S stays at the current site (against sweep direction).
    Left,
    /// S accompanies the sweep direction (standard algorithm).
    #[default]
    Right,
    /// √S applied to both sides (symmetric split).
    /// Result is not mixed-canonical.
    Both,
}

/// Parameters for MPS/MPO truncation.
#[derive(Debug, Clone)]
pub struct TruncateParams {
    /// Truncated-SVD parameters (bond cap and singular-value cutoff).
    pub svd: TruncSvdParams,
    /// Where the singular values are absorbed after each split.
    pub absorb: SvdAbsorb,
    /// Optional target orthogonality center for the finishing pass;
    /// `None` leaves the center at the sweep's natural end site.
    pub center: Option<usize>,
}

impl From<TruncSvdParams> for TruncateParams {
    fn from(svd: TruncSvdParams) -> Self {
        Self {
            svd,
            absorb: SvdAbsorb::default(),
            center: None,
        }
    }
}

/// Initial-guess generator for [`ApplyMethod::Variational`]. Its truncation
/// to `chi_max` sets the fixed bond the variational sweeps then refine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariationalInit {
    /// Seed from the Stoudenmire-White zip-up result.
    ZipUp,
    /// Seed from the density-matrix result.
    DensityMatrix,
}

/// Algorithm used by [`apply_with_method`](super::dispatch::apply_with_method)
/// to multiply an MPO into an MPS.
///
/// `Eq` is intentionally not derived: [`Variational`](ApplyMethod::Variational)
/// carries an `f64` tolerance, which is `PartialEq` but not `Eq`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApplyMethod {
    /// Per-site MPO·MPS product with a streaming forward QR sweep
    /// (or truncated SVD when `forward_cap = Some(k)` and the natural
    /// per-site forward rank exceeds `k * chi_max`). When the caller
    /// passes `Some(TruncateParams)`, the forward sweep is followed by a
    /// standard `canonicalize` + `truncate` finishing pass that honors
    /// every `SvdAbsorb` variant and any in-range `params.center`. When
    /// `params` is `None`, no canonicalization or truncation runs after
    /// the forward sweep, and the result is left in `Mixed { center: n
    /// - 1 }`.
    ///
    /// `forward_cap = None` is lossless streaming naive: the forward
    /// branch is QR-only, and the final state matches a hypothetical
    /// materialize-then-compress baseline modulo QR sign and
    /// floating-point roundoff. The streaming forward keeps peak per-site
    /// bond bounded by the QR ranks rather than the fully inflated
    /// `w_R * chi_R`.
    StreamingNaive {
        /// Cap on the per-site forward rank; `None` keeps the forward
        /// sweep lossless (QR-only), `Some(k)` switches to truncated SVD
        /// once the natural rank exceeds `k * chi_max`.
        forward_cap: Option<std::num::NonZeroUsize>,
    },
    /// Stoudenmire-White single-pass zip-up algorithm: right-canonicalize
    /// the input, then a forward sweep where each site's SVD bond is
    /// truncated directly to `chi_max`, with no separate backward
    /// truncation pass. Consumes `params.svd` (`chi_max`,
    /// `target_trunc_err`); `params = None` keeps full SVD rank at every
    /// bond (lossless). `params.absorb` and `params.center` are not
    /// consulted, because zip-up intrinsically carries the singular values
    /// rightward and ends with the orthogonality center at the last site.
    /// Distinct from
    /// [`StreamingNaive`](ApplyMethod::StreamingNaive), which keeps a full
    /// `chi_max` backward sweep.
    ZipUp,
    /// Density-matrix compression: materialize the untruncated product
    /// `φ = Wψ`, accumulate the `⟨φ|φ⟩` right environment, then a single
    /// left-to-right sweep that forms the reduced density matrix
    /// `ρ = θ · R · θ†` at each bond and keeps its largest `chi_max`
    /// eigenvectors. Because `ρ` is Hermitian positive-semidefinite, its
    /// dominant eigenvectors are its dominant left singular vectors, so the
    /// truncation reuses the SVD (for a PSD matrix the SVD coincides with the
    /// eigendecomposition). Only `params.svd.chi_max` is consulted; `params =
    /// None` (or `chi_max = None`) keeps full rank at every bond (lossless).
    /// `params.svd.target_trunc_err`, `params.absorb`, and `params.center` are
    /// not consulted — the sweep carries the orthogonality center to the last
    /// site like [`ZipUp`](ApplyMethod::ZipUp), and a `target_trunc_err` cutoff
    /// in the caller's Schmidt-value domain would need a dedicated truncated
    /// eigensolver (forming `ρ` moves truncation into the squared-eigenvalue
    /// domain). Forming `ρ` also squares the Schmidt spectrum, so this method
    /// carries the standard density-matrix `√ε` accuracy floor on very small
    /// Schmidt values relative to [`ZipUp`](ApplyMethod::ZipUp).
    DensityMatrix,
    /// Variational (fit) compression: minimize `‖φ − Wψ‖` at the fixed bond
    /// set by the initial guess, via single-site DMRG-style sweeps whose local
    /// update replaces the center tensor with the `⟨φ|W|ψ⟩` environment
    /// projection `P_j = L(j)·W_j·ψ_j·R(j+1)`. Because the off-center sites are
    /// isometric, `P_j` is the exact per-site minimizer of `‖φ − Wψ‖²`.
    ///
    /// Seeded from `init` (zip-up or density-matrix, truncated to
    /// `params.svd.chi_max`), then swept until the relative change of the
    /// center overlap `‖P_center‖²` between successive cycles is at or below
    /// `tol`, or `max_sweeps` full L→R + R→L cycles have run. Only
    /// `params.svd.chi_max` is consulted (it sizes the seed and thus the fixed
    /// bond); `params.absorb`, `params.center`, and `params.target_trunc_err`
    /// are not — the bond is held fixed at the seed's. Like the other
    /// compression methods the sweep maintains a single orthogonality center (a
    /// single-site gauge); it merely ends at a different site (see below).
    ///
    /// Host-pinned: this method builds on the host-resident
    /// [`BraketEnvs`](crate::BraketEnvs) primitive, so — like DMRG — the whole
    /// computation runs on the `Host` substrate and the `backend` passed to
    /// [`apply_with_method`](super::dispatch::apply_with_method) is **not
    /// consulted**. The result is left in `Mixed { center: 0 }` (the R→L
    /// half-sweep runs last, matching the DMRG sweep convention), unlike
    /// [`ZipUp`](ApplyMethod::ZipUp) / [`DensityMatrix`](ApplyMethod::DensityMatrix)
    /// which end at `center: n − 1`.
    Variational {
        /// Initial-guess generator; its `chi_max` truncation fixes the bond
        /// the sweeps refine.
        init: VariationalInit,
        /// Maximum number of full L→R + R→L sweep cycles.
        max_sweeps: usize,
        /// Relative-change convergence tolerance on the center overlap
        /// `‖P_center‖²` between successive cycles.
        tol: f64,
    },
}

impl Default for ApplyMethod {
    fn default() -> Self {
        Self::StreamingNaive { forward_cap: None }
    }
}

/// Result of a truncation operation.
#[derive(Debug, Clone)]
pub struct TruncResult<T: Scalar> {
    /// Truncation error: the discarded singular-value weight accumulated
    /// over the operation.
    pub error: T::Real,
}

/// Internal data container for MPS/MPO tensor chains.
#[derive(Debug, Clone)]
pub(crate) struct TensorChainData<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    pub(crate) sites: Vec<Tensor<St, L>>,
    pub(crate) canonical_form: CanonicalForm,
}

/// Matrix Product State — rank-3 tensor chain.
#[derive(Debug, Clone)]
pub struct Mps<St, L>(pub(crate) TensorChainData<St, L>)
where
    St: Storage + StorageFor<L>,
    L: TensorLayout;

/// Matrix Product Operator — rank-4 tensor chain.
#[derive(Debug, Clone)]
pub struct Mpo<St, L>(pub(crate) TensorChainData<St, L>)
where
    St: Storage + StorageFor<L>,
    L: TensorLayout;

// ============================================================================
// Generic constructors (Mps / Mpo). The chain carries no backend; sites
// are stored as-is and the supplied backend reaches each operation at its
// call site.
// ============================================================================

impl<St, L> Mps<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Create an MPS from a non-empty list of sites.
    ///
    /// # Panics
    ///
    /// Panics if `sites` is empty (use [`empty`](Self::empty) instead).
    pub fn from_sites(sites: Vec<Tensor<St, L>>) -> Self {
        assert!(
            !sites.is_empty(),
            "Mps::from_sites: pass at least one site, or use Mps::empty() for an empty chain",
        );
        Self(TensorChainData {
            sites,
            canonical_form: CanonicalForm::Unknown,
        })
    }

    /// Create an empty MPS.
    pub fn empty() -> Self {
        Self(TensorChainData {
            sites: Vec::new(),
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

impl<St, L> Mpo<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Create an MPO from a non-empty list of sites. See
    /// [`Mps::from_sites`] for semantics.
    pub fn from_sites(sites: Vec<Tensor<St, L>>) -> Self {
        assert!(
            !sites.is_empty(),
            "Mpo::from_sites: pass at least one site, or use Mpo::empty() for an empty chain",
        );
        Self(TensorChainData {
            sites,
            canonical_form: CanonicalForm::Unknown,
        })
    }

    /// Create an empty MPO.
    pub fn empty() -> Self {
        Self(TensorChainData {
            sites: Vec::new(),
            canonical_form: CanonicalForm::Unknown,
        })
    }
}
