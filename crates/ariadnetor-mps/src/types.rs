//! Core MPS/MPO data types.
//!
//! `Mps` / `Mpo` store a `Vec<Tensor<St, L>>` of joined-form site
//! tensors. They carry no compute backend: every operation receives the
//! backend explicitly at its call site (the call-site-supply design), so
//! the chain types are backend-agnostic data containers.

use arnet_core::Scalar;
use arnet_linalg::TruncSvdParams;
use arnet_tensor::{Storage, StorageFor, Tensor, TensorLayout};

/// Canonical form of a tensor chain.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Algorithm used by [`apply_with_method`](super::dispatch::apply_with_method)
/// to multiply an MPO into an MPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Reserved for the literature Stoudenmire-White single-pass
    /// interleaved-truncation algorithm — a forward sweep where the SVD bond
    /// is bounded by `chi_max` at each site, with no separate backward
    /// truncation pass. Not yet implemented; selecting this variant panics
    /// at dispatch time. Distinct from
    /// [`StreamingNaive`](ApplyMethod::StreamingNaive), which keeps a full
    /// `chi_max` backward sweep.
    ZipUp,
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
