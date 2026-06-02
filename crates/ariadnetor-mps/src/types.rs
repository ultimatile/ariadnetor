//! Core MPS/MPO data types.
//!
//! `Mps` / `Mpo` store a `Vec<Tensor<St, L, B>>` of joined-form site
//! tensors plus a cached `Arc<B>` backend. The Tier 1 ordering
//! invariant — every site's `layout().order()` matches
//! `backend.preferred_order()` — is enforced by the constructors
//! below via the doc-hidden [`LayoutOrderCheck`] trait (`pub` so the
//! `where`-bound on the constructor impl is satisfiable from
//! downstream crates without naming the trait), so downstream linalg
//! kernels never have to defensively align site memory order.

use std::sync::Arc;

use arnet::{
    BlockSparseLayout, BlockSparseStorage, ComputeBackend, DenseLayout, DenseStorage,
    NativeBackend, Scalar, Sector, Storage, StorageFor, Tensor, TensorLayout, TruncSvdParams,
};

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
    Partial { left_end: usize, right_start: usize },
    /// Single-site orthogonality center at `center`.
    /// 0..center left-canonical, center+1..N right-canonical.
    Mixed { center: usize },
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
    pub svd: TruncSvdParams,
    pub absorb: SvdAbsorb,
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
    pub error: T::Real,
}

/// Internal data container for MPS/MPO tensor chains.
#[derive(Debug, Clone)]
pub(crate) struct TensorChainData<St, L, B>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    pub(crate) sites: Vec<Tensor<St, L, B>>,
    pub(crate) backend: Arc<B>,
    pub(crate) canonical_form: CanonicalForm,
}

/// Matrix Product State — rank-3 tensor chain.
#[derive(Debug, Clone)]
pub struct Mps<St, L, B = NativeBackend>(pub(crate) TensorChainData<St, L, B>)
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    B: ComputeBackend;

/// Matrix Product Operator — rank-4 tensor chain.
#[derive(Debug, Clone)]
pub struct Mpo<St, L, B = NativeBackend>(pub(crate) TensorChainData<St, L, B>)
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    B: ComputeBackend;

// ============================================================================
// Per-flavor order check (crate-private)
//
// `TensorLayout` doesn't expose `order()` (only shape / storage_extent),
// so the Tier 1 order assertion routes through this trait. One impl per
// concrete (Storage, Layout) flavor compares the site's layout order against
// the backend's preferred order. The comparison is folded inside the trait
// (rather than returning a `MemoryOrder` for the caller to compare) so the
// layout-order type never appears in this crate's surface — keeping it off
// the umbrella's public API.
// ============================================================================

#[doc(hidden)]
pub trait LayoutOrderCheck<L>
where
    L: TensorLayout,
{
    /// Returns `true` if `layout`'s memory order matches `backend`'s
    /// preferred order.
    fn order_matches<B: ComputeBackend>(layout: &L, backend: &B) -> bool;
}

impl<T: Scalar> LayoutOrderCheck<DenseLayout> for DenseStorage<T> {
    fn order_matches<B: ComputeBackend>(layout: &DenseLayout, backend: &B) -> bool {
        layout.order() == backend.preferred_order()
    }
}

impl<T: Scalar, S: Sector> LayoutOrderCheck<BlockSparseLayout<S>> for BlockSparseStorage<T> {
    fn order_matches<B: ComputeBackend>(layout: &BlockSparseLayout<S>, backend: &B) -> bool {
        layout.order() == backend.preferred_order()
    }
}

// ============================================================================
// Generic constructors (Mps / Mpo). The order check is performed via
// `LayoutOrderCheck`, so calls like `Mps::from_sites(sites)` resolve
// uniquely without turbofish at the call site.
// ============================================================================

impl<St, L, B> Mps<St, L, B>
where
    St: Storage + StorageFor<L> + LayoutOrderCheck<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    /// Create an MPS from sites. The backend is derived from `sites[0]`'s
    /// cached backend Arc, and every site's order must match that
    /// backend's preferred order.
    ///
    /// # Panics
    ///
    /// Panics if `sites` is empty (use [`empty`](Self::empty) instead)
    /// or if any site's `layout().order()` differs from the derived
    /// backend's `preferred_order()`.
    pub fn from_sites(sites: Vec<Tensor<St, L, B>>) -> Self {
        assert!(
            !sites.is_empty(),
            "Mps::from_sites: pass at least one site, or use Mps::empty(backend) for an empty chain",
        );
        let backend = Arc::clone(sites[0].backend_arc());
        Self::with_backend(sites, backend)
    }

    /// Create an MPS from sites with an explicit backend (allows empty
    /// chains).
    pub fn with_backend(sites: Vec<Tensor<St, L, B>>, backend: Arc<B>) -> Self {
        assert_sites_match_backend_order::<St, L, B>(&sites, &backend, "Mps::with_backend");
        Self(TensorChainData {
            sites,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }

    /// Create an empty MPS anchored on the given backend.
    pub fn empty(backend: Arc<B>) -> Self {
        Self(TensorChainData {
            sites: Vec::new(),
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

impl<St, L, B> Mpo<St, L, B>
where
    St: Storage + StorageFor<L> + LayoutOrderCheck<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    /// Create an MPO from sites. See [`Mps::from_sites`] for semantics.
    pub fn from_sites(sites: Vec<Tensor<St, L, B>>) -> Self {
        assert!(
            !sites.is_empty(),
            "Mpo::from_sites: pass at least one site, or use Mpo::empty(backend) for an empty chain",
        );
        let backend = Arc::clone(sites[0].backend_arc());
        Self::with_backend(sites, backend)
    }

    /// Create an MPO from sites with an explicit backend.
    pub fn with_backend(sites: Vec<Tensor<St, L, B>>, backend: Arc<B>) -> Self {
        assert_sites_match_backend_order::<St, L, B>(&sites, &backend, "Mpo::with_backend");
        Self(TensorChainData {
            sites,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }

    /// Create an empty MPO anchored on the given backend.
    pub fn empty(backend: Arc<B>) -> Self {
        Self(TensorChainData {
            sites: Vec::new(),
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

fn assert_sites_match_backend_order<St, L, B>(
    sites: &[Tensor<St, L, B>],
    backend: &Arc<B>,
    ctx: &str,
) where
    St: Storage + StorageFor<L> + LayoutOrderCheck<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    let expected = backend.preferred_order();
    for (i, s) in sites.iter().enumerate() {
        assert!(
            St::order_matches(s.data().layout(), backend.as_ref()),
            "{ctx}: site {i} layout order does not match backend.preferred_order() ({expected:?})",
        );
        // The site's cached backend Arc need not be `Arc::ptr_eq` with the
        // chain backend, but its preferred_order must agree — otherwise
        // linalg kernels invoked via the site's backend would assume a
        // layout different from the one the chain enforced.
        let site_backend_order = s.backend().preferred_order();
        assert_eq!(
            site_backend_order, expected,
            "{ctx}: site {i} cached backend preferred_order ({site_backend_order:?}) does not match chain backend preferred_order ({expected:?})",
        );
    }
}
