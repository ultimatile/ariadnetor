//! Core MPS/MPO data types.
//!
//! `Mps` / `Mpo` store a `Vec<Tensor<St, L, B>>` of joined-form site
//! tensors plus a cached `Arc<B>` backend. The Tier 1 ordering
//! invariant — every site's `layout().order()` matches
//! `backend.preferred_order()` — is enforced by the per-flavor
//! constructors below, so downstream linalg kernels never have to
//! defensively align site memory order.

use std::sync::Arc;

use arnet::{
    BlockSparseLayout, BlockSparseStorage, ComputeBackend, DenseLayout, DenseStorage,
    NativeBackend, Scalar, Sector, Tensor, TruncSvdParams,
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
///
/// `Right` (default) and `Left` both produce mixed-canonical form after the
/// three-sweep truncation. They differ in which intermediate tensors carry S,
/// but the final isometry structure is the same. `Both` distributes √S to
/// both sides, so the result is not mixed-canonical.
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
    /// SVD truncation parameters (chi_max, target_trunc_err).
    pub svd: TruncSvdParams,
    /// How to absorb singular values at each step.
    pub absorb: SvdAbsorb,
    /// Target orthogonality center for auto-canonicalization.
    /// Used when the chain is not already in Mixed canonical form.
    /// Defaults to 0 if None.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApplyMethod {
    /// Materialize the exact MPO·MPS product, then canonicalize and truncate.
    #[default]
    Naive,
    /// Interleave contraction with QR / truncated SVD so the inflated bond
    /// dimension never appears simultaneously across all sites.
    ZipUp,
}

/// Result of a truncation operation.
#[derive(Debug, Clone)]
pub struct TruncResult<T: Scalar> {
    /// Total truncation error (Frobenius norm of discarded SVs).
    pub error: T::Real,
}

/// Internal data container for MPS/MPO tensor chains.
///
/// Holds the joined-form site tensors, a shared backend, and canonical
/// form state. This type is `pub(crate)` — users interact through
/// `Mps` / `Mpo` newtypes and the `TensorChain` trait.
#[derive(Debug, Clone)]
pub(crate) struct TensorChainData<St, L, B>
where
    St: arnet::Storage + arnet::StorageFor<L>,
    L: arnet::TensorLayout,
    B: ComputeBackend,
{
    pub(crate) sites: Vec<Tensor<St, L, B>>,
    pub(crate) backend: Arc<B>,
    pub(crate) canonical_form: CanonicalForm,
}

/// Matrix Product State — rank-3 tensor chain.
///
/// Each site tensor has shape `(χ_L, d, χ_R)`:
/// - mode 0: left bond dimension
/// - mode 1: physical dimension
/// - mode 2: right bond dimension
///
/// Edge tensors use dummy bonds (dim 1) to maintain rank 3.
#[derive(Debug, Clone)]
pub struct Mps<St, L, B = NativeBackend>(pub(crate) TensorChainData<St, L, B>)
where
    St: arnet::Storage + arnet::StorageFor<L>,
    L: arnet::TensorLayout,
    B: ComputeBackend;

/// Matrix Product Operator — rank-4 tensor chain.
///
/// Each site tensor has shape `(χ_L, d_ket, d_bra, χ_R)`:
/// - mode 0: left bond dimension
/// - mode 1: ket physical dimension
/// - mode 2: bra physical dimension
/// - mode 3: right bond dimension
///
/// Edge tensors use dummy bonds (dim 1) to maintain rank 4.
#[derive(Debug, Clone)]
pub struct Mpo<St, L, B = NativeBackend>(pub(crate) TensorChainData<St, L, B>)
where
    St: arnet::Storage + arnet::StorageFor<L>,
    L: arnet::TensorLayout,
    B: ComputeBackend;

// ============================================================================
// Dense Mps/Mpo constructors with Tier 1 order assertions
//
// The order check is per-flavor because `TensorLayout` does not expose
// `order()` (only shape / storage_extent). Concrete `DenseLayout` and
// `BlockSparseLayout<S>` impls have the accessor, so the assertion
// lives in their dedicated impl blocks.
// ============================================================================

impl<T: Scalar, B: ComputeBackend> Mps<DenseStorage<T>, DenseLayout, B> {
    /// Create a Dense MPS from sites. The backend is derived from
    /// `sites[0]`'s cached backend Arc, and every site's order must
    /// already match that backend's preferred order.
    ///
    /// # Panics
    ///
    /// Panics if `sites` is empty (use [`empty`](Self::empty) instead)
    /// or if any site's `layout().order()` differs from the derived
    /// backend's `preferred_order()`.
    pub fn from_sites(sites: Vec<Tensor<DenseStorage<T>, DenseLayout, B>>) -> Self {
        assert!(
            !sites.is_empty(),
            "Mps::from_sites: pass at least one site, or use Mps::empty(backend) for an empty chain",
        );
        let backend = Arc::clone(sites[0].backend_arc());
        Self::with_backend(sites, backend)
    }

    /// Create a Dense MPS from sites with an explicit backend (allows
    /// empty chains).
    ///
    /// # Panics
    ///
    /// Panics if any site's `layout().order()` differs from
    /// `backend.preferred_order()`.
    pub fn with_backend(
        sites: Vec<Tensor<DenseStorage<T>, DenseLayout, B>>,
        backend: Arc<B>,
    ) -> Self {
        assert_dense_sites_match_backend_order(&sites, &backend, "Mps::with_backend");
        Self(TensorChainData {
            sites,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }

    /// Create an empty Dense MPS anchored on the given backend.
    pub fn empty(backend: Arc<B>) -> Self {
        Self(TensorChainData {
            sites: Vec::new(),
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

impl<T: Scalar, B: ComputeBackend> Mpo<DenseStorage<T>, DenseLayout, B> {
    /// Create a Dense MPO from sites. See [`Mps::from_sites`] for
    /// semantics.
    pub fn from_sites(sites: Vec<Tensor<DenseStorage<T>, DenseLayout, B>>) -> Self {
        assert!(
            !sites.is_empty(),
            "Mpo::from_sites: pass at least one site, or use Mpo::empty(backend) for an empty chain",
        );
        let backend = Arc::clone(sites[0].backend_arc());
        Self::with_backend(sites, backend)
    }

    /// Create a Dense MPO from sites with an explicit backend.
    pub fn with_backend(
        sites: Vec<Tensor<DenseStorage<T>, DenseLayout, B>>,
        backend: Arc<B>,
    ) -> Self {
        assert_dense_sites_match_backend_order(&sites, &backend, "Mpo::with_backend");
        Self(TensorChainData {
            sites,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }

    /// Create an empty Dense MPO anchored on the given backend.
    pub fn empty(backend: Arc<B>) -> Self {
        Self(TensorChainData {
            sites: Vec::new(),
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

// ============================================================================
// BlockSparse Mps/Mpo constructors with Tier 1 order assertions
// ============================================================================

impl<T: Scalar, S: Sector, B: ComputeBackend> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B> {
    /// Create a BlockSparse MPS from sites.
    pub fn from_sites(sites: Vec<Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, B>>) -> Self {
        assert!(
            !sites.is_empty(),
            "Mps::from_sites: pass at least one site, or use Mps::empty(backend) for an empty chain",
        );
        let backend = Arc::clone(sites[0].backend_arc());
        Self::with_backend(sites, backend)
    }

    /// Create a BlockSparse MPS with an explicit backend.
    pub fn with_backend(
        sites: Vec<Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, B>>,
        backend: Arc<B>,
    ) -> Self {
        assert_bsp_sites_match_backend_order(&sites, &backend, "Mps::with_backend");
        Self(TensorChainData {
            sites,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }

    /// Create an empty BlockSparse MPS anchored on the given backend.
    pub fn empty(backend: Arc<B>) -> Self {
        Self(TensorChainData {
            sites: Vec::new(),
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

impl<T: Scalar, S: Sector, B: ComputeBackend> Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>, B> {
    /// Create a BlockSparse MPO from sites.
    pub fn from_sites(sites: Vec<Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, B>>) -> Self {
        assert!(
            !sites.is_empty(),
            "Mpo::from_sites: pass at least one site, or use Mpo::empty(backend) for an empty chain",
        );
        let backend = Arc::clone(sites[0].backend_arc());
        Self::with_backend(sites, backend)
    }

    /// Create a BlockSparse MPO with an explicit backend.
    pub fn with_backend(
        sites: Vec<Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, B>>,
        backend: Arc<B>,
    ) -> Self {
        assert_bsp_sites_match_backend_order(&sites, &backend, "Mpo::with_backend");
        Self(TensorChainData {
            sites,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }

    /// Create an empty BlockSparse MPO anchored on the given backend.
    pub fn empty(backend: Arc<B>) -> Self {
        Self(TensorChainData {
            sites: Vec::new(),
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

// ============================================================================
// Tier 1 helpers — per-site order check against the chain's backend.
// ============================================================================

fn assert_dense_sites_match_backend_order<T: Scalar, B: ComputeBackend>(
    sites: &[Tensor<DenseStorage<T>, DenseLayout, B>],
    backend: &Arc<B>,
    ctx: &str,
) {
    let expected = backend.preferred_order();
    for (i, s) in sites.iter().enumerate() {
        let got = s.data().layout().order();
        assert_eq!(
            got, expected,
            "{ctx}: site {i} layout order ({got:?}) does not match backend.preferred_order() ({expected:?})",
        );
    }
}

fn assert_bsp_sites_match_backend_order<T: Scalar, S: Sector, B: ComputeBackend>(
    sites: &[Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, B>],
    backend: &Arc<B>,
    ctx: &str,
) {
    let expected = backend.preferred_order();
    for (i, s) in sites.iter().enumerate() {
        let got = s.data().layout().order();
        assert_eq!(
            got, expected,
            "{ctx}: site {i} layout order ({got:?}) does not match backend.preferred_order() ({expected:?})",
        );
    }
}
