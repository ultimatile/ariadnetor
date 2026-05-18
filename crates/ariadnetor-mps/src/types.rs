//! Core MPS / MPO data types.
//!
//! Two parameterizations coexist:
//!
//! - [`Mps<St, L, B>`] / [`Mpo<St, L, B>`] hold sites as
//!   [`TensorData<St, L>`](arnet_tensor::TensorData) over a
//!   [`Storage`](arnet_tensor::Storage) +
//!   [`TensorLayout`](arnet_tensor::TensorLayout) pair.
//! - [`MpsRepr<R, B>`] / [`MpoRepr<R, B>`] hold sites as
//!   `R: TensorRepr` (i.e. [`Dense<T>`](arnet_tensor::Dense) or
//!   [`BlockSparse<T, S>`](arnet_tensor::BlockSparse)).

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::TruncSvdParams;
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, Storage, StorageFor, TensorData, TensorLayout, TensorRepr};

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
///
/// Both methods produce the same exact state in the no-truncation limit and
/// agree element-wise when `chi_max` is at least the bond dimension of the
/// untruncated product. They differ in cost and in the truncation behavior
/// when `chi_max` is binding:
///
/// - [`Naive`](Self::Naive) (default) materializes the inflated bond
///   dimension `w * χ` across all sites, then runs a global canonicalize +
///   truncate sweep. The per-cut SVD sees the full environment, so for a
///   given `chi_max` the truncation is Eckart-Young optimal but the peak
///   memory and contraction cost scale with the inflated bonds.
/// - [`ZipUp`](Self::ZipUp) interleaves contraction and compression so the
///   inflated bonds never appear simultaneously. Each per-site SVD is taken
///   before the right environment is fully resolved, so the truncation is
///   greedy rather than globally optimal — accuracy at fixed `chi_max` is
///   typically a bit lower than naive, but cost is much lower for large
///   MPO/MPS.
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

// ============================================================================
// `R: TensorRepr` form — `MpsRepr` / `MpoRepr`
// ============================================================================

/// Internal data container for `R: TensorRepr` MPS / MPO tensor chains.
///
/// Backs [`MpsRepr`] / [`MpoRepr`].
#[derive(Debug, Clone)]
pub(crate) struct TensorChainDataRepr<R, B: ComputeBackend = NativeBackend> {
    pub(crate) storages: Vec<R>,
    pub(crate) backend: Arc<B>,
    pub(crate) canonical_form: CanonicalForm,
}

/// Matrix Product State whose sites are `R: TensorRepr`.
///
/// Counterpart of [`Mps`](crate::Mps), which holds sites as
/// [`TensorData`](arnet_tensor::TensorData).
#[derive(Debug, Clone)]
pub struct MpsRepr<R = Dense<f64>, B: ComputeBackend = NativeBackend>(
    pub(crate) TensorChainDataRepr<R, B>,
);

/// Matrix Product Operator whose sites are `R: TensorRepr`.
///
/// Counterpart of [`Mpo`](crate::Mpo), which holds sites as
/// [`TensorData`](arnet_tensor::TensorData).
#[derive(Debug, Clone)]
pub struct MpoRepr<R = Dense<f64>, B: ComputeBackend = NativeBackend>(
    pub(crate) TensorChainDataRepr<R, B>,
);

impl<R: TensorRepr, B: ComputeBackend> MpsRepr<R, B> {
    /// Create an MPS from raw site storages and an explicit backend.
    ///
    /// Each storage should have rank 3 with shape `(χ_L, d, χ_R)`.
    /// The canonical form is initially `Unknown`.
    pub fn with_backend(storages: Vec<R>, backend: Arc<B>) -> Self {
        Self(TensorChainDataRepr {
            storages,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

impl<R: TensorRepr, B: ComputeBackend> MpoRepr<R, B> {
    /// Create an MPO from raw site storages and an explicit backend.
    ///
    /// Each storage should have rank 4 with shape `(χ_L, d_ket, d_bra, χ_R)`.
    /// The canonical form is initially `Unknown`.
    pub fn with_backend(storages: Vec<R>, backend: Arc<B>) -> Self {
        Self(TensorChainDataRepr {
            storages,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

impl<R: TensorRepr> MpsRepr<R, NativeBackend> {
    /// Create an MPS from raw site storages using the default NativeBackend.
    pub fn from_storages(storages: Vec<R>) -> Self {
        Self::with_backend(storages, NativeBackend::shared())
    }
}

impl<R: TensorRepr> MpoRepr<R, NativeBackend> {
    /// Create an MPO from raw site storages using the default NativeBackend.
    pub fn from_storages(storages: Vec<R>) -> Self {
        Self::with_backend(storages, NativeBackend::shared())
    }
}

// ============================================================================
// `TensorData<St, L>` form — `Mps` / `Mpo`
// ============================================================================

/// Internal data container for tensor chains whose sites are
/// [`TensorData<St, L>`](arnet_tensor::TensorData) over
/// `St: Storage + StorageFor<L>` and `L: TensorLayout`. The backend
/// is held once at the chain level (not per site).
pub(crate) struct TensorChainData<St, L, B: ComputeBackend = NativeBackend>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    pub(crate) sites: Vec<TensorData<St, L>>,
    pub(crate) backend: Arc<B>,
    pub(crate) canonical_form: CanonicalForm,
}

impl<St, L, B: ComputeBackend> Clone for TensorChainData<St, L, B>
where
    St: Storage + StorageFor<L> + Clone,
    L: TensorLayout + Clone,
{
    fn clone(&self) -> Self {
        Self {
            sites: self.sites.clone(),
            backend: Arc::clone(&self.backend),
            canonical_form: self.canonical_form.clone(),
        }
    }
}

/// Matrix Product State — rank-3 tensor chain.
///
/// Each site is a [`TensorData<St, L>`] with shape `(χ_L, d, χ_R)`:
/// - mode 0: left bond dimension
/// - mode 1: physical dimension
/// - mode 2: right bond dimension
///
/// Edge tensors use dummy bonds (dim 1) to maintain rank 3.
pub struct Mps<St, L, B: ComputeBackend = NativeBackend>(pub(crate) TensorChainData<St, L, B>)
where
    St: Storage + StorageFor<L>,
    L: TensorLayout;

impl<St, L, B: ComputeBackend> Clone for Mps<St, L, B>
where
    St: Storage + StorageFor<L> + Clone,
    L: TensorLayout + Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// Matrix Product Operator — rank-4 tensor chain.
///
/// Each site is a [`TensorData<St, L>`] with shape
/// `(χ_L, d_ket, d_bra, χ_R)`:
/// - mode 0: left bond dimension
/// - mode 1: ket physical dimension
/// - mode 2: bra physical dimension
/// - mode 3: right bond dimension
///
/// Edge tensors use dummy bonds (dim 1) to maintain rank 4.
pub struct Mpo<St, L, B: ComputeBackend = NativeBackend>(pub(crate) TensorChainData<St, L, B>)
where
    St: Storage + StorageFor<L>,
    L: TensorLayout;

impl<St, L, B: ComputeBackend> Clone for Mpo<St, L, B>
where
    St: Storage + StorageFor<L> + Clone,
    L: TensorLayout + Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

// ----------------------------------------------------------------------------
// Constructors (storage-agnostic, any backend)
// ----------------------------------------------------------------------------

impl<St, L, B: ComputeBackend> Mps<St, L, B>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Create an MPS from raw site `TensorData` and an explicit backend.
    ///
    /// Each site should have rank 3 with shape `(χ_L, d, χ_R)`. The
    /// canonical form is initially [`CanonicalForm::Unknown`].
    pub fn with_backend(sites: Vec<TensorData<St, L>>, backend: Arc<B>) -> Self {
        Self(TensorChainData {
            sites,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

impl<St, L, B: ComputeBackend> Mpo<St, L, B>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Create an MPO from raw site `TensorData` and an explicit backend.
    ///
    /// Each site should have rank 4 with shape
    /// `(χ_L, d_ket, d_bra, χ_R)`. The canonical form is initially
    /// [`CanonicalForm::Unknown`].
    pub fn with_backend(sites: Vec<TensorData<St, L>>, backend: Arc<B>) -> Self {
        Self(TensorChainData {
            sites,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

// ----------------------------------------------------------------------------
// Constructors (default NativeBackend)
// ----------------------------------------------------------------------------

impl<St, L> Mps<St, L, NativeBackend>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Create an MPS from raw site `TensorData` using the default
    /// [`NativeBackend`].
    pub fn from_sites(sites: Vec<TensorData<St, L>>) -> Self {
        Self::with_backend(sites, NativeBackend::shared())
    }
}

impl<St, L> Mpo<St, L, NativeBackend>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Create an MPO from raw site `TensorData` using the default
    /// [`NativeBackend`].
    pub fn from_sites(sites: Vec<TensorData<St, L>>) -> Self {
        Self::with_backend(sites, NativeBackend::shared())
    }
}
