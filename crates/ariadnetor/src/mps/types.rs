//! Core MPS/MPO data types

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::TruncSvdParams;
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, TensorRepr};

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

/// Algorithm used by [`apply`](super::dispatch::apply_with_method) to multiply
/// an MPO into an MPS.
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

/// Internal data container for MPS/MPO tensor chains.
///
/// Holds raw tensor storages, a shared backend, and canonical form state.
/// This type is `pub(crate)` — users interact through `Mps` / `Mpo` newtypes.
#[derive(Debug, Clone)]
pub(crate) struct TensorChainData<R, B: ComputeBackend = NativeBackend> {
    pub(crate) storages: Vec<R>,
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
pub struct Mps<R = Dense<f64>, B: ComputeBackend = NativeBackend>(pub(crate) TensorChainData<R, B>);

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
pub struct Mpo<R = Dense<f64>, B: ComputeBackend = NativeBackend>(pub(crate) TensorChainData<R, B>);

// ============================================================================
// Constructors (storage-agnostic, any backend)
// ============================================================================

impl<R: TensorRepr, B: ComputeBackend> Mps<R, B> {
    /// Create an MPS from raw site storages and an explicit backend.
    ///
    /// Each storage should have rank 3 with shape `(χ_L, d, χ_R)`.
    /// The canonical form is initially `Unknown`.
    pub fn with_backend(storages: Vec<R>, backend: Arc<B>) -> Self {
        Self(TensorChainData {
            storages,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

impl<R: TensorRepr, B: ComputeBackend> Mpo<R, B> {
    /// Create an MPO from raw site storages and an explicit backend.
    ///
    /// Each storage should have rank 4 with shape `(χ_L, d_ket, d_bra, χ_R)`.
    /// The canonical form is initially `Unknown`.
    pub fn with_backend(storages: Vec<R>, backend: Arc<B>) -> Self {
        Self(TensorChainData {
            storages,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

// ============================================================================
// Constructors (storage-agnostic, default NativeBackend)
// ============================================================================

impl<R: TensorRepr> Mps<R, NativeBackend> {
    /// Create an MPS from raw site storages using the default NativeBackend.
    pub fn from_storages(storages: Vec<R>) -> Self {
        Self::with_backend(storages, NativeBackend::shared())
    }
}

impl<R: TensorRepr> Mpo<R, NativeBackend> {
    /// Create an MPO from raw site storages using the default NativeBackend.
    pub fn from_storages(storages: Vec<R>) -> Self {
        Self::with_backend(storages, NativeBackend::shared())
    }
}
