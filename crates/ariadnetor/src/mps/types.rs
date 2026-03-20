//! Core MPS/MPO data types

use std::sync::Arc;

use arnet_core::backend::ComputeBackend;
use arnet_native::NativeBackend;
use arnet_tensor::TensorStorage;

/// Canonical form of a tensor chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalForm {
    /// No canonicalization guarantees.
    Unknown,
    /// 0..llim left-canonical, rlim..N right-canonical.
    /// Non-canonical region spans multiple sites (rlim - llim > 1).
    PartiallyCanonicalized { llim: usize, rlim: usize },
    /// Single-site orthogonality center at `center`.
    /// 0..center left-canonical, center+1..N right-canonical.
    Canonicalized { center: usize },
}

/// Internal data container for MPS/MPO tensor chains.
///
/// Holds raw tensor storages, a shared backend, and canonical form state.
/// This type is `pub(crate)` — users interact through `Mps` / `Mpo` newtypes.
#[derive(Debug, Clone)]
pub(crate) struct TensorChainData<T, B: ComputeBackend = NativeBackend> {
    pub(crate) storages: Vec<TensorStorage<T>>,
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
pub struct Mps<T = f64, B: ComputeBackend = NativeBackend>(pub(crate) TensorChainData<T, B>);

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
pub struct Mpo<T = f64, B: ComputeBackend = NativeBackend>(pub(crate) TensorChainData<T, B>);

// ============================================================================
// Constructors (any backend)
// ============================================================================

impl<T, B: ComputeBackend> Mps<T, B> {
    /// Create an MPS from raw site storages and an explicit backend.
    ///
    /// Each storage should have rank 3 with shape `(χ_L, d, χ_R)`.
    /// The canonical form is initially `Unknown`.
    pub fn with_backend(storages: Vec<TensorStorage<T>>, backend: Arc<B>) -> Self {
        Self(TensorChainData {
            storages,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

impl<T, B: ComputeBackend> Mpo<T, B> {
    /// Create an MPO from raw site storages and an explicit backend.
    ///
    /// Each storage should have rank 4 with shape `(χ_L, d_ket, d_bra, χ_R)`.
    /// The canonical form is initially `Unknown`.
    pub fn with_backend(storages: Vec<TensorStorage<T>>, backend: Arc<B>) -> Self {
        Self(TensorChainData {
            storages,
            backend,
            canonical_form: CanonicalForm::Unknown,
        })
    }
}

// ============================================================================
// Constructors (default NativeBackend)
// ============================================================================

impl<T> Mps<T, NativeBackend> {
    /// Create an MPS from raw site storages using the default NativeBackend.
    pub fn from_storages(storages: Vec<TensorStorage<T>>) -> Self {
        Self::with_backend(storages, NativeBackend::shared())
    }
}

impl<T> Mpo<T, NativeBackend> {
    /// Create an MPO from raw site storages using the default NativeBackend.
    pub fn from_storages(storages: Vec<TensorStorage<T>>) -> Self {
        Self::with_backend(storages, NativeBackend::shared())
    }
}
