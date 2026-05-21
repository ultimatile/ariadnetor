//! Tensor type combining storage, layout, and compute backend.
//!
//! `Tensor<St, L, B>` joins a [`TensorData<St, L>`](crate::TensorData)
//! bundle with an `Arc<B>` compute backend. Concrete user-facing
//! aliases:
//!
//! - [`DenseTensor<T, B>`] = `Tensor<DenseStorage<T>, DenseLayout, B>`
//! - [`BlockSparseTensor<T, S, B>`] =
//!   `Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, B>`
//!
//! The backend type parameter defaults to
//! [`NativeBackend`](arnet_native::NativeBackend), so CPU users can
//! write `DenseTensor<f64>` without naming a backend.

use std::fmt;
use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_native::NativeBackend;
use num_traits::Zero;

use crate::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensorData, DenseLayout,
    DenseStorage, DenseTensorData, QNIndex, Sector, Storage, StorageFor, TensorData, TensorLayout,
};

mod dense_ops;

mod block_sparse_ops;

#[cfg(test)]
mod tests;

/// Tensor combining a [`TensorData`] bundle with a compute backend.
///
/// # Type Parameters
///
/// * `St` - Storage half ([`DenseStorage<T>`] or [`BlockSparseStorage<T>`])
/// * `L`  - Layout half ([`DenseLayout`] or [`BlockSparseLayout<S>`])
/// * `B`  - Compute backend (default: [`NativeBackend`])
///
/// # Examples
///
/// ```
/// use arnet_tensor::DenseTensor;
///
/// // CPU tensor: DenseStorage<f64> + DenseLayout + NativeBackend
/// let a = DenseTensor::<f64>::zeros(vec![2, 2]);
/// assert_eq!(a.shape(), &[2, 2]);
/// ```
pub struct Tensor<St, L, B = NativeBackend>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    data: TensorData<St, L>,
    backend: Arc<B>,
}

/// Backend-aware Dense tensor alias.
pub type DenseTensor<T = f64, B = NativeBackend> = Tensor<DenseStorage<T>, DenseLayout, B>;

/// Backend-aware BlockSparse tensor alias.
pub type BlockSparseTensor<T, S, B = NativeBackend> =
    Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, B>;

// ============================================================================
// Manual Clone / Debug
//
// `Tensor` is generic over `St` and `L`; deriving requires bounds on
// both that are not always present. The manual impls add the bounds
// only where needed.
// ============================================================================

impl<St, L, B> Clone for Tensor<St, L, B>
where
    St: Storage + StorageFor<L> + Clone,
    L: TensorLayout + Clone,
    B: ComputeBackend,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            backend: Arc::clone(&self.backend),
        }
    }
}

impl<St, L, B> fmt::Debug for Tensor<St, L, B>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.data.layout().shape())
            .finish()
    }
}

// ============================================================================
// Generic methods (all storage / layout combinations)
// ============================================================================

impl<St, L, B> Tensor<St, L, B>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    /// Build a tensor from a pre-bundled [`TensorData`] and a backend.
    pub fn with_backend(data: TensorData<St, L>, backend: Arc<B>) -> Self {
        Self { data, backend }
    }

    /// Reference to the joined [`TensorData`] bundle.
    pub fn data(&self) -> &TensorData<St, L> {
        &self.data
    }

    /// Mutable reference to the joined [`TensorData`] bundle.
    pub fn data_mut(&mut self) -> &mut TensorData<St, L> {
        &mut self.data
    }

    /// Reference to the compute backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Shared reference to the backend `Arc`.
    pub fn backend_arc(&self) -> &Arc<B> {
        &self.backend
    }

    /// Logical shape (delegates to the layout).
    pub fn shape(&self) -> &[usize] {
        self.data.layout().shape()
    }

    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Total number of logical elements (`product(shape)`).
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Whether the tensor has zero logical elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// Dense generic-backend constructor
//
// Tensor-surface entry point for callers that need an explicit memory
// order and an explicit backend (e.g. mps tests pinning the Tier 1
// rejection path, internal kernel-output wrapping). Saves callers from
// reaching into the `DenseTensorData::from_raw_parts` joined surface.
// ============================================================================

impl<T, B> Tensor<DenseStorage<T>, DenseLayout, B>
where
    T: Clone,
    B: ComputeBackend,
{
    /// Construct a Dense tensor from flat data, shape, memory order, and
    /// an explicit backend `Arc`.
    ///
    /// Equivalent to building a `DenseTensorData` via the joined
    /// `from_raw_parts` accessor and pairing it with `backend`; the
    /// stored layout's `order()` reflects the `order` argument, not the
    /// backend's preferred order — downstream Tier 1 / Tier 2 asserts
    /// are the authority on whether the resulting tensor is valid for
    /// the target chain.
    pub fn from_raw_parts(
        data: Vec<T>,
        shape: Vec<usize>,
        order: arnet_core::backend::MemoryOrder,
        backend: Arc<B>,
    ) -> Self {
        let td = DenseTensorData::from_raw_parts(data, shape, order);
        Self::with_backend(td, backend)
    }
}

// ============================================================================
// Dense-specific constructors with the NativeBackend pin
//
// `ComputeBackend` exposes no constructor, so the impl cannot
// generalize over `B`. The pin is intentional and preserved from the
// pre-split shape.
// ============================================================================

impl<S: Scalar> Tensor<DenseStorage<S>, DenseLayout, NativeBackend> {
    /// Create a Dense tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::dense_filled(shape, S::zero())
    }

    /// Create a Dense tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        Self::dense_filled(shape, S::one())
    }

    /// Create a Dense tensor filled with a constant value.
    pub fn constant(shape: Vec<usize>, value: S) -> Self {
        Self::dense_filled(shape, value)
    }

    /// Create an n×n identity matrix.
    pub fn eye(n: usize) -> Self {
        let backend = NativeBackend::shared();
        let order = backend.preferred_order();
        let mut data = vec![S::zero(); n * n];
        // The identity matrix is symmetric, so the flat data is the
        // same regardless of memory order; only the layout's `order()`
        // field differs.
        for i in 0..n {
            data[i * n + i] = S::one();
        }
        let td = DenseTensorData::from_raw_parts(data, vec![n, n], order);
        Self::with_backend(td, backend)
    }

    /// Create a Dense tensor filled with values drawn from the
    /// standard distribution via the supplied RNG.
    pub fn random<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<S>,
    {
        let backend = NativeBackend::shared();
        let order = backend.preferred_order();
        let total: usize = shape.iter().product();
        let data: Vec<S> = (0..total).map(|_| rng.random()).collect();
        let td = DenseTensorData::from_raw_parts(data, shape, order);
        Self::with_backend(td, backend)
    }

    fn dense_filled(shape: Vec<usize>, value: S) -> Self {
        let backend = NativeBackend::shared();
        let order = backend.preferred_order();
        let len: usize = shape.iter().product();
        let data = DenseTensorData::from_raw_parts(vec![value; len], shape, order);
        Self::with_backend(data, backend)
    }
}

// ============================================================================
// BlockSparse-specific constructors with the NativeBackend pin
//
// As with Dense, `ComputeBackend` exposes no constructor, so the pin
// to `NativeBackend` is intentional. The memory order is taken from
// `NativeBackend::preferred_order()`; users needing arbitrary order
// must go through the joined-path `TensorData::new` route.
// ============================================================================

impl<T, S> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, NativeBackend>
where
    T: Clone + Zero,
    S: Sector,
{
    /// Create a zero-filled `BlockSparseTensor` enumerating every
    /// flux-allowed block of the supplied `QNIndex` legs.
    pub fn zeros(indices: Vec<QNIndex<S>>, flux: S) -> Self {
        let backend = NativeBackend::shared();
        let order = backend.preferred_order();
        let td = BlockSparseTensorData::zeros(indices, flux, order);
        Self::with_backend(td, backend)
    }
}

impl<T, S, B> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, B>
where
    T: Clone + Zero,
    S: Sector,
    B: ComputeBackend,
{
    /// Create a zero-filled `BlockSparseTensor` anchored on an explicit
    /// backend. The layout's memory order is taken from the backend's
    /// preferred order so that the per-tensor Tier 1 invariant
    /// (`layout.order() == backend.preferred_order()`) holds at
    /// construction.
    pub fn zeros_with_backend(indices: Vec<QNIndex<S>>, flux: S, backend: Arc<B>) -> Self {
        let order = backend.preferred_order();
        let td = BlockSparseTensorData::zeros(indices, flux, order);
        Self::with_backend(td, backend)
    }
}

impl<T, S> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, NativeBackend>
where
    T: Clone,
    S: Sector,
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
{
    /// Create a `BlockSparseTensor` whose flux-allowed blocks are
    /// filled with values drawn from the standard distribution via the
    /// supplied RNG.
    pub fn random<R: rand::Rng>(indices: Vec<QNIndex<S>>, flux: S, rng: &mut R) -> Self {
        let backend = NativeBackend::shared();
        let order = backend.preferred_order();
        let td = BlockSparseTensorData::random(indices, flux, order, rng);
        Self::with_backend(td, backend)
    }
}

impl<T, S> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>, NativeBackend>
where
    T: Clone + Zero,
    S: Sector,
{
    /// Construct a `BlockSparseTensor` by populating each flux-allowed
    /// block from a closure receiving the block coordinate and its
    /// dense block shape.
    pub fn from_block_fn<F>(indices: Vec<QNIndex<S>>, flux: S, f: F) -> Self
    where
        F: FnMut(&BlockCoord, &[usize]) -> Vec<T>,
    {
        let backend = NativeBackend::shared();
        let order = backend.preferred_order();
        let td = BlockSparseTensorData::from_block_fn(indices, flux, order, f);
        Self::with_backend(td, backend)
    }
}
