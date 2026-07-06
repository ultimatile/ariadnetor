//! Tensor type combining storage and layout.
//!
//! `Tensor<St, L>` is a thin wrapper over a
//! [`TensorData<St, L>`](crate::TensorData) bundle. Concrete user-facing
//! aliases:
//!
//! - [`DenseTensor<T>`] = `Tensor<DenseStorage<T>, DenseLayout>`
//! - [`BlockSparseTensor<T, S>`] =
//!   `Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>`
//!
//! The tensor carries no compute backend: operations take the backend
//! explicitly at the call site (see `ariadnetor-linalg`). Convenience
//! constructors that need a memory order read it from the host substrate
//! ([`Host`](crate::Host)) without binding the tensor to any backend.

use std::fmt;

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{ComputeBackend, MemoryOrder};
use num_traits::Zero;

use crate::capability::Host;
use crate::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensorData, DenseLayout,
    DenseStorage, DenseTensorData, QNIndex, Sector, Storage, StorageFor, TensorData, TensorLayout,
};

mod dense_ops;

mod block_sparse_ops;

#[cfg(test)]
mod tests;

/// Memory order for host-resident convenience constructors.
///
/// Read through the [`Host`](crate::Host) substrate alias rather than a
/// `NativeBackend` literal so the host order has a single source and the
/// substrate can be repointed in one place.
fn host_order() -> MemoryOrder {
    Host::shared().preferred_order()
}

/// Tensor wrapping a [`TensorData`] bundle.
///
/// # Type Parameters
///
/// * `St` - Storage half ([`DenseStorage<T>`] or [`BlockSparseStorage<T>`])
/// * `L`  - Layout half ([`DenseLayout`] or [`BlockSparseLayout<S>`])
///
/// # Examples
///
/// ```
/// use ariadnetor_tensor::DenseTensor;
///
/// let a = DenseTensor::<f64>::zeros(vec![2, 2]);
/// assert_eq!(a.shape(), &[2, 2]);
/// ```
pub struct Tensor<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    data: TensorData<St, L>,
}

/// Dense tensor alias.
pub type DenseTensor<T = f64> = Tensor<DenseStorage<T>, DenseLayout>;

/// BlockSparse tensor alias.
pub type BlockSparseTensor<T, S> = Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>;

// ============================================================================
// Manual Clone / Debug
//
// `Tensor` is generic over `St` and `L`; deriving requires bounds on
// both that are not always present. The manual impls add the bounds
// only where needed.
// ============================================================================

impl<St, L> Clone for Tensor<St, L>
where
    St: Storage + StorageFor<L> + Clone,
    L: TensorLayout + Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<St, L> fmt::Debug for Tensor<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
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

impl<St, L> Tensor<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Build a tensor from a pre-bundled [`TensorData`].
    pub fn from_data(data: TensorData<St, L>) -> Self {
        Self { data }
    }

    /// Internal escape hatch: reference to the joined [`TensorData`]
    /// bundle.
    ///
    /// Intended for cross-crate kernel-access paths inside `ariadnetor-linalg`
    /// and `ariadnetor-mps`; user code should reach for the inherent methods
    /// on [`DenseTensor`] / [`BlockSparseTensor`] instead.
    pub fn data(&self) -> &TensorData<St, L> {
        &self.data
    }

    /// Internal escape hatch: mutable reference to the joined
    /// [`TensorData`] bundle.
    ///
    /// Same audience as [`Tensor::data`] — cross-crate kernel paths
    /// that need to mutate raw storage / layout state.
    pub fn data_mut(&mut self) -> &mut TensorData<St, L> {
        &mut self.data
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
// Dense-specific host constructors
//
// The memory order is taken from the host substrate's preferred order so
// dispatch paths that require preferred-order alignment find it satisfied
// at construction. These build host-resident data and bind the tensor to
// no backend.
// ============================================================================

impl<S: Scalar> Tensor<DenseStorage<S>, DenseLayout> {
    /// Create a Dense tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::from_data(DenseTensorData::zeros_in_order(shape, host_order()))
    }

    /// Create a Dense tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        Self::from_data(DenseTensorData::ones_in_order(shape, host_order()))
    }

    /// Create a Dense tensor filled with `value`.
    pub fn filled(shape: Vec<usize>, value: S) -> Self {
        Self::from_data(DenseTensorData::filled_in_order(shape, value, host_order()))
    }

    /// Create an n×n identity matrix.
    pub fn eye(n: usize) -> Self {
        Self::from_data(DenseTensorData::eye_in_order(n, host_order()))
    }

    /// Create a Dense tensor filled with values drawn from the
    /// standard distribution via the supplied RNG.
    pub fn random<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<S>,
    {
        Self::from_data(DenseTensorData::random_in_order(shape, host_order(), rng))
    }
}

// ============================================================================
// BlockSparse-specific host constructors
//
// As with Dense, the memory order is taken from the host substrate's
// preferred order; users needing arbitrary order must go through the
// joined-path `TensorData::new` route.
// ============================================================================

impl<T, S> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    T: Clone + Zero,
    S: Sector,
{
    /// Create a zero-filled `BlockSparseTensor` enumerating every
    /// flux-allowed block of the supplied `QNIndex` legs.
    pub fn zeros(indices: Vec<QNIndex<S>>, flux: S) -> Self {
        let order = host_order();
        let td = BlockSparseTensorData::zeros(indices, flux, order);
        Self::from_data(td)
    }
}

impl<T, S> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    T: Clone,
    S: Sector,
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
{
    /// Create a `BlockSparseTensor` whose flux-allowed blocks are
    /// filled with values drawn from the standard distribution via the
    /// supplied RNG.
    pub fn random<R: rand::Rng>(indices: Vec<QNIndex<S>>, flux: S, rng: &mut R) -> Self {
        let order = host_order();
        let td = BlockSparseTensorData::random(indices, flux, order, rng);
        Self::from_data(td)
    }
}

impl<T, S> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>
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
        let order = host_order();
        let td = BlockSparseTensorData::from_block_fn(indices, flux, order, f);
        Self::from_data(td)
    }
}
