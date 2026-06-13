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
//! explicitly at the call site (see `arnet-linalg`). Convenience
//! constructors that need a memory order read it from the host substrate
//! ([`Host`](crate::Host)) without binding the tensor to any backend.

use std::collections::HashMap;
use std::fmt;

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use num_traits::Zero;
use rand::RngExt;

use crate::block_sparse::BlockMeta;
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
/// use arnet_tensor::DenseTensor;
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
    /// Intended for cross-crate kernel-access paths inside `arnet-linalg`
    /// and `arnet-mps`; user code should reach for the inherent methods
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
// Dense raw constructor
//
// Tensor-surface entry point for callers that have a flat buffer (e.g.
// internal kernel-output wrapping). Saves callers from reaching into the
// `DenseTensorData::from_raw_parts` joined surface.
// ============================================================================

impl<T> Tensor<DenseStorage<T>, DenseLayout>
where
    T: Clone,
{
    /// Construct a Dense tensor from flat data and shape. The flat `data`
    /// is taken to be already laid out in the host substrate's preferred
    /// order, and the layout is tagged accordingly — this constructor
    /// cannot tag any other order. To wrap a flat buffer already laid out
    /// in some other order, build `DenseTensorData` with that explicit
    /// order and call [`Tensor::from_data`]; `reordered` is for converting
    /// an already-valid tensor to a different layout, not for reinterpreting
    /// a raw buffer (it would permute values under the wrong source order).
    pub fn from_raw_parts(data: Vec<T>, shape: Vec<usize>) -> Self {
        let td = DenseTensorData::from_raw_parts(data, shape, host_order());
        Self::from_data(td)
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
        Self::dense_filled(shape, S::zero())
    }

    /// Create a Dense tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        Self::dense_filled(shape, S::one())
    }

    /// Create a Dense tensor filled with `value`.
    pub fn filled(shape: Vec<usize>, value: S) -> Self {
        Self::dense_filled(shape, value)
    }

    /// Create an n×n identity matrix.
    pub fn eye(n: usize) -> Self {
        let order = host_order();
        let mut data = vec![S::zero(); n * n];
        // The identity matrix is symmetric, so the flat data is the
        // same regardless of memory order; only the layout's `order()`
        // field differs.
        for i in 0..n {
            data[i * n + i] = S::one();
        }
        let td = DenseTensorData::from_raw_parts(data, vec![n, n], order);
        Self::from_data(td)
    }

    /// Create a Dense tensor filled with values drawn from the
    /// standard distribution via the supplied RNG.
    pub fn random<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<S>,
    {
        let order = host_order();
        let total: usize = shape.iter().product();
        let data: Vec<S> = (0..total).map(|_| rng.random()).collect();
        let td = DenseTensorData::from_raw_parts(data, shape, order);
        Self::from_data(td)
    }

    fn dense_filled(shape: Vec<usize>, value: S) -> Self {
        let order = host_order();
        let len: usize = shape.iter().product();
        let data = DenseTensorData::from_raw_parts(vec![value; len], shape, order);
        Self::from_data(data)
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

// ============================================================================
// BlockSparse raw constructor
//
// Tensor-surface entry point for callers that need pre-validated raw
// parts with an explicit memory order. Unlike `DenseTensor::from_raw_parts`
// it takes the operating backend explicitly: dense paths self-normalize
// (`reordered` round-trips through any target order), so dense construction
// can fix the host order and let downstream ops reorder, whereas
// block-sparse kernels have no reorder step and must read the buffer under
// the operating backend's preferred order — so that order is validated at
// construction against the call-site backend, not the host. Primary
// consumers are internal kernel-output wrapping (block-sparse decomposition
// / matvec pipelines) and tests that pin the Tier 1 rejection path with a
// fabricated layout.
// ============================================================================

impl<T, S> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    T: Clone,
    S: Sector,
{
    /// Construct a BlockSparse tensor from pre-validated raw parts,
    /// an explicit memory order, and a backend supplied only to check the
    /// order (the backend is not stored).
    ///
    /// `block_index` is derived from `blocks` internally to avoid the
    /// duplication-mismatch risk of passing both. Caller is responsible
    /// for the remaining invariants enforced by [`BlockSparseLayout::new`]:
    /// sector conservation per block, coord uniqueness, packed offsets
    /// without gap or overlap, blocks sorted by coordinate.
    /// `data.len() == sum(blocks.size)` is checked by `TensorData::new`.
    ///
    /// # Panics
    ///
    /// Panics if `order != backend.preferred_order()`. Block-sparse
    /// kernels (`contract_block_sparse`, `svd_block_sparse`, etc.) read
    /// the per-sector packed buffer under the backend's preferred order
    /// and have no internal reorder step; a mismatch here would yield
    /// silently wrong output. This is the same Tier 1 invariant the
    /// `Mps` / `Mpo` constructors enforce on their sites.
    pub fn from_raw_parts<B: ComputeBackend>(
        data: Vec<T>,
        blocks: Vec<BlockMeta>,
        indices: Vec<QNIndex<S>>,
        flux: S,
        shape: Vec<usize>,
        order: MemoryOrder,
        backend: &B,
    ) -> Self {
        assert_eq!(
            order,
            backend.preferred_order(),
            "BlockSparseTensor::from_raw_parts: layout order {:?} != backend preferred_order {:?} (Tier 1 invariant; block-sparse kernels do not reorder)",
            order,
            backend.preferred_order(),
        );
        let block_index: HashMap<BlockCoord, usize> = blocks
            .iter()
            .enumerate()
            .map(|(i, m)| (m.coord.clone(), i))
            .collect();
        let td = BlockSparseTensorData::from_raw_parts(
            data,
            blocks,
            block_index,
            indices,
            flux,
            shape,
            order,
        );
        Self::from_data(td)
    }
}
