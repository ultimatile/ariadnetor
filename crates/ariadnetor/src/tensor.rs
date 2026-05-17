//! Tensor type combining storage, layout, and compute backend.
//!
//! `Tensor<St, L, B>` joins a [`TensorData<St, L>`](arnet_tensor::TensorData)
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
use arnet_tensor::{
    BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, DenseTensorData, Storage,
    StorageFor, TensorData, TensorLayout,
};
use num_traits::{One, Zero};
use std::ops::{Add, Mul};

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
/// use arnet::DenseTensor;
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

    fn dense_filled(shape: Vec<usize>, value: S) -> Self {
        let backend = NativeBackend::shared();
        let order = backend.preferred_order();
        let len: usize = shape.iter().product();
        let data = DenseTensorData::from_raw_parts(vec![value; len], shape, order);
        Self::with_backend(data, backend)
    }
}

// ============================================================================
// Dense-specific data access (all backends)
// ============================================================================

impl<S, B: ComputeBackend> Tensor<DenseStorage<S>, DenseLayout, B> {
    /// Get a reference to the underlying contiguous data buffer.
    pub fn data_slice(&self) -> &[S] {
        self.data.storage().data()
    }

    /// Get a mutable reference to the underlying data buffer
    /// (CoW-aware).
    pub fn data_slice_mut(&mut self) -> &mut [S]
    where
        S: Clone,
    {
        self.data.storage_mut().data_mut()
    }
}

impl<S: Scalar, B: ComputeBackend> Tensor<DenseStorage<S>, DenseLayout, B> {
    /// Memory order this tensor's flat data is laid out in.
    pub fn order(&self) -> arnet_core::backend::MemoryOrder {
        self.data.layout().order()
    }

    /// Get element at the given indices.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len() != rank` or any index exceeds the
    /// corresponding axis dimension.
    pub fn get(&self, indices: &[usize]) -> S {
        let shape = self.shape();
        assert_eq!(
            indices.len(),
            shape.len(),
            "Tensor::get: indices length {} doesn't match rank {}",
            indices.len(),
            shape.len(),
        );
        for (axis, (&idx, &dim)) in indices.iter().zip(shape).enumerate() {
            assert!(
                idx < dim,
                "Tensor::get: index {idx} out of bounds for axis {axis} with size {dim}",
            );
        }
        let order = self.order();
        let flat = arnet_tensor::flat_index(indices, shape, order);
        self.data.storage().data()[flat]
    }

    /// Set element at the given indices.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len() != rank` or any index exceeds the
    /// corresponding axis dimension.
    pub fn set(&mut self, indices: &[usize], value: S) {
        let shape_owned: Vec<usize> = self.shape().to_vec();
        assert_eq!(
            indices.len(),
            shape_owned.len(),
            "Tensor::set: indices length {} doesn't match rank {}",
            indices.len(),
            shape_owned.len(),
        );
        for (axis, (&idx, &dim)) in indices.iter().zip(&shape_owned).enumerate() {
            assert!(
                idx < dim,
                "Tensor::set: index {idx} out of bounds for axis {axis} with size {dim}",
            );
        }
        let order = self.order();
        let flat = arnet_tensor::flat_index(indices, &shape_owned, order);
        self.data.storage_mut().data_mut()[flat] = value;
    }

    /// Fill the tensor with a constant value.
    pub fn fill(&mut self, value: S) {
        for slot in self.data.storage_mut().data_mut().iter_mut() {
            *slot = value;
        }
    }
}

// ============================================================================
// Dense-specific arithmetic operations (all backends)
// ============================================================================

impl<S: Clone, B: ComputeBackend> Tensor<DenseStorage<S>, DenseLayout, B> {
    /// Scale every element by a factor (in-place).
    pub fn scale<F>(&mut self, factor: F)
    where
        S: Mul<F, Output = S>,
        F: Clone,
    {
        for slot in self.data.storage_mut().data_mut().iter_mut() {
            *slot = slot.clone() * factor.clone();
        }
    }

    /// Scale every element by a factor (out-of-place).
    pub fn scaled<F>(&self, factor: F) -> Self
    where
        S: Mul<F, Output = S>,
        F: Clone,
    {
        let new_data: Vec<S> = self
            .data
            .storage()
            .data()
            .iter()
            .map(|x| x.clone() * factor.clone())
            .collect();
        let shape = self.shape().to_vec();
        let order = self.data.layout().order();
        let td = DenseTensorData::from_raw_parts(new_data, shape, order);
        Self::with_backend(td, Arc::clone(&self.backend))
    }
}

impl<S, B: ComputeBackend> Tensor<DenseStorage<S>, DenseLayout, B>
where
    S: Clone + Zero + One + Add<Output = S> + Mul<Output = S>,
{
    /// Linear combination of tensors: `Σ_i coefs[i] * tensors[i]`.
    ///
    /// All inputs must have the same shape and memory order. The
    /// result inherits the order of the first input.
    pub fn linear_combine(
        tensors: &[&Tensor<DenseStorage<S>, DenseLayout, B>],
        coefs: &[S],
    ) -> Result<Tensor<DenseStorage<S>, DenseLayout, B>, String> {
        if tensors.is_empty() {
            return Err("Cannot combine empty tensor list".to_string());
        }
        if tensors.len() != coefs.len() {
            return Err(format!(
                "linear_combine: tensors.len() = {} != coefs.len() = {}",
                tensors.len(),
                coefs.len(),
            ));
        }
        let shape0 = tensors[0].shape().to_vec();
        let order0 = tensors[0].data.layout().order();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.shape() != shape0.as_slice() {
                return Err(format!(
                    "linear_combine: shape mismatch at index {i}: expected {shape0:?}, got {:?}",
                    t.shape(),
                ));
            }
            if t.data.layout().order() != order0 {
                return Err(format!(
                    "linear_combine: memory-order mismatch at index {i}",
                ));
            }
        }
        let len: usize = shape0.iter().product();
        let mut acc = vec![S::zero(); len];
        for (t, c) in tensors.iter().zip(coefs) {
            for (a, s) in acc.iter_mut().zip(t.data.storage().data()) {
                *a = a.clone() + s.clone() * c.clone();
            }
        }
        let td = DenseTensorData::from_raw_parts(acc, shape0, order0);
        Ok(Tensor::with_backend(
            td,
            Arc::clone(tensors[0].backend_arc()),
        ))
    }

    /// Add all tensors (coefficients all = 1).
    pub fn add_all(
        tensors: &[&Tensor<DenseStorage<S>, DenseLayout, B>],
    ) -> Result<Tensor<DenseStorage<S>, DenseLayout, B>, String> {
        let coefs = vec![S::one(); tensors.len()];
        Self::linear_combine(tensors, &coefs)
    }
}

// ============================================================================
// Dense-specific norm / normalization (all backends)
// ============================================================================

impl<S, B: ComputeBackend> Tensor<DenseStorage<S>, DenseLayout, B>
where
    S: Scalar,
{
    /// Frobenius norm.
    pub fn norm(&self) -> S::Real {
        let mut sq = S::Real::zero();
        for &x in self.data.storage().data() {
            let a = x.abs();
            sq = sq + a * a;
        }
        <S::Real as num_traits::Float>::sqrt(sq)
    }

    /// Normalize to unit norm (in-place). Returns the original norm.
    ///
    /// # Panics
    ///
    /// Panics if the tensor has zero norm.
    pub fn normalize(&mut self) -> S::Real {
        let norm = self.norm();
        assert!(norm != S::Real::zero(), "Cannot normalize zero tensor");
        let inv_norm = S::Real::one() / norm;
        for slot in self.data.storage_mut().data_mut().iter_mut() {
            *slot = slot.scale_real(inv_norm);
        }
        norm
    }

    /// Normalize and return a new tensor (out-of-place).
    pub fn normalized(&self) -> (Self, S::Real) {
        let mut clone = self.clone();
        let n = clone.normalize();
        (clone, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tensor_mutation<S>(zero: S, val: S, fill_val: S, scale_factor: S)
    where
        S: Scalar + PartialEq + fmt::Debug + Mul<S, Output = S>,
    {
        let mut t = DenseTensor::<S>::zeros(vec![2, 3]);

        // set / get round-trip
        t.set(&[1, 2], val);
        assert_eq!(t.get(&[1, 2]), val);
        assert_eq!(t.get(&[0, 0]), zero);

        // fill overwrites all elements
        t.fill(fill_val);
        assert_eq!(t.get(&[0, 0]), fill_val);
        assert_eq!(t.get(&[1, 2]), fill_val);

        // data_slice_mut provides mutable access
        t.data_slice_mut()[0] = val;
        assert_eq!(t.get(&[0, 0]), val);

        // scale multiplies all elements
        t.fill(val);
        t.scale(scale_factor);
        assert_eq!(t.get(&[0, 0]), val * scale_factor);
    }

    #[test]
    fn test_tensor_mutation() {
        assert_tensor_mutation(0.0f64, 42.0, 2.72, 3.0);
        assert_tensor_mutation(0.0f32, 42.0, 2.72, 3.0);
    }

    #[test]
    fn scaled_out_of_place_preserves_original() {
        let mut a = DenseTensor::<f64>::zeros(vec![2, 2]);
        a.fill(3.0);
        let b = a.scaled(2.0);
        // a unchanged
        assert_eq!(a.get(&[0, 0]), 3.0);
        // b scaled
        assert_eq!(b.get(&[0, 0]), 6.0);
        assert_eq!(b.get(&[1, 1]), 6.0);
        assert_eq!(b.shape(), a.shape());
    }

    #[test]
    fn norm_matches_frobenius_definition() {
        let mut t = DenseTensor::<f64>::zeros(vec![2, 2]);
        t.set(&[0, 0], 3.0);
        t.set(&[1, 1], 4.0);
        // sqrt(9 + 16) = 5
        let n = t.norm();
        assert!((n - 5.0).abs() < 1e-12, "expected 5.0, got {n}");
    }

    #[test]
    fn normalize_in_place_returns_original_norm_and_unitizes() {
        let mut t = DenseTensor::<f64>::zeros(vec![2]);
        t.set(&[0], 3.0);
        t.set(&[1], 4.0);
        let n = t.normalize();
        assert!((n - 5.0).abs() < 1e-12, "returned norm {n}, expected 5");
        // post-normalize Frobenius norm is 1
        assert!((t.norm() - 1.0).abs() < 1e-12);
        // elements scaled by 1/5
        assert!((t.get(&[0]) - 0.6).abs() < 1e-12);
        assert!((t.get(&[1]) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn normalized_out_of_place_keeps_original_intact() {
        let mut a = DenseTensor::<f64>::zeros(vec![2]);
        a.set(&[0], 3.0);
        a.set(&[1], 4.0);
        let (b, n) = a.normalized();
        assert!((n - 5.0).abs() < 1e-12);
        // original elements preserved
        assert_eq!(a.get(&[0]), 3.0);
        assert_eq!(a.get(&[1]), 4.0);
        // normalized copy has unit norm
        assert!((b.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "Cannot normalize zero tensor")]
    fn normalize_panics_on_zero_tensor() {
        let mut t = DenseTensor::<f64>::zeros(vec![3, 3]);
        t.normalize();
    }

    #[test]
    fn linear_combine_sums_with_coefs() {
        let mut a = DenseTensor::<f64>::zeros(vec![2]);
        a.set(&[0], 1.0);
        a.set(&[1], 2.0);
        let mut b = DenseTensor::<f64>::zeros(vec![2]);
        b.set(&[0], 10.0);
        b.set(&[1], 20.0);
        let r = DenseTensor::linear_combine(&[&a, &b], &[3.0, 4.0]).unwrap();
        // 3*1 + 4*10 = 43; 3*2 + 4*20 = 86
        assert_eq!(r.get(&[0]), 43.0);
        assert_eq!(r.get(&[1]), 86.0);
        assert_eq!(r.shape(), a.shape());
    }

    #[test]
    fn add_all_sums_with_unit_coefs() {
        let mut a = DenseTensor::<f64>::zeros(vec![2]);
        a.set(&[0], 1.0);
        a.set(&[1], 2.0);
        let mut b = DenseTensor::<f64>::zeros(vec![2]);
        b.set(&[0], 10.0);
        b.set(&[1], 20.0);
        let r = DenseTensor::add_all(&[&a, &b]).unwrap();
        assert_eq!(r.get(&[0]), 11.0);
        assert_eq!(r.get(&[1]), 22.0);
    }

    #[test]
    fn linear_combine_rejects_empty_list() {
        let err = DenseTensor::<f64>::linear_combine(&[], &[]).unwrap_err();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn linear_combine_rejects_length_mismatch() {
        let a = DenseTensor::<f64>::zeros(vec![2]);
        let b = DenseTensor::<f64>::zeros(vec![2]);
        let err = DenseTensor::linear_combine(&[&a, &b], &[1.0]).unwrap_err();
        assert!(
            err.contains("tensors.len()") && err.contains("coefs.len()"),
            "got: {err}",
        );
    }

    #[test]
    fn linear_combine_rejects_shape_mismatch() {
        let a = DenseTensor::<f64>::zeros(vec![2]);
        let b = DenseTensor::<f64>::zeros(vec![3]);
        let err = DenseTensor::linear_combine(&[&a, &b], &[1.0, 1.0]).unwrap_err();
        assert!(err.contains("shape mismatch"), "got: {err}");
    }

    #[test]
    fn block_sparse_tensor_alias_resolves_and_basics_work() {
        use arnet_tensor::{BlockSparseTensorData, Direction, QNIndex, U1Sector};

        let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
        let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
        let backend = NativeBackend::shared();
        let order = backend.preferred_order();
        let td: BlockSparseTensorData<f64, U1Sector> =
            BlockSparseTensorData::zeros(vec![row, col], U1Sector(0), order);
        let t: BlockSparseTensor<f64, U1Sector> = Tensor::with_backend(td, backend);

        assert_eq!(t.shape(), &[5, 5]);
        assert_eq!(t.rank(), 2);
    }
}
