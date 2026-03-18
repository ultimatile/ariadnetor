//! Tensor type combining storage and compute backend
//!
//! `Tensor<T, B>` is the main user-facing type. It holds a [`TensorStorage<T>`]
//! for data and an `Arc<B>` for computation dispatch.
//!
//! The backend type parameter defaults to [`NativeBackend`], so CPU users
//! can simply write `Tensor<f64>` without specifying a backend.

use std::sync::Arc;

use arnet_core::backend::ComputeBackend;
use arnet_core::scalar::Scalar;
use arnet_native::NativeBackend;
pub use arnet_tensor::{DenseTensor, TensorStorage};
use num_traits::{One, Zero};
use std::ops::{Add, Mul};

/// Tensor combining data storage with a compute backend.
///
/// # Type Parameters
///
/// * `T` - Element type (default: `f64`)
/// * `B` - Compute backend (default: [`NativeBackend`])
///
/// # Examples
///
/// ```
/// use arnet::Tensor;
///
/// // CPU tensor (NativeBackend is implicit)
/// let a = Tensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// assert_eq!(a.shape(), &[2, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct Tensor<T = f64, B: ComputeBackend = NativeBackend> {
    /// Underlying data storage
    pub storage: TensorStorage<T>,
    backend: Arc<B>,
}

impl<T, B: ComputeBackend> Tensor<T, B> {
    /// Create a Tensor from storage and an explicit backend.
    pub fn with_backend(storage: TensorStorage<T>, backend: Arc<B>) -> Self {
        Self { storage, backend }
    }

    /// Get a reference to the compute backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Get a shared reference to the backend Arc.
    pub fn backend_arc(&self) -> &Arc<B> {
        &self.backend
    }

    /// Get the shape of the underlying tensor.
    pub fn shape(&self) -> &[usize] {
        self.storage.shape()
    }

    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.storage.rank()
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
}

// ============================================================================
// Constructors with default NativeBackend
// ============================================================================

impl<T> Tensor<T, NativeBackend>
where
    T: Clone,
{
    /// Create a tensor filled with zeros (default: NativeBackend).
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero,
    {
        Self::with_backend(TensorStorage::zeros(shape), NativeBackend::shared())
    }

    /// Create a tensor filled with ones (default: NativeBackend).
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Zero,
    {
        Self::with_backend(TensorStorage::ones(shape), NativeBackend::shared())
    }

    /// Create a tensor filled with a constant value (default: NativeBackend).
    pub fn constant(shape: Vec<usize>, value: T) -> Self {
        Self::with_backend(TensorStorage::constant(shape, value), NativeBackend::shared())
    }

    /// Create a tensor from existing data (default: NativeBackend).
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::with_backend(TensorStorage::from_data(data, shape), NativeBackend::shared())
    }
}

// ============================================================================
// Data access (all backends)
// ============================================================================

impl<T, B: ComputeBackend> Tensor<T, B>
where
    T: Clone,
{
    /// Get a reference to the underlying data (only for Dense).
    pub fn data(&self) -> Option<&[T]> {
        self.storage.data()
    }

    /// Get a mutable reference to the underlying data (only for Dense).
    pub fn data_mut(&mut self) -> Option<&mut [T]> {
        self.storage.data_mut()
    }

    /// Get element at given indices.
    pub fn get(&self, indices: &[usize]) -> T {
        self.storage.get(indices)
    }

    /// Set element at given indices.
    pub fn set(&mut self, indices: &[usize], value: T) {
        self.storage.set(indices, value)
    }

    /// Fill tensor with a constant value.
    pub fn fill(&mut self, value: T) {
        self.storage.fill(value)
    }
}

// ============================================================================
// Arithmetic operations (all backends)
// ============================================================================

impl<T, B: ComputeBackend> Tensor<T, B>
where
    T: Clone + Mul<Output = T>,
{
    /// Scale tensor by a scalar factor (in-place).
    pub fn scale(&mut self, factor: T) {
        self.storage.scale(factor);
    }

    /// Scale tensor and return new tensor (out-of-place).
    pub fn scaled(&self, factor: T) -> Self {
        Self {
            storage: self.storage.scaled(factor),
            backend: Arc::clone(&self.backend),
        }
    }
}

impl<T, B: ComputeBackend> Tensor<T, B>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    /// Linear combination of tensors.
    ///
    /// All tensors must have the same shape.
    pub fn linear_combine(tensors: &[&Tensor<T, B>], coefs: &[T]) -> Result<Tensor<T, B>, String> {
        if tensors.is_empty() {
            return Err("Cannot combine empty tensor list".to_string());
        }

        let backend = Arc::clone(&tensors[0].backend);
        let raw_tensors: Vec<_> = tensors.iter().map(|t| &t.storage).collect();
        let result_storage = TensorStorage::linear_combine(&raw_tensors, coefs)?;

        Ok(Tensor {
            storage: result_storage,
            backend,
        })
    }

    /// Add all tensors (coefficients all = 1).
    pub fn add_all(tensors: &[&Tensor<T, B>]) -> Result<Tensor<T, B>, String> {
        let coefs = vec![T::one(); tensors.len()];
        Self::linear_combine(tensors, &coefs)
    }
}

// ============================================================================
// Norm and normalization operations (all backends)
// ============================================================================

impl<T, B: ComputeBackend> Tensor<T, B>
where
    T: Scalar,
{
    /// Compute Frobenius norm.
    pub fn norm(&self) -> T::Real {
        self.storage.norm()
    }

    /// Normalize to unit norm (in-place).
    ///
    /// Returns the norm before normalization.
    pub fn normalize(&mut self) -> T::Real {
        self.storage.normalize()
    }

    /// Normalize and return new tensor (out-of-place).
    ///
    /// Returns `(normalized_tensor, original_norm)`.
    pub fn normalized(&self) -> (Self, T::Real) {
        let (normalized_storage, norm) = self.storage.normalized();
        (
            Self {
                storage: normalized_storage,
                backend: Arc::clone(&self.backend),
            },
            norm,
        )
    }
}

