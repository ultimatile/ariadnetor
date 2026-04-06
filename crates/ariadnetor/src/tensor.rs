//! Tensor type combining storage and compute backend
//!
//! `Tensor<T, B>` is the main user-facing type. It holds a storage `T`
//! (defaulting to [`Dense<f64>`]) and an `Arc<B>` for computation dispatch.
//!
//! The backend type parameter defaults to [`NativeBackend`], so CPU users
//! can simply write `Tensor` without specifying a backend.

use std::sync::Arc;

use arnet_core::backend::ComputeBackend;
use arnet_core::scalar::Scalar;
use arnet_native::NativeBackend;
pub use arnet_tensor::Dense;
use arnet_tensor::TensorRepr;
use num_traits::{One, Zero};
use std::ops::{Add, Mul};

/// Tensor combining data storage with a compute backend.
///
/// # Type Parameters
///
/// * `T` - Storage representation (default: [`Dense<f64>`])
/// * `B` - Compute backend (default: [`NativeBackend`])
///
/// # Examples
///
/// ```
/// use arnet::{Dense, Tensor};
///
/// // CPU tensor (Dense<f64> + NativeBackend are implicit)
/// let a = Tensor::<Dense<f64>>::zeros(vec![2, 2]);
/// assert_eq!(a.shape(), &[2, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct Tensor<T = Dense<f64>, B: ComputeBackend = NativeBackend> {
    /// Underlying data storage
    pub storage: T,
    backend: Arc<B>,
}

// ============================================================================
// Generic methods (all storage types)
// ============================================================================

impl<T: TensorRepr, B: ComputeBackend> Tensor<T, B> {
    /// Create a Tensor from storage and an explicit backend.
    pub fn with_backend(storage: T, backend: Arc<B>) -> Self {
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
// Dense-specific constructors with default NativeBackend
// ============================================================================

impl<S> Tensor<Dense<S>, NativeBackend>
where
    S: Clone,
{
    /// Create a tensor filled with zeros (default: NativeBackend).
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        S: Zero,
    {
        Self::with_backend(Dense::zeros(shape), NativeBackend::shared())
    }

    /// Create a tensor filled with ones (default: NativeBackend).
    pub fn ones(shape: Vec<usize>) -> Self
    where
        S: One + Zero,
    {
        Self::with_backend(Dense::ones(shape), NativeBackend::shared())
    }

    /// Create a tensor filled with a constant value (default: NativeBackend).
    pub fn constant(shape: Vec<usize>, value: S) -> Self {
        Self::with_backend(Dense::constant(shape, value), NativeBackend::shared())
    }
}

// ============================================================================
// Dense-specific data access (all backends)
// ============================================================================

impl<S, B: ComputeBackend> Tensor<Dense<S>, B>
where
    S: Clone,
{
    /// Get a reference to the underlying data.
    pub fn data(&self) -> &[S] {
        self.storage.data()
    }

    /// Get a mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut [S] {
        self.storage.data_mut()
    }

    /// Get element at given indices.
    pub fn get(&self, indices: &[usize]) -> S {
        self.storage.get(indices)
    }

    /// Set element at given indices.
    pub fn set(&mut self, indices: &[usize], value: S) {
        self.storage.set(indices, value)
    }

    /// Fill tensor with a constant value.
    pub fn fill(&mut self, value: S) {
        self.storage.fill(value)
    }
}

// ============================================================================
// Dense-specific arithmetic operations (all backends)
// ============================================================================

impl<S: Clone, B: ComputeBackend> Tensor<Dense<S>, B> {
    /// Scale tensor by a scalar factor (in-place).
    pub fn scale<F>(&mut self, factor: F)
    where
        S: Mul<F, Output = S>,
        F: Clone,
    {
        self.storage.scale(factor);
    }

    /// Scale tensor and return new tensor (out-of-place).
    pub fn scaled<F>(&self, factor: F) -> Self
    where
        S: Mul<F, Output = S>,
        F: Clone,
    {
        Self {
            storage: self.storage.scaled(factor),
            backend: Arc::clone(&self.backend),
        }
    }
}

impl<S, B: ComputeBackend> Tensor<Dense<S>, B>
where
    S: Clone + Zero + One + Add<Output = S> + Mul<Output = S>,
{
    /// Linear combination of tensors.
    ///
    /// All tensors must have the same shape.
    pub fn linear_combine(
        tensors: &[&Tensor<Dense<S>, B>],
        coefs: &[S],
    ) -> Result<Tensor<Dense<S>, B>, String> {
        if tensors.is_empty() {
            return Err("Cannot combine empty tensor list".to_string());
        }

        let backend = Arc::clone(&tensors[0].backend);
        let raw_tensors: Vec<_> = tensors.iter().map(|t| &t.storage).collect();
        let result_storage = Dense::linear_combine(&raw_tensors, coefs)?;

        Ok(Tensor {
            storage: result_storage,
            backend,
        })
    }

    /// Add all tensors (coefficients all = 1).
    pub fn add_all(tensors: &[&Tensor<Dense<S>, B>]) -> Result<Tensor<Dense<S>, B>, String> {
        let coefs = vec![S::one(); tensors.len()];
        Self::linear_combine(tensors, &coefs)
    }
}

// ============================================================================
// Dense-specific norm and normalization operations (all backends)
// ============================================================================

impl<S, B: ComputeBackend> Tensor<Dense<S>, B>
where
    S: Scalar,
{
    /// Compute Frobenius norm.
    pub fn norm(&self) -> S::Real {
        self.storage.norm()
    }

    /// Normalize to unit norm (in-place).
    ///
    /// Returns the norm before normalization.
    pub fn normalize(&mut self) -> S::Real {
        self.storage.normalize()
    }

    /// Normalize and return new tensor (out-of-place).
    ///
    /// Returns `(normalized_tensor, original_norm)`.
    pub fn normalized(&self) -> (Self, S::Real) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tensor_mutation<S>(zero: S, val: S, fill_val: S, scale_factor: S)
    where
        S: Scalar + PartialEq + std::fmt::Debug + Mul<S, Output = S>,
    {
        let mut t = Tensor::<Dense<S>>::zeros(vec![2, 3]);

        // set / get round-trip
        t.set(&[1, 2], val);
        assert_eq!(t.get(&[1, 2]), val);
        assert_eq!(t.get(&[0, 0]), zero);

        // fill overwrites all elements
        t.fill(fill_val);
        assert_eq!(t.get(&[0, 0]), fill_val);
        assert_eq!(t.get(&[1, 2]), fill_val);

        // data_mut provides mutable access
        t.data_mut()[0] = val;
        assert_eq!(t.get(&[0, 0]), val);

        // scale multiplies all elements
        t.fill(val);
        t.scale(scale_factor);
        assert_eq!(t.get(&[0, 0]), val * scale_factor);
    }

    #[test]
    fn test_tensor_mutation() {
        assert_tensor_mutation(0.0f64, 42.0, 2.718, 3.0);
        assert_tensor_mutation(0.0f32, 42.0, 2.718, 3.0);
    }
}
