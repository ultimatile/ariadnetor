//! Tensor with storage abstraction

use crate::tensor_storage::TensorStorage;
use num_traits::{One, Zero};
use std::ops::{Add, Mul};

/// Tensor wrapping a TensorStorage
///
/// This is the main tensor type for tensor network computations.
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64). See [`DenseTensor`](crate::DenseTensor) for details.
#[derive(Debug, Clone)]
pub struct Tensor<T = f64> {
    pub storage: TensorStorage<T>,
}

impl<T> Tensor<T> {
    /// Create a new Tensor from storage
    pub fn new(storage: TensorStorage<T>) -> Self {
        Self { storage }
    }

    /// Get the shape of the underlying tensor
    pub fn shape(&self) -> &[usize] {
        self.storage.shape()
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.storage.rank()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
}

impl<T> Tensor<T>
where
    T: Clone,
{
    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero,
    {
        Self::new(TensorStorage::zeros(shape))
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Zero,
    {
        Self::new(TensorStorage::ones(shape))
    }

    /// Create a tensor filled with a constant value
    pub fn constant(shape: Vec<usize>, value: T) -> Self {
        Self::new(TensorStorage::constant(shape, value))
    }

    /// Create a tensor from existing data
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::new(TensorStorage::from_data(data, shape))
    }

    /// Get a reference to the underlying data (only for Dense)
    pub fn data(&self) -> Option<&[T]> {
        self.storage.data()
    }

    /// Get a mutable reference to the underlying data (only for Dense)
    pub fn data_mut(&mut self) -> Option<&mut [T]> {
        self.storage.data_mut()
    }

    /// Get element at given indices
    pub fn get(&self, indices: &[usize]) -> T {
        self.storage.get(indices)
    }

    /// Set element at given indices
    pub fn set(&mut self, indices: &[usize], value: T) {
        self.storage.set(indices, value)
    }

    /// Fill tensor with a constant value
    pub fn fill(&mut self, value: T) {
        self.storage.fill(value)
    }
}

// ============================================================================
// Arithmetic operations
// ============================================================================

impl<T> Tensor<T>
where
    T: Clone + Mul<Output = T>,
{
    /// Scale tensor by a scalar factor (in-place)
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::Tensor;
    ///
    /// let mut t = Tensor::<f64>::ones(vec![2, 3]);
    /// t.scale(2.5);
    /// ```
    pub fn scale(&mut self, factor: T) {
        self.storage.scale(factor);
    }

    /// Scale tensor and return new tensor (out-of-place)
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::Tensor;
    ///
    /// let t = Tensor::<f64>::ones(vec![2, 2]);
    /// let scaled = t.scaled(3.0);
    /// ```
    pub fn scaled(&self, factor: T) -> Self {
        Self {
            storage: self.storage.scaled(factor),
        }
    }
}

impl<T> Tensor<T>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    /// Linear combination of tensors
    ///
    /// All tensors must have the same shape.
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::Tensor;
    ///
    /// let a = Tensor::<f64>::constant(vec![2], 1.0);
    /// let b = Tensor::<f64>::constant(vec![2], 2.0);
    ///
    /// // 2*a + 3*b = 2*1 + 3*2 = 8
    /// let result = Tensor::linear_combine(&[&a, &b], &[2.0, 3.0]).unwrap();
    /// ```
    pub fn linear_combine(tensors: &[&Tensor<T>], coefs: &[T]) -> Result<Tensor<T>, String> {
        if tensors.is_empty() {
            return Err("Cannot combine empty tensor list".to_string());
        }

        let raw_tensors: Vec<_> = tensors.iter().map(|t| &t.storage).collect();
        let result_storage = TensorStorage::linear_combine(&raw_tensors, coefs)?;

        Ok(Tensor {
            storage: result_storage,
        })
    }

    /// Add all tensors (coefficients all = 1)
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::Tensor;
    ///
    /// let a = Tensor::<f64>::constant(vec![2], 1.0);
    /// let b = Tensor::<f64>::constant(vec![2], 2.0);
    ///
    /// let result = Tensor::add_all(&[&a, &b]).unwrap();
    /// ```
    pub fn add_all(tensors: &[&Tensor<T>]) -> Result<Tensor<T>, String> {
        let coefs = vec![T::one(); tensors.len()];
        Self::linear_combine(tensors, &coefs)
    }
}

// ============================================================================
// Norm and normalization operations
// ============================================================================

use crate::Scalar;

impl<T> Tensor<T>
where
    T: Scalar,
{
    /// Compute Frobenius norm
    ///
    /// Returns sqrt(sum |element|^2) as a real value
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::Tensor;
    ///
    /// let t = Tensor::<f64>::ones(vec![2, 3]);
    /// let norm = t.norm();
    /// assert!((norm - 6.0f64.sqrt()).abs() < 1e-10);
    /// ```
    pub fn norm(&self) -> T::Real {
        self.storage.norm()
    }

    /// Normalize to unit norm (in-place)
    ///
    /// Returns the norm before normalization.
    /// Panics if the tensor has zero norm.
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::Tensor;
    ///
    /// let mut t = Tensor::<f64>::ones(vec![2, 2]);
    /// let norm = t.normalize();
    /// assert!((norm - 2.0).abs() < 1e-10);
    /// assert!((t.norm() - 1.0).abs() < 1e-10);
    /// ```
    pub fn normalize(&mut self) -> T::Real {
        self.storage.normalize()
    }

    /// Normalize and return new tensor (out-of-place)
    ///
    /// Returns `(normalized_tensor, original_norm)`.
    /// Panics if the tensor has zero norm.
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::Tensor;
    ///
    /// let t = Tensor::<f64>::constant(vec![3, 3], 2.0);
    /// let (normalized, norm) = t.normalized();
    /// assert!((norm - 6.0).abs() < 1e-10);
    /// assert!((normalized.norm() - 1.0).abs() < 1e-10);
    /// ```
    pub fn normalized(&self) -> (Self, T::Real) {
        let (normalized_storage, norm) = self.storage.normalized();
        (
            Self {
                storage: normalized_storage,
            },
            norm,
        )
    }
}
