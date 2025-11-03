//! Fat tensor with metadata (storage + indices)

use crate::index::IndexSet;
use crate::raw_tensor::RawTensor;
use num_complex::Complex;
use num_traits::{One, Zero};
use std::ops::{Add, Mul};

/// Fat tensor: RawTensor + Index metadata
///
/// This is the main tensor type for tensor network computations.
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64). See [`DenseTensor`](crate::DenseTensor) for details.
#[derive(Debug, Clone)]
pub struct FatTensor<T = f64> {
    pub tensor: RawTensor<T>,
    pub indices: IndexSet,
}

impl<T> FatTensor<T> {
    /// Create a new FatTensor
    pub fn new(tensor: RawTensor<T>, indices: IndexSet) -> Self {
        // TODO: Validate that tensor rank matches number of indices
        Self { tensor, indices }
    }

    /// Get the shape of the underlying tensor
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.tensor.rank()
    }
}

// ============================================================================
// Arithmetic operations
// ============================================================================

impl<T> FatTensor<T>
where
    T: Clone + Mul<Output = T>,
{
    /// Scale tensor by a scalar factor (in-place)
    ///
    /// Preserves indices.
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor, Index, IndexSet};
    ///
    /// let raw = RawTensor::<f64>::ones(vec![2, 3]);
    /// let indices = IndexSet::new(vec![Index::with_dim("i", 2), Index::with_dim("j", 3)], 0);
    /// let mut fat = FatTensor::new(raw, indices);
    ///
    /// fat.scale(2.5);
    /// ```
    pub fn scale(&mut self, factor: T) {
        self.tensor.scale(factor);
    }

    /// Scale tensor and return new tensor (out-of-place)
    ///
    /// Preserves indices.
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor, Index, IndexSet};
    ///
    /// let raw = RawTensor::<f64>::ones(vec![2, 2]);
    /// let indices = IndexSet::new(vec![Index::with_dim("a", 2), Index::with_dim("b", 2)], 0);
    /// let fat = FatTensor::new(raw, indices);
    ///
    /// let scaled = fat.scaled(3.0);
    /// ```
    pub fn scaled(&self, factor: T) -> Self {
        Self {
            tensor: self.tensor.scaled(factor),
            indices: self.indices.clone(),
        }
    }
}

impl<T> FatTensor<T>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    /// Linear combination of tensors (validates index compatibility)
    ///
    /// All tensors must have matching indices.
    ///
    /// # Errors
    /// - Tensors have different indices
    /// - Empty input
    /// - Mismatched lengths
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor, Index, IndexSet};
    ///
    /// let indices = IndexSet::new(vec![Index::with_dim("i", 2)], 0);
    ///
    /// let a = FatTensor::new(
    ///     RawTensor::<f64>::constant(vec![2], 1.0),
    ///     indices.clone(),
    /// );
    /// let b = FatTensor::new(
    ///     RawTensor::<f64>::constant(vec![2], 2.0),
    ///     indices.clone(),
    /// );
    ///
    /// // 2*a + 3*b = 2*1 + 3*2 = 8
    /// let result = FatTensor::linear_combine(&[&a, &b], &[2.0, 3.0]).unwrap();
    /// ```
    pub fn linear_combine(
        tensors: &[&FatTensor<T>],
        coefs: &[T],
    ) -> Result<FatTensor<T>, String> {
        if tensors.is_empty() {
            return Err("Cannot combine empty tensor list".to_string());
        }

        // Validate indices match
        let indices = &tensors[0].indices;
        for t in &tensors[1..] {
            if &t.indices != indices {
                return Err("All tensors must have matching indices".to_string());
            }
        }

        // Delegate to RawTensor
        let raw_tensors: Vec<_> = tensors.iter().map(|t| &t.tensor).collect();
        let result_tensor = RawTensor::linear_combine(&raw_tensors, coefs)?;

        Ok(FatTensor {
            tensor: result_tensor,
            indices: indices.clone(),
        })
    }

    /// Add all tensors (coefficients all = 1)
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor, Index, IndexSet};
    ///
    /// let indices = IndexSet::new(vec![Index::with_dim("x", 2)], 0);
    ///
    /// let a = FatTensor::new(RawTensor::<f64>::constant(vec![2], 1.0), indices.clone());
    /// let b = FatTensor::new(RawTensor::<f64>::constant(vec![2], 2.0), indices.clone());
    ///
    /// let result = FatTensor::add_all(&[&a, &b]).unwrap();
    /// ```
    pub fn add_all(tensors: &[&FatTensor<T>]) -> Result<FatTensor<T>, String> {
        let coefs = vec![T::one(); tensors.len()];
        Self::linear_combine(tensors, &coefs)
    }
}

// ============================================================================
// Norm and normalization operations
// ============================================================================

impl FatTensor<f64> {
    /// Compute Frobenius norm
    pub fn norm(&self) -> f64 {
        self.tensor.norm()
    }

    /// Normalize to unit norm (in-place)
    pub fn normalize(&mut self) -> f64 {
        self.tensor.normalize()
    }

    /// Normalize and return new tensor (out-of-place)
    pub fn normalized(&self) -> (Self, f64) {
        let (normalized_tensor, norm) = self.tensor.normalized();
        (
            Self {
                tensor: normalized_tensor,
                indices: self.indices.clone(),
            },
            norm,
        )
    }
}

impl FatTensor<f32> {
    /// Compute Frobenius norm
    pub fn norm(&self) -> f32 {
        self.tensor.norm()
    }

    /// Normalize to unit norm (in-place)
    pub fn normalize(&mut self) -> f32 {
        self.tensor.normalize()
    }

    /// Normalize and return new tensor (out-of-place)
    pub fn normalized(&self) -> (Self, f32) {
        let (normalized_tensor, norm) = self.tensor.normalized();
        (
            Self {
                tensor: normalized_tensor,
                indices: self.indices.clone(),
            },
            norm,
        )
    }
}

impl FatTensor<Complex<f64>> {
    /// Compute Frobenius norm (for complex tensors)
    pub fn norm(&self) -> f64 {
        self.tensor.norm()
    }

    /// Normalize complex tensor to unit norm (in-place)
    pub fn normalize(&mut self) -> f64 {
        self.tensor.normalize()
    }

    /// Normalize complex tensor and return new tensor (out-of-place)
    pub fn normalized(&self) -> (Self, f64) {
        let (normalized_tensor, norm) = self.tensor.normalized();
        (
            Self {
                tensor: normalized_tensor,
                indices: self.indices.clone(),
            },
            norm,
        )
    }
}

impl FatTensor<Complex<f32>> {
    /// Compute Frobenius norm (for complex tensors)
    pub fn norm(&self) -> f32 {
        self.tensor.norm()
    }

    /// Normalize complex tensor to unit norm (in-place)
    pub fn normalize(&mut self) -> f32 {
        self.tensor.normalize()
    }

    /// Normalize complex tensor and return new tensor (out-of-place)
    pub fn normalized(&self) -> (Self, f32) {
        let (normalized_tensor, norm) = self.tensor.normalized();
        (
            Self {
                tensor: normalized_tensor,
                indices: self.indices.clone(),
            },
            norm,
        )
    }
}
