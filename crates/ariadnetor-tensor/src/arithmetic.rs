//! Arithmetic operations for tensors
//!
//! This module implements TCI-spec arithmetic operations:
//! - `scale`: Scalar multiplication
//! - `linear_combine`: Linear combination of tensors
//! - `norm`: Frobenius norm
//! - `normalize`: Normalize to unit norm

use crate::tensor_storage::TensorStorage;
use num_traits::{One, Zero};
use std::ops::{Add, Mul};

// ============================================================================
// TensorStorage arithmetic operations
// ============================================================================

impl<T> TensorStorage<T>
where
    T: Clone + Mul<Output = T>,
{
    /// Scale tensor by a scalar factor (in-place)
    ///
    /// Multiplies every element by `factor`.
    ///
    /// # TCI-spec
    /// Corresponds to `tci::scale` overload (1)
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::TensorStorage;
    ///
    /// let mut tensor = TensorStorage::<f64>::ones(vec![2, 3]);
    /// tensor.scale(2.5);
    /// assert_eq!(tensor.get(&[0, 0]), 2.5);
    /// ```
    pub fn scale(&mut self, factor: T) {
        match self {
            Self::Dense(d) => {
                for elem in d.data_mut() {
                    *elem = elem.clone() * factor.clone();
                }
            }
        }
    }

    /// Scale tensor and return new tensor (out-of-place)
    ///
    /// Creates a new tensor with all elements multiplied by `factor`.
    ///
    /// # TCI-spec
    /// Corresponds to `tci::scale` overload (2)
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::TensorStorage;
    ///
    /// let tensor = TensorStorage::<f64>::ones(vec![2, 2]);
    /// let scaled = tensor.scaled(3.0);
    /// assert_eq!(scaled.get(&[0, 0]), 3.0);
    /// assert_eq!(tensor.get(&[0, 0]), 1.0); // Original unchanged
    /// ```
    pub fn scaled(&self, factor: T) -> Self {
        let mut result = self.clone();
        result.scale(factor);
        result
    }
}

impl<T> TensorStorage<T>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
{
    /// Linear combination: Σ coefs\[i\] * tensors\[i\]
    ///
    /// Forms `out` as Σ coefs\[i\] * tensors\[i\].
    ///
    /// # TCI-spec
    /// Corresponds to `tci::linear_combine` overload (2)
    ///
    /// # Errors
    /// - Tensors have different shapes
    /// - Empty input
    /// - Mismatched lengths between tensors and coefficients
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::TensorStorage;
    ///
    /// let a = TensorStorage::<f64>::constant(vec![2, 2], 1.0);
    /// let b = TensorStorage::<f64>::constant(vec![2, 2], 2.0);
    ///
    /// // 2*a + 3*b = 2*1 + 3*2 = 8
    /// let result = TensorStorage::linear_combine(&[&a, &b], &[2.0, 3.0]).unwrap();
    /// assert_eq!(result.get(&[0, 0]), 8.0);
    /// ```
    pub fn linear_combine(
        tensors: &[&TensorStorage<T>],
        coefs: &[T],
    ) -> Result<TensorStorage<T>, String> {
        if tensors.is_empty() {
            return Err("Cannot combine empty tensor list".to_string());
        }
        if tensors.len() != coefs.len() {
            return Err(format!(
                "Mismatched lengths: {} tensors vs {} coefficients",
                tensors.len(),
                coefs.len()
            ));
        }

        // Check shape consistency
        let shape = tensors[0].shape();
        for t in &tensors[1..] {
            if t.shape() != shape {
                return Err("All tensors must have the same shape".to_string());
            }
        }

        // Compute sum
        match tensors[0] {
            Self::Dense(_) => {
                let mut result = Self::zeros(shape.to_vec());
                let Self::Dense(result_dense) = &mut result;
                for (tensor, coef) in tensors.iter().zip(coefs) {
                    let Self::Dense(t) = tensor;
                    for (res, val) in result_dense.data_mut().iter_mut().zip(t.data()) {
                        *res = res.clone() + coef.clone() * val.clone();
                    }
                }
                Ok(result)
            }
        }
    }

    /// Add all tensors (coefficients all = 1)
    ///
    /// # TCI-spec
    /// Corresponds to `tci::linear_combine` overload (1)
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::TensorStorage;
    ///
    /// let a = TensorStorage::<f64>::constant(vec![2, 2], 1.0);
    /// let b = TensorStorage::<f64>::constant(vec![2, 2], 2.0);
    /// let c = TensorStorage::<f64>::constant(vec![2, 2], 3.0);
    ///
    /// // a + b + c = 6
    /// let result = TensorStorage::add_all(&[&a, &b, &c]).unwrap();
    /// assert_eq!(result.get(&[0, 0]), 6.0);
    /// ```
    pub fn add_all(tensors: &[&TensorStorage<T>]) -> Result<TensorStorage<T>, String>
    where
        T: One,
    {
        let coefs = vec![T::one(); tensors.len()];
        Self::linear_combine(tensors, &coefs)
    }
}

// ============================================================================
// Norm and normalization operations
// ============================================================================

use crate::Scalar;
use num_traits::Float;

impl<T> TensorStorage<T>
where
    T: Scalar,
{
    /// Compute Frobenius norm
    ///
    /// Returns √(Σ |element|²) as a real value
    ///
    /// For real tensors: √(Σ element²)
    /// For complex tensors: √(Σ |element|²)
    ///
    /// # TCI-spec
    /// Corresponds to `tci::norm`
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::TensorStorage;
    ///
    /// let tensor = TensorStorage::<f64>::ones(vec![2, 2]);
    /// let norm = tensor.norm();
    /// assert!((norm - 2.0).abs() < 1e-10);
    /// ```
    pub fn norm(&self) -> T::Real {
        self.norm_squared().sqrt()
    }

    /// Compute squared Frobenius norm
    fn norm_squared(&self) -> T::Real {
        match self {
            Self::Dense(d) => d
                .data()
                .iter()
                .map(|&x| {
                    let abs_val = x.abs();
                    abs_val * abs_val
                })
                .fold(T::Real::zero(), |acc, x| acc + x),
        }
    }

    /// Normalize tensor to unit norm (in-place)
    ///
    /// Returns the norm before normalization.
    /// Panics if the tensor has zero norm.
    ///
    /// # TCI-spec
    /// Corresponds to `tci::normalize` overload (1)
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::TensorStorage;
    ///
    /// let mut tensor = TensorStorage::<f64>::ones(vec![2, 2]);
    /// let norm = tensor.normalize();
    /// assert!((norm - 2.0).abs() < 1e-10);
    /// assert!((tensor.norm() - 1.0).abs() < 1e-10);
    /// ```
    pub fn normalize(&mut self) -> T::Real {
        let norm = self.norm();
        if norm == T::Real::zero() {
            panic!("Cannot normalize zero tensor");
        }
        let inv_norm = T::Real::one() / norm;

        match self {
            Self::Dense(d) => {
                for elem in d.data_mut() {
                    *elem = elem.scale_real(inv_norm);
                }
            }
        }
        norm
    }

    /// Normalize tensor and return new tensor (out-of-place)
    ///
    /// Returns `(normalized_tensor, original_norm)`.
    /// Panics if the tensor has zero norm.
    ///
    /// # TCI-spec
    /// Corresponds to `tci::normalize` overload (2)
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::TensorStorage;
    ///
    /// let tensor = TensorStorage::<f64>::ones(vec![3, 3]);
    /// let (normalized, norm) = tensor.normalized();
    /// assert!((norm - 3.0).abs() < 1e-10);
    /// assert!((normalized.norm() - 1.0).abs() < 1e-10);
    /// ```
    pub fn normalized(&self) -> (Self, T::Real) {
        let mut result = self.clone();
        let norm = result.normalize();
        (result, norm)
    }
}
