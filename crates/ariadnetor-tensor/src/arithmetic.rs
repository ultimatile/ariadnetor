//! Arithmetic operations for tensors
//!
//! This module implements TCI-spec arithmetic operations:
//! - `scale`: Scalar multiplication
//! - `linear_combine`: Linear combination of tensors
//! - `norm`: Frobenius norm
//! - `normalize`: Normalize to unit norm

use crate::raw_tensor::RawTensor;
use num_complex::Complex;
use num_traits::{One, Zero};
use std::ops::{Add, Mul};

// ============================================================================
// RawTensor arithmetic operations
// ============================================================================

impl<T> RawTensor<T>
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
    /// use arnet_tensor::RawTensor;
    ///
    /// let mut tensor = RawTensor::<f64>::ones(vec![2, 3]);
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
    /// use arnet_tensor::RawTensor;
    ///
    /// let tensor = RawTensor::<f64>::ones(vec![2, 2]);
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

impl<T> RawTensor<T>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
{
    /// Linear combination: sum_i coefs[i] * tensors[i]
    ///
    /// Forms `out` as Σ coefs[i] * tensors[i].
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
    /// use arnet_tensor::RawTensor;
    ///
    /// let a = RawTensor::<f64>::constant(vec![2, 2], 1.0);
    /// let b = RawTensor::<f64>::constant(vec![2, 2], 2.0);
    ///
    /// // 2*a + 3*b = 2*1 + 3*2 = 8
    /// let result = RawTensor::linear_combine(&[&a, &b], &[2.0, 3.0]).unwrap();
    /// assert_eq!(result.get(&[0, 0]), 8.0);
    /// ```
    pub fn linear_combine(
        tensors: &[&RawTensor<T>],
        coefs: &[T],
    ) -> Result<RawTensor<T>, String> {
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
    /// use arnet_tensor::RawTensor;
    ///
    /// let a = RawTensor::<f64>::constant(vec![2, 2], 1.0);
    /// let b = RawTensor::<f64>::constant(vec![2, 2], 2.0);
    /// let c = RawTensor::<f64>::constant(vec![2, 2], 3.0);
    ///
    /// // a + b + c = 6
    /// let result = RawTensor::add_all(&[&a, &b, &c]).unwrap();
    /// assert_eq!(result.get(&[0, 0]), 6.0);
    /// ```
    pub fn add_all(tensors: &[&RawTensor<T>]) -> Result<RawTensor<T>, String>
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

// NOTE: Due to Rust's orphan rules and potential trait implementation conflicts,
// we implement norm/normalize operations for specific types rather than using
// generic Float bounds. This is a temporary solution until we migrate to the
// Storage/Compute separation pattern (see dev-docs/design/future_dtype_system.md).

// Real f64
impl RawTensor<f64> {
    /// Compute Frobenius norm
    ///
    /// Returns √(Σ element²)
    ///
    /// # TCI-spec
    /// Corresponds to `tci::norm`
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    fn norm_squared(&self) -> f64 {
        match self {
            Self::Dense(d) => d.data().iter().map(|&x| x * x).sum(),
        }
    }

    /// Normalize tensor to unit norm (in-place)
    ///
    /// # TCI-spec
    /// Corresponds to `tci::normalize` overload (1)
    pub fn normalize(&mut self) -> f64 {
        let norm = self.norm();
        if norm == 0.0 {
            panic!("Cannot normalize zero tensor");
        }
        self.scale(1.0 / norm);
        norm
    }

    /// Normalize tensor and return new tensor (out-of-place)
    ///
    /// # TCI-spec
    /// Corresponds to `tci::normalize` overload (2)
    pub fn normalized(&self) -> (Self, f64) {
        let mut result = self.clone();
        let norm = result.normalize();
        (result, norm)
    }
}

// Real f32
impl RawTensor<f32> {
    /// Compute Frobenius norm
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    fn norm_squared(&self) -> f32 {
        match self {
            Self::Dense(d) => d.data().iter().map(|&x| x * x).sum(),
        }
    }

    /// Normalize tensor to unit norm (in-place)
    pub fn normalize(&mut self) -> f32 {
        let norm = self.norm();
        if norm == 0.0 {
            panic!("Cannot normalize zero tensor");
        }
        self.scale(1.0 / norm);
        norm
    }

    /// Normalize tensor and return new tensor (out-of-place)
    pub fn normalized(&self) -> (Self, f32) {
        let mut result = self.clone();
        let norm = result.normalize();
        (result, norm)
    }
}

// Complex f64
impl RawTensor<Complex<f64>> {
    /// Compute Frobenius norm (for complex tensors)
    ///
    /// Returns √(Σ |element|²) as a real value
    ///
    /// # TCI-spec
    /// Corresponds to `tci::norm`
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    fn norm_squared(&self) -> f64 {
        match self {
            Self::Dense(d) => d.data().iter().map(|z| z.re * z.re + z.im * z.im).sum(),
        }
    }

    /// Normalize complex tensor to unit norm (in-place)
    pub fn normalize(&mut self) -> f64 {
        let norm = self.norm();
        if norm == 0.0 {
            panic!("Cannot normalize zero tensor");
        }
        self.scale(Complex::new(1.0 / norm, 0.0));
        norm
    }

    /// Normalize complex tensor and return new tensor (out-of-place)
    pub fn normalized(&self) -> (Self, f64) {
        let mut result = self.clone();
        let norm = result.normalize();
        (result, norm)
    }
}

// Complex f32
impl RawTensor<Complex<f32>> {
    /// Compute Frobenius norm (for complex tensors)
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    fn norm_squared(&self) -> f32 {
        match self {
            Self::Dense(d) => d.data().iter().map(|z| z.re * z.re + z.im * z.im).sum(),
        }
    }

    /// Normalize complex tensor to unit norm (in-place)
    pub fn normalize(&mut self) -> f32 {
        let norm = self.norm();
        if norm == 0.0 {
            panic!("Cannot normalize zero tensor");
        }
        self.scale(Complex::new(1.0 / norm, 0.0));
        norm
    }

    /// Normalize complex tensor and return new tensor (out-of-place)
    pub fn normalized(&self) -> (Self, f32) {
        let mut result = self.clone();
        let norm = result.normalize();
        (result, norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_scale_basic() {
        let mut tensor = RawTensor::<f64>::ones(vec![2, 2]);
        tensor.scale(3.0);
        assert_eq!(tensor.get(&[0, 0]), 3.0);
        assert_eq!(tensor.get(&[1, 1]), 3.0);
    }

    #[test]
    fn test_scaled_immutable() {
        let tensor = RawTensor::<f64>::constant(vec![2, 2], 2.0);
        let scaled = tensor.scaled(5.0);
        assert_eq!(tensor.get(&[0, 0]), 2.0);
        assert_eq!(scaled.get(&[0, 0]), 10.0);
    }

    #[test]
    fn test_scale_complex() {
        let mut tensor = RawTensor::<Complex<f64>>::ones(vec![2, 2]);
        tensor.scale(Complex::new(2.0, 3.0));
        // (1 + 0i) * (2 + 3i) = (2 + 3i)
        assert_eq!(tensor.get(&[0, 0]), Complex::new(2.0, 3.0));
    }

    #[test]
    fn test_linear_combine_basic() {
        let a = RawTensor::<f64>::constant(vec![2, 2], 1.0);
        let b = RawTensor::<f64>::constant(vec![2, 2], 2.0);
        let result = RawTensor::linear_combine(&[&a, &b], &[3.0, 4.0]).unwrap();
        // 3*1 + 4*2 = 11
        assert_eq!(result.get(&[0, 0]), 11.0);
    }

    #[test]
    fn test_add_all_basic() {
        let a = RawTensor::<f64>::constant(vec![2, 2], 1.0);
        let b = RawTensor::<f64>::constant(vec![2, 2], 2.0);
        let c = RawTensor::<f64>::constant(vec![2, 2], 3.0);
        let result = RawTensor::add_all(&[&a, &b, &c]).unwrap();
        assert_eq!(result.get(&[0, 0]), 6.0);
    }
}
