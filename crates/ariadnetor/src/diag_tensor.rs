//! Diagonal tensor type
//!
//! `DiagTensor<S, B>` is a newtype wrapping a rank-1 `Tensor<Dense<S>, B>` that
//! represents an n×n diagonal matrix by storing only the n diagonal elements.

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_native::NativeBackend;
use arnet_tensor::Dense;

use crate::Tensor;

/// Diagonal matrix represented as a rank-1 tensor of diagonal elements.
///
/// Wraps a rank-1 [`Tensor<Dense<S>, B>`] and interprets it as an n×n diagonal
/// matrix where only the n diagonal entries are stored (O(n) instead of O(n²)).
///
/// # Examples
///
/// ```
/// use arnet::DiagTensor;
///
/// let d = DiagTensor::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(d.len(), 3);
/// assert_eq!(d.matrix_size(), [3, 3]);
/// assert_eq!(d.data(), &[1.0, 2.0, 3.0]);
/// ```
#[derive(Debug, Clone)]
pub struct DiagTensor<S = f64, B: ComputeBackend = NativeBackend>(Tensor<Dense<S>, B>);

impl<S: Scalar, B: ComputeBackend> DiagTensor<S, B> {
    /// Create a `DiagTensor` from a rank-1 tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not rank 1.
    pub fn from_tensor(tensor: Tensor<Dense<S>, B>) -> Result<Self, String> {
        if tensor.rank() != 1 {
            return Err(format!(
                "DiagTensor requires a rank-1 tensor, got rank {}",
                tensor.rank()
            ));
        }
        Ok(Self(tensor))
    }

    /// Number of diagonal elements (= n for an n×n matrix).
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if diagonal has zero elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Matrix dimensions `[n, n]` that this diagonal represents.
    pub fn matrix_size(&self) -> [usize; 2] {
        let n = self.len();
        [n, n]
    }

    /// Get a reference to the underlying rank-1 tensor.
    pub fn as_tensor(&self) -> &Tensor<Dense<S>, B> {
        &self.0
    }

    /// Consume the `DiagTensor` and return the underlying tensor.
    pub fn into_tensor(self) -> Tensor<Dense<S>, B> {
        self.0
    }

    /// Create a `DiagTensor` from a vector and an explicit backend.
    pub fn from_vec_with_backend(diag: Vec<S>, backend: Arc<B>) -> Self {
        let n = diag.len();
        let storage = Dense::new(diag, vec![n]);
        Self(Tensor::with_backend(storage, backend))
    }

    /// Get diagonal elements as a slice.
    pub fn data(&self) -> &[S] {
        self.0.data()
    }
}

impl<S: Scalar, B: ComputeBackend> DiagTensor<S, B> {
    /// Extract the diagonal of a square matrix as a `DiagTensor`.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is not a rank-2 square matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use arnet::{Dense, DiagTensor, Tensor};
    /// use arnet_native::NativeBackend;
    ///
    /// // Diagonal matrix: data stored in column-major (NativeBackend order)
    /// let dense = Dense::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
    /// let t = Tensor::with_backend(dense, NativeBackend::shared());
    /// let d = DiagTensor::from_matrix(&t).unwrap();
    /// assert_eq!(d.data(), &[1.0, 2.0]);
    /// ```
    pub fn from_matrix(tensor: &Tensor<Dense<S>, B>) -> Result<Self, String> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(format!(
                "from_matrix requires a rank-2 tensor, got rank {}",
                shape.len()
            ));
        }
        if shape[0] != shape[1] {
            return Err(format!(
                "from_matrix requires a square matrix, got {}×{}",
                shape[0], shape[1]
            ));
        }
        let result = arnet_linalg::diag(&tensor.storage)?;
        Ok(Self(Tensor::with_backend(
            result,
            Arc::clone(tensor.backend_arc()),
        )))
    }

    /// Expand to a full n×n dense matrix.
    ///
    /// Off-diagonal elements are zero. Data layout is order-agnostic:
    /// for diagonal matrices, RM and CM flat layouts are identical
    /// (`data[i*n + i]` holds the diagonal element for both orders).
    ///
    /// # Examples
    ///
    /// ```
    /// use arnet::DiagTensor;
    ///
    /// let d = DiagTensor::from_vec(vec![2.0, 3.0]);
    /// let dense = d.to_dense();
    /// assert_eq!(dense.shape(), &[2, 2]);
    /// assert_eq!(dense.get(&[0, 0]), 2.0);
    /// assert_eq!(dense.get(&[0, 1]), 0.0);
    /// assert_eq!(dense.get(&[1, 0]), 0.0);
    /// assert_eq!(dense.get(&[1, 1]), 3.0);
    /// ```
    pub fn to_dense(&self) -> Tensor<Dense<S>, B> {
        let diag = self.data();
        let n = diag.len();
        let mut data = vec![S::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = diag[i];
        }
        let dense = Dense::new(data, vec![n, n]);
        Tensor::with_backend(dense, Arc::clone(self.0.backend_arc()))
    }
}

impl<S: Scalar> DiagTensor<S, NativeBackend> {
    /// Create a `DiagTensor` from a vector (default: NativeBackend).
    pub fn from_vec(diag: Vec<S>) -> Self {
        Self::from_vec_with_backend(diag, NativeBackend::shared())
    }
}

impl<S: Scalar, B: ComputeBackend> Deref for DiagTensor<S, B> {
    type Target = Tensor<Dense<S>, B>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S, B: ComputeBackend> fmt::Display for DiagTensor<S, B>
where
    S: Scalar + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiagTensor(")?;
        let data = self.data();
        for (i, val) in data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{val}")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_basic() {
        let d = DiagTensor::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(d.len(), 3);
        assert!(!d.is_empty());
        assert_eq!(d.matrix_size(), [3, 3]);
        assert_eq!(d.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_vec_empty() {
        let d = DiagTensor::<f64>::from_vec(vec![]);
        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        assert_eq!(d.matrix_size(), [0, 0]);
    }

    #[test]
    fn from_tensor_rank1() {
        let t = Tensor::<Dense<f64>>::constant(vec![3], 5.0);
        let d = DiagTensor::from_tensor(t).unwrap();
        assert_eq!(d.len(), 3);
        assert_eq!(d.data(), &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn from_tensor_rejects_rank2() {
        let t = Tensor::<Dense<f64>>::zeros(vec![2, 2]);
        let err = DiagTensor::from_tensor(t).unwrap_err();
        assert!(err.contains("rank-1"));
    }

    #[test]
    fn to_dense_identity_like() {
        let d = DiagTensor::from_vec(vec![1.0, 1.0, 1.0]);
        let dense = d.to_dense();
        assert_eq!(dense.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(dense.get(&[i, j]), expected);
            }
        }
    }

    #[test]
    fn to_dense_values() {
        let d = DiagTensor::from_vec(vec![2.0, 3.0, 5.0]);
        let dense = d.to_dense();
        assert_eq!(dense.get(&[0, 0]), 2.0);
        assert_eq!(dense.get(&[1, 1]), 3.0);
        assert_eq!(dense.get(&[2, 2]), 5.0);
        assert_eq!(dense.get(&[0, 1]), 0.0);
        assert_eq!(dense.get(&[1, 0]), 0.0);
    }

    #[test]
    fn to_dense_empty() {
        let d = DiagTensor::<f64>::from_vec(vec![]);
        let dense = d.to_dense();
        assert_eq!(dense.shape(), &[0, 0]);
    }

    #[test]
    fn roundtrip_diag_to_dense_to_diag() {
        let original = vec![1.0, 4.0, 9.0];
        let d = DiagTensor::from_vec(original.clone());
        let dense = d.to_dense();

        // Extract diagonal back using get
        let extracted: Vec<f64> = (0..3).map(|i| dense.get(&[i, i])).collect();
        assert_eq!(extracted, original);
    }

    #[test]
    fn deref_to_tensor() {
        let d = DiagTensor::from_vec(vec![1.0, 2.0]);
        // Deref allows calling Tensor methods directly
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.rank(), 1);
    }

    #[test]
    fn display() {
        let d = DiagTensor::from_vec(vec![1.0, 2.0, 3.0]);
        let s = format!("{d}");
        assert_eq!(s, "DiagTensor(1, 2, 3)");
    }

    #[test]
    fn into_tensor() {
        let d = DiagTensor::from_vec(vec![1.0, 2.0]);
        let t = d.into_tensor();
        assert_eq!(t.shape(), &[2]);
        assert_eq!(t.data(), &[1.0, 2.0]);
    }

    #[test]
    fn from_matrix_extracts_diagonal() {
        // Data layout: diag extraction is order-agnostic for square matrices
        // (diagonal elements have equal row/col indices → same flat index in both RM and CM)
        let dense = Dense::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );
        let t = Tensor::with_backend(dense, NativeBackend::shared());
        let d = DiagTensor::from_matrix(&t).unwrap();
        assert_eq!(d.data(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn from_matrix_rejects_non_square() {
        let dense = Dense::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t = Tensor::with_backend(dense, NativeBackend::shared());
        assert!(DiagTensor::<f64>::from_matrix(&t).is_err());
    }

    #[test]
    fn from_matrix_rejects_rank1() {
        let t = Tensor::<Dense<f64>>::constant(vec![3], 1.0);
        assert!(DiagTensor::from_matrix(&t).is_err());
    }

    #[test]
    fn roundtrip_from_matrix_to_dense() {
        let original = DiagTensor::from_vec(vec![2.0, 7.0, 11.0]);
        let dense = original.to_dense();
        let extracted = DiagTensor::from_matrix(&dense).unwrap();
        assert_eq!(extracted.data(), original.data());
    }
}
