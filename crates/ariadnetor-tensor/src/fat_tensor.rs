//! Fat tensor with metadata (storage + labels)

use crate::LabelId;
use crate::raw_tensor::RawTensor;
use num_traits::{One, Zero};
use std::ops::{Add, Mul};

/// Fat tensor: RawTensor + Label metadata
///
/// This is the main tensor type for tensor network computations.
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64). See [`DenseTensor`](crate::DenseTensor) for details.
#[derive(Debug, Clone)]
pub struct FatTensor<T = f64> {
    pub tensor: RawTensor<T>,
    pub labels: Vec<LabelId>,
}

impl<T> FatTensor<T> {
    /// Create a new FatTensor
    pub fn new(tensor: RawTensor<T>, labels: Vec<LabelId>) -> Self {
        assert_eq!(tensor.shape().len(), labels.len());
        Self { tensor, labels }
    }

    /// Create from raw tensor with string labels
    pub fn from_raw(tensor: RawTensor<T>, label_names: &[&str]) -> Self {
        let labels = label_names.iter().map(|s| LabelId::intern(s)).collect();
        Self::new(tensor, labels)
    }

    /// Get the shape of the underlying tensor
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.labels.len()
    }

    /// Get label names as strings (for debugging)
    pub fn label_names(&self) -> Vec<String> {
        self.labels.iter().map(|l| l.name()).collect()
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
    /// Preserves labels.
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor};
    ///
    /// let raw = RawTensor::<f64>::ones(vec![2, 3]);
    /// let mut fat = FatTensor::from_raw(raw, &["i", "j"]);
    ///
    /// fat.scale(2.5);
    /// ```
    pub fn scale(&mut self, factor: T) {
        self.tensor.scale(factor);
    }

    /// Scale tensor and return new tensor (out-of-place)
    ///
    /// Preserves labels.
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor};
    ///
    /// let raw = RawTensor::<f64>::ones(vec![2, 2]);
    /// let fat = FatTensor::from_raw(raw, &["a", "b"]);
    ///
    /// let scaled = fat.scaled(3.0);
    /// ```
    pub fn scaled(&self, factor: T) -> Self {
        Self {
            tensor: self.tensor.scaled(factor),
            labels: self.labels.clone(),
        }
    }
}

impl<T> FatTensor<T>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    /// Linear combination of tensors (validates label compatibility)
    ///
    /// All tensors must have matching labels.
    ///
    /// # Errors
    /// - Tensors have different labels
    /// - Empty input
    /// - Mismatched lengths
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor};
    ///
    /// let a = FatTensor::from_raw(
    ///     RawTensor::<f64>::constant(vec![2], 1.0),
    ///     &["i"],
    /// );
    /// let b = FatTensor::from_raw(
    ///     RawTensor::<f64>::constant(vec![2], 2.0),
    ///     &["i"],
    /// );
    ///
    /// // 2*a + 3*b = 2*1 + 3*2 = 8
    /// let result = FatTensor::linear_combine(&[&a, &b], &[2.0, 3.0]).unwrap();
    /// ```
    pub fn linear_combine(tensors: &[&FatTensor<T>], coefs: &[T]) -> Result<FatTensor<T>, String> {
        if tensors.is_empty() {
            return Err("Cannot combine empty tensor list".to_string());
        }

        // Validate labels match
        let labels = &tensors[0].labels;
        for t in &tensors[1..] {
            if &t.labels != labels {
                return Err("All tensors must have matching labels".to_string());
            }
        }

        // Delegate to RawTensor
        let raw_tensors: Vec<_> = tensors.iter().map(|t| &t.tensor).collect();
        let result_tensor = RawTensor::linear_combine(&raw_tensors, coefs)?;

        Ok(FatTensor {
            tensor: result_tensor,
            labels: labels.clone(),
        })
    }

    /// Add all tensors (coefficients all = 1)
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor};
    ///
    /// let a = FatTensor::from_raw(RawTensor::<f64>::constant(vec![2], 1.0), &["x"]);
    /// let b = FatTensor::from_raw(RawTensor::<f64>::constant(vec![2], 2.0), &["x"]);
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

use crate::Scalar;

impl<T> FatTensor<T>
where
    T: Scalar,
{
    /// Compute Frobenius norm
    ///
    /// Returns √(Σ |element|²) as a real value
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor};
    ///
    /// let raw = RawTensor::<f64>::ones(vec![2, 3]);
    /// let fat = FatTensor::from_raw(raw, &["i", "j"]);
    ///
    /// let norm = fat.norm();
    /// assert!((norm - 6.0f64.sqrt()).abs() < 1e-10);
    /// ```
    pub fn norm(&self) -> T::Real {
        self.tensor.norm()
    }

    /// Normalize to unit norm (in-place)
    ///
    /// Returns the norm before normalization.
    /// Panics if the tensor has zero norm.
    /// Preserves labels.
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor};
    ///
    /// let raw = RawTensor::<f64>::ones(vec![2, 2]);
    /// let mut fat = FatTensor::from_raw(raw, &["a", "b"]);
    ///
    /// let norm = fat.normalize();
    /// assert!((norm - 2.0).abs() < 1e-10);
    /// assert!((fat.norm() - 1.0).abs() < 1e-10);
    /// ```
    pub fn normalize(&mut self) -> T::Real {
        self.tensor.normalize()
    }

    /// Normalize and return new tensor (out-of-place)
    ///
    /// Returns `(normalized_tensor, original_norm)`.
    /// Panics if the tensor has zero norm.
    /// Preserves labels.
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor};
    ///
    /// let raw = RawTensor::<f64>::constant(vec![3, 3], 2.0);
    /// let fat = FatTensor::from_raw(raw, &["x", "y"]);
    ///
    /// let (normalized, norm) = fat.normalized();
    /// assert!((norm - 6.0).abs() < 1e-10);
    /// assert!((normalized.norm() - 1.0).abs() < 1e-10);
    /// ```
    pub fn normalized(&self) -> (Self, T::Real) {
        let (normalized_tensor, norm) = self.tensor.normalized();
        (
            Self {
                tensor: normalized_tensor,
                labels: self.labels.clone(),
            },
            norm,
        )
    }
}

// ============================================================================
// Tensor contraction operations
// ============================================================================

use crate::ContractionError;
use crate::EinsumExpr;
use std::collections::HashMap;

impl<T> FatTensor<T>
where
    T: Clone + Zero + One + std::ops::AddAssign + std::ops::Mul<Output = T> + 'static,
{
    /// Contract two tensors using Einstein notation
    ///
    /// This method performs tensor contraction based on Einstein summation notation.
    /// Labels that appear in both input tensors but not in the output are summed over.
    ///
    /// # Arguments
    /// * `other` - The other tensor to contract with
    /// * `notation` - Einstein notation string (e.g., "ij,jk->ik")
    ///
    /// # Examples
    /// ```
    /// use arnet_tensor::{FatTensor, RawTensor};
    ///
    /// // Matrix multiplication
    /// let a = FatTensor::from_raw(
    ///     RawTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    ///     &["i", "j"]
    /// );
    /// let b = FatTensor::from_raw(
    ///     RawTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
    ///     &["j", "k"]
    /// );
    ///
    /// let c = a.contract(&b, "ij,jk->ik").unwrap();
    /// assert_eq!(c.shape(), &[2, 2]);
    /// assert_eq!(c.label_names(), vec!["i", "k"]);
    /// ```
    ///
    /// # Errors
    /// - `InvalidNotation`: Einstein notation string is malformed
    /// - `LabelMismatch`: Number of labels in notation doesn't match tensor rank
    /// - `DimensionMismatch`: Contracted labels have different dimensions
    /// - `LabelNotFound`: Label in notation not found in tensor
    pub fn contract(&self, other: &Self, notation: &str) -> Result<Self, ContractionError> {
        // Parse Einstein notation
        let expr = EinsumExpr::parse(notation)
            .map_err(ContractionError::InvalidNotation)?;

        // Validate label counts
        if self.labels.len() != expr.lhs_indices.len() {
            return Err(ContractionError::LabelMismatch {
                expected: expr.lhs_indices.len(),
                actual: self.labels.len(),
                tensor: "lhs".to_string(),
            });
        }
        if other.labels.len() != expr.rhs_indices.len() {
            return Err(ContractionError::LabelMismatch {
                expected: expr.rhs_indices.len(),
                actual: other.labels.len(),
                tensor: "rhs".to_string(),
            });
        }

        // Build label name to byte mapping for notation
        let mut label_to_byte: HashMap<String, u8> = HashMap::new();

        // Map lhs labels
        for (&label_id, &notation_byte) in self.labels.iter().zip(expr.lhs_indices.iter()) {
            let label_name = label_id.name();
            label_to_byte.insert(label_name, notation_byte);
        }

        // Map rhs labels (reuse existing mappings)
        for (&label_id, &notation_byte) in other.labels.iter().zip(expr.rhs_indices.iter()) {
            let label_name = label_id.name();
            if let Some(&existing_byte) = label_to_byte.get(&label_name) {
                if existing_byte != notation_byte {
                    return Err(ContractionError::InvalidNotation(
                        format!("Label '{}' mapped to different characters", label_name)
                    ));
                }
            } else {
                label_to_byte.insert(label_name, notation_byte);
            }
        }

        // Validate dimensions for contracted labels
        for (&lhs_label, &lhs_byte) in self.labels.iter().zip(expr.lhs_indices.iter()) {
            for (&rhs_label, &rhs_byte) in other.labels.iter().zip(expr.rhs_indices.iter()) {
                if lhs_byte == rhs_byte && !expr.out_indices.contains(&lhs_byte) {
                    // This is a contracted label
                    let lhs_pos = self.labels.iter().position(|&l| l == lhs_label).unwrap();
                    let rhs_pos = other.labels.iter().position(|&l| l == rhs_label).unwrap();
                    let lhs_dim = self.shape()[lhs_pos];
                    let rhs_dim = other.shape()[rhs_pos];

                    if lhs_dim != rhs_dim {
                        return Err(ContractionError::DimensionMismatch {
                            label: lhs_label.name(),
                            lhs_dim,
                            rhs_dim,
                        });
                    }
                }
            }
        }

        // Extract DenseTensor from RawTensor
        let crate::raw_tensor::RawTensor::Dense(lhs_dense) = &self.tensor;
        let crate::raw_tensor::RawTensor::Dense(rhs_dense) = &other.tensor;

        // Perform contraction using existing contract_naive
        let result_dense = lhs_dense.contract_naive(rhs_dense, notation);

        // Build output labels from notation
        let output_labels: Vec<LabelId> = expr.out_indices.iter()
            .map(|&byte| {
                // Find which label corresponds to this byte
                for (&label_id, &notation_byte) in self.labels.iter().zip(expr.lhs_indices.iter()) {
                    if notation_byte == byte {
                        return label_id;
                    }
                }
                for (&label_id, &notation_byte) in other.labels.iter().zip(expr.rhs_indices.iter()) {
                    if notation_byte == byte {
                        return label_id;
                    }
                }
                // Should not reach here if notation is valid
                LabelId::intern(&(byte as char).to_string())
            })
            .collect();

        Ok(Self {
            tensor: crate::raw_tensor::RawTensor::Dense(result_dense),
            labels: output_labels,
        })
    }
}
