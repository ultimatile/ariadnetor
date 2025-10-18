//! Einsum notation parser and utilities
//!
//! This module provides parsing and validation for Einstein summation notation,
//! commonly used to describe tensor contractions.
//!
//! # Examples
//!
//! ```rust,ignore
//! use tn_mlir::einsum::EinsumExpr;
//!
//! // Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
//! let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
//! assert_eq!(expr.lhs_indices(), &['i', 'j']);
//! assert_eq!(expr.rhs_indices(), &['j', 'k']);
//! assert_eq!(expr.out_indices(), &['i', 'k']);
//!
//! // Trace: tr = A[i,i]
//! let expr = EinsumExpr::parse("ii->").unwrap();
//!
//! // Transpose: B[j,i] = A[i,j]
//! let expr = EinsumExpr::parse("ij->ji").unwrap();
//! ```

use anyhow::{anyhow, bail, Context, Result};
use std::collections::{HashMap, HashSet};

/// Represents a parsed Einstein summation expression
///
/// Einsum notation describes tensor contractions using index characters.
/// The general form is: "lhs_indices,rhs_indices->out_indices"
///
/// # Examples
///
/// - `"ij,jk->ik"` - Matrix multiplication
/// - `"ii->"` - Trace (sum of diagonal)
/// - `"ij->ji"` - Transpose
/// - `"ijk,jkl->il"` - Higher-dimensional contraction
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EinsumExpr {
    /// Indices for the left-hand side tensor
    lhs_indices: Vec<char>,
    /// Indices for the right-hand side tensor
    rhs_indices: Vec<char>,
    /// Indices for the output tensor
    out_indices: Vec<char>,
}

impl EinsumExpr {
    /// Parse an einsum expression from string notation
    ///
    /// # Arguments
    ///
    /// * `notation` - Einsum expression (e.g., "ij,jk->ik")
    ///
    /// # Returns
    ///
    /// Parsed `EinsumExpr` on success, or error if notation is invalid
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Expression format is invalid
    /// - Contains invalid characters (only a-z allowed)
    /// - Missing required components
    pub fn parse(notation: &str) -> Result<Self> {
        let notation = notation.trim();

        // Split by "->" to separate inputs from output
        let parts: Vec<&str> = notation.split("->").collect();
        if parts.len() != 2 {
            bail!(
                "Invalid einsum notation: expected format 'inputs->output', got '{}'",
                notation
            );
        }

        let inputs = parts[0];
        let output = parts[1];

        // Split inputs by "," to get lhs and rhs
        let input_parts: Vec<&str> = inputs.split(',').collect();
        if input_parts.len() != 2 {
            bail!(
                "Invalid einsum notation: expected two input tensors separated by ',', got '{}'",
                inputs
            );
        }

        let lhs_str = input_parts[0].trim();
        let rhs_str = input_parts[1].trim();
        let out_str = output.trim();

        // Parse indices
        let lhs_indices = Self::parse_indices(lhs_str)
            .context("Failed to parse left-hand side indices")?;
        let rhs_indices = Self::parse_indices(rhs_str)
            .context("Failed to parse right-hand side indices")?;
        let out_indices = Self::parse_indices(out_str)
            .context("Failed to parse output indices")?;

        // Create expression
        let expr = Self {
            lhs_indices,
            rhs_indices,
            out_indices,
        };

        // Validate the expression
        expr.validate()?;

        Ok(expr)
    }

    /// Parse index string into vector of characters
    ///
    /// Only lowercase letters a-z are allowed as indices.
    fn parse_indices(s: &str) -> Result<Vec<char>> {
        let indices: Vec<char> = s.chars().collect();

        // Validate characters
        for &ch in &indices {
            if !ch.is_ascii_lowercase() {
                bail!(
                    "Invalid index character '{}': only lowercase letters a-z are allowed",
                    ch
                );
            }
        }

        Ok(indices)
    }

    /// Validate the einsum expression
    ///
    /// Checks that:
    /// - All output indices appear in at least one input
    /// - No unexpected index patterns
    /// - Repeated indices within a tensor are valid (for trace operations)
    pub fn validate(&self) -> Result<()> {
        // Collect all input indices
        let mut input_indices = HashSet::new();
        for &idx in &self.lhs_indices {
            input_indices.insert(idx);
        }
        for &idx in &self.rhs_indices {
            input_indices.insert(idx);
        }

        // Check that all output indices appear in inputs
        for &idx in &self.out_indices {
            if !input_indices.contains(&idx) {
                bail!(
                    "Output index '{}' does not appear in any input tensor",
                    idx
                );
            }
        }

        // Find contracted indices (appear in inputs but not in output)
        let output_set: HashSet<char> = self.out_indices.iter().copied().collect();
        let contracted: Vec<char> = input_indices
            .iter()
            .filter(|&&idx| !output_set.contains(&idx))
            .copied()
            .collect();

        // Validate contracted indices appear in both inputs
        for &idx in &contracted {
            let in_lhs = self.lhs_indices.contains(&idx);
            let in_rhs = self.rhs_indices.contains(&idx);

            if !in_lhs || !in_rhs {
                bail!(
                    "Contracted index '{}' must appear in both input tensors, found: lhs={}, rhs={}",
                    idx, in_lhs, in_rhs
                );
            }
        }

        Ok(())
    }

    /// Infer the output tensor shape from input shapes
    ///
    /// # Arguments
    ///
    /// * `lhs_shape` - Shape of the left-hand side tensor
    /// * `rhs_shape` - Shape of the right-hand side tensor
    ///
    /// # Returns
    ///
    /// Inferred output shape, or error if shapes are incompatible
    pub fn infer_output_shape(&self, lhs_shape: &[i64], rhs_shape: &[i64]) -> Result<Vec<i64>> {
        // Verify input shapes match index counts
        if lhs_shape.len() != self.lhs_indices.len() {
            bail!(
                "LHS shape rank {} does not match index count {}",
                lhs_shape.len(),
                self.lhs_indices.len()
            );
        }

        if rhs_shape.len() != self.rhs_indices.len() {
            bail!(
                "RHS shape rank {} does not match index count {}",
                rhs_shape.len(),
                self.rhs_indices.len()
            );
        }

        // Build index -> dimension mapping
        let mut index_dims: HashMap<char, i64> = HashMap::new();

        // Process LHS
        for (i, &idx) in self.lhs_indices.iter().enumerate() {
            let dim = lhs_shape[i];
            if let Some(&existing_dim) = index_dims.get(&idx) {
                // Index appears multiple times (e.g., "ii" for trace)
                if existing_dim != dim {
                    bail!(
                        "Dimension mismatch for index '{}': found {} and {}",
                        idx,
                        existing_dim,
                        dim
                    );
                }
            } else {
                index_dims.insert(idx, dim);
            }
        }

        // Process RHS
        for (i, &idx) in self.rhs_indices.iter().enumerate() {
            let dim = rhs_shape[i];
            if let Some(&existing_dim) = index_dims.get(&idx) {
                // Check consistency with LHS
                if existing_dim != dim {
                    bail!(
                        "Dimension mismatch for index '{}': LHS has {}, RHS has {}",
                        idx,
                        existing_dim,
                        dim
                    );
                }
            } else {
                index_dims.insert(idx, dim);
            }
        }

        // Build output shape
        let mut output_shape = Vec::new();
        for &idx in &self.out_indices {
            let dim = index_dims.get(&idx).ok_or_else(|| {
                anyhow!("Output index '{}' not found in input tensors", idx)
            })?;
            output_shape.push(*dim);
        }

        Ok(output_shape)
    }

    /// Get left-hand side indices
    pub fn lhs_indices(&self) -> &[char] {
        &self.lhs_indices
    }

    /// Get right-hand side indices
    pub fn rhs_indices(&self) -> &[char] {
        &self.rhs_indices
    }

    /// Get output indices
    pub fn out_indices(&self) -> &[char] {
        &self.out_indices
    }

    /// Get contracted indices (appear in inputs but not in output)
    pub fn contracted_indices(&self) -> Vec<char> {
        let output_set: HashSet<char> = self.out_indices.iter().copied().collect();
        let mut contracted = Vec::new();

        for &idx in &self.lhs_indices {
            if !output_set.contains(&idx) && !contracted.contains(&idx) {
                contracted.push(idx);
            }
        }
        for &idx in &self.rhs_indices {
            if !output_set.contains(&idx) && !contracted.contains(&idx) {
                contracted.push(idx);
            }
        }

        contracted
    }

    /// Check if this is a matrix multiplication pattern (ij,jk->ik)
    pub fn is_matrix_multiply(&self) -> bool {
        // Must have exactly 3 unique indices
        let mut all_indices: HashSet<char> = HashSet::new();
        all_indices.extend(&self.lhs_indices);
        all_indices.extend(&self.rhs_indices);

        if all_indices.len() != 3 {
            return false;
        }

        // LHS and RHS must each have 2 indices
        if self.lhs_indices.len() != 2 || self.rhs_indices.len() != 2 {
            return false;
        }

        // Output must have 2 indices
        if self.out_indices.len() != 2 {
            return false;
        }

        // One index must be contracted (appear in both inputs but not output)
        let contracted = self.contracted_indices();
        contracted.len() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_matrix_multiply() {
        let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
        assert_eq!(expr.lhs_indices(), &['i', 'j']);
        assert_eq!(expr.rhs_indices(), &['j', 'k']);
        assert_eq!(expr.out_indices(), &['i', 'k']);
        assert!(expr.is_matrix_multiply());
    }

    #[test]
    #[ignore = "Single tensor operations not yet supported - requires one input tensor"]
    fn test_parse_trace() {
        // Future: support single tensor operations like trace
        // let expr = EinsumExpr::parse("ii->").unwrap();
        // assert_eq!(expr.lhs_indices(), &['i', 'i']);
    }

    #[test]
    fn test_parse_element_wise() {
        // Element-wise product (like transpose with dummy second tensor)
        let expr = EinsumExpr::parse("ij,ij->ij").unwrap();
        assert_eq!(expr.lhs_indices(), &['i', 'j']);
        assert_eq!(expr.rhs_indices(), &['i', 'j']);
        assert_eq!(expr.out_indices(), &['i', 'j']);
    }

    #[test]
    fn test_parse_higher_dimensional() {
        let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
        assert_eq!(expr.lhs_indices(), &['i', 'j', 'k']);
        assert_eq!(expr.rhs_indices(), &['j', 'k', 'l']);
        assert_eq!(expr.out_indices(), &['i', 'l']);
        assert!(!expr.is_matrix_multiply());
    }

    #[test]
    fn test_contracted_indices() {
        let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
        let contracted = expr.contracted_indices();
        assert_eq!(contracted, vec!['j']);
    }

    #[test]
    fn test_invalid_format_no_arrow() {
        let result = EinsumExpr::parse("ij,jk");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected format"));
    }

    #[test]
    fn test_invalid_format_one_input() {
        let result = EinsumExpr::parse("ij->ik");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("two input tensors"));
    }

    #[test]
    fn test_invalid_character() {
        let result = EinsumExpr::parse("iJ,jk->ik");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Error message should indicate parsing failure
        assert!(
            err_msg.contains("lowercase")
            || err_msg.contains("Invalid")
            || err_msg.contains("Failed to parse")
        );
    }

    #[test]
    fn test_output_index_not_in_input() {
        let result = EinsumExpr::parse("ij,jk->im");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not appear"));
    }

    #[test]
    fn test_infer_shape_matrix_multiply() {
        let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
        let output = expr.infer_output_shape(&[10, 20], &[20, 30]).unwrap();
        assert_eq!(output, vec![10, 30]);
    }

    #[test]
    fn test_infer_shape_dimension_mismatch() {
        let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
        let result = expr.infer_output_shape(&[10, 20], &[25, 30]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Dimension mismatch"));
    }

    #[test]
    fn test_infer_shape_higher_dimensional() {
        let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
        let output = expr.infer_output_shape(&[5, 10, 15], &[10, 15, 20]).unwrap();
        assert_eq!(output, vec![5, 20]);
    }

    #[test]
    fn test_whitespace_handling() {
        let expr = EinsumExpr::parse("  ij , jk -> ik  ").unwrap();
        assert_eq!(expr.lhs_indices(), &['i', 'j']);
        assert_eq!(expr.rhs_indices(), &['j', 'k']);
        assert_eq!(expr.out_indices(), &['i', 'k']);
    }
}
