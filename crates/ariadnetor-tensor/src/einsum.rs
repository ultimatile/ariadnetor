//! Einstein summation (einsum) notation parser and contraction primitives
//!
//! This module implements NumPy-compatible einsum parsing and execution.
//!
//! # Notation
//!
//! Einstein notation: `"ijk,jkl->il"`
//! - Inputs: comma-separated index strings
//! - Output: after `->`
//! - Summation: indices not in output
//!
//! # Internal Representation
//!
//! Indices are represented as ASCII character codes (u8), following NumPy's approach.
//! This limits to 52 indices (A-Z, a-z) but ensures compatibility.

use std::collections::HashSet;

/// Einstein summation expression (internal representation)
///
/// Indices are stored as ASCII character codes for NumPy compatibility.
///
/// # Example
/// ```
/// use arnet_tensor::einsum::EinsumExpr;
///
/// let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
/// assert_eq!(expr.lhs_indices, vec![b'i', b'j', b'k']);
/// assert_eq!(expr.rhs_indices, vec![b'j', b'k', b'l']);
/// assert_eq!(expr.out_indices, vec![b'i', b'l']);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EinsumExpr {
    /// Left-hand side indices (e.g., "ijk" → [105, 106, 107])
    pub lhs_indices: Vec<u8>,
    /// Right-hand side indices (e.g., "jkl" → [106, 107, 108])
    pub rhs_indices: Vec<u8>,
    /// Output indices (e.g., "il" → [105, 108])
    pub out_indices: Vec<u8>,
}

impl EinsumExpr {
    /// Parse Einstein notation string
    ///
    /// # Format
    ///
    /// `"<lhs_indices>,<rhs_indices>-><out_indices>"`
    ///
    /// # Examples
    ///
    /// ```
    /// use arnet_tensor::einsum::EinsumExpr;
    ///
    /// // Matrix multiplication: C[i,j] = Σ_k A[i,k] * B[k,j]
    /// let expr = EinsumExpr::parse("ik,kj->ij").unwrap();
    ///
    /// // Tensor contraction: C[i,l] = Σ_j Σ_k A[i,j,k] * B[j,k,l]
    /// let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    ///
    /// // Trace: scalar = Σ_i A[i,i]
    /// let expr = EinsumExpr::parse("ii->").unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Missing "->" separator
    /// - Not exactly 2 input tensors (currently only binary operations supported)
    /// - Invalid characters (must be ASCII alphabetic)
    pub fn parse(notation: &str) -> Result<Self, String> {
        // Remove whitespace
        let notation = notation.chars().filter(|c| !c.is_whitespace()).collect::<String>();

        // Split by "->"
        let parts: Vec<&str> = notation.split("->").collect();
        if parts.len() != 2 {
            return Err(format!(
                "Invalid einsum notation: expected 'inputs->output', got '{}'",
                notation
            ));
        }

        // Split inputs by ","
        let inputs: Vec<&str> = parts[0].split(',').collect();

        // Parse indices (validate ASCII alphabetic)
        let (lhs_indices, rhs_indices) = match inputs.len() {
            1 => {
                // Unary operation (e.g., trace "ii->")
                (Self::parse_indices(inputs[0])?, Vec::new())
            }
            2 => {
                // Binary operation (e.g., "ijk,jkl->il")
                (Self::parse_indices(inputs[0])?, Self::parse_indices(inputs[1])?)
            }
            _ => {
                return Err(format!(
                    "Only unary or binary operations supported, got {} inputs",
                    inputs.len()
                ));
            }
        };

        let out_indices = Self::parse_indices(parts[1])?;

        Ok(Self {
            lhs_indices,
            rhs_indices,
            out_indices,
        })
    }

    /// Parse index string into ASCII character codes
    fn parse_indices(s: &str) -> Result<Vec<u8>, String> {
        let mut indices = Vec::new();
        for c in s.chars() {
            if !c.is_ascii_alphabetic() {
                return Err(format!(
                    "Invalid index character '{}': must be ASCII alphabetic (A-Z, a-z)",
                    c
                ));
            }
            indices.push(c as u8);
        }
        Ok(indices)
    }

    /// Get all unique indices across all tensors
    pub fn all_indices(&self) -> HashSet<u8> {
        let mut indices = HashSet::new();
        indices.extend(&self.lhs_indices);
        indices.extend(&self.rhs_indices);
        indices.extend(&self.out_indices);
        indices
    }
}

/// Contraction plan analysis
///
/// Identifies which indices are contracted (summed over) and which are free (remain in output).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionPlan {
    /// Indices that are summed over (in both inputs, not in output)
    pub contracted: Vec<u8>,
    /// Free indices from left-hand side (in lhs and in output)
    pub free_lhs: Vec<u8>,
    /// Free indices from right-hand side (in rhs and in output)
    pub free_rhs: Vec<u8>,
}

impl ContractionPlan {
    /// Analyze contraction indices from einsum expression
    ///
    /// # Example
    /// ```
    /// use arnet_tensor::einsum::{EinsumExpr, ContractionPlan};
    ///
    /// let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    /// let plan = ContractionPlan::from_expr(&expr);
    ///
    /// assert_eq!(plan.contracted, vec![b'j', b'k']);
    /// assert_eq!(plan.free_lhs, vec![b'i']);
    /// assert_eq!(plan.free_rhs, vec![b'l']);
    /// ```
    pub fn from_expr(expr: &EinsumExpr) -> Self {
        let lhs_set: HashSet<u8> = expr.lhs_indices.iter().copied().collect();
        let rhs_set: HashSet<u8> = expr.rhs_indices.iter().copied().collect();
        let out_set: HashSet<u8> = expr.out_indices.iter().copied().collect();

        // Contracted indices: in both inputs, not in output
        let mut contracted: Vec<u8> = lhs_set
            .intersection(&rhs_set)
            .filter(|idx| !out_set.contains(idx))
            .copied()
            .collect();

        // Sort for deterministic behavior
        contracted.sort_unstable();

        // Free indices (lhs): in output and in lhs, in output order
        let free_lhs: Vec<u8> = expr
            .out_indices
            .iter()
            .filter(|idx| lhs_set.contains(idx))
            .copied()
            .collect();

        // Free indices (rhs): in output and in rhs, in output order
        let free_rhs: Vec<u8> = expr
            .out_indices
            .iter()
            .filter(|idx| rhs_set.contains(idx))
            .copied()
            .collect();

        Self {
            contracted,
            free_lhs,
            free_rhs,
        }
    }

    /// Compute permutation for left-hand side tensor
    ///
    /// Reorders indices to: [free_lhs..., contracted...]
    /// where contracted indices are ordered as they appear in RHS for GEMM compatibility
    ///
    /// # Returns
    ///
    /// Permutation array, or `None` if no permutation needed (already in correct order)
    pub fn lhs_permutation(&self, lhs_indices: &[u8], rhs_indices: &[u8]) -> Option<Vec<usize>> {
        let contracted_set: HashSet<u8> = self.contracted.iter().copied().collect();
        let mut target_order = Vec::new();

        // Free indices first (in output order)
        target_order.extend(&self.free_lhs);

        // Then contracted indices (in RHS order for GEMM)
        for &idx in rhs_indices {
            if contracted_set.contains(&idx) && !target_order.contains(&idx) {
                target_order.push(idx);
            }
        }

        compute_permutation(lhs_indices, &target_order)
    }

    /// Compute permutation for right-hand side tensor
    ///
    /// Reorders indices to: [contracted..., free_rhs...]
    ///
    /// # Returns
    ///
    /// Permutation array, or `None` if no permutation needed
    pub fn rhs_permutation(&self, rhs_indices: &[u8]) -> Option<Vec<usize>> {
        let contracted_set: HashSet<u8> = self.contracted.iter().copied().collect();
        let mut target_order = Vec::new();

        // Contracted indices first (in their original RHS relative order)
        for &idx in rhs_indices {
            if contracted_set.contains(&idx) && !target_order.contains(&idx) {
                target_order.push(idx);
            }
        }

        // Then free indices (in output order)
        target_order.extend(&self.free_rhs);

        compute_permutation(rhs_indices, &target_order)
    }
}

/// Compute permutation array from current order to target order
///
/// # Returns
///
/// - `Some(permutation)` if reordering is needed
/// - `None` if current order matches target (identity permutation)
///
/// # Example
/// ```
/// use arnet_tensor::einsum::compute_permutation;
///
/// let current = vec![b'i', b'k', b'j'];
/// let target = vec![b'i', b'j', b'k'];
/// let perm = compute_permutation(&current, &target).unwrap();
/// assert_eq!(perm, vec![0, 2, 1]); // swap positions 1 and 2
/// ```
pub fn compute_permutation(current: &[u8], target: &[u8]) -> Option<Vec<usize>> {
    assert_eq!(
        current.len(),
        target.len(),
        "Permutation length mismatch: {} != {}",
        current.len(),
        target.len()
    );

    let permutation: Vec<usize> = target
        .iter()
        .map(|&idx| {
            current
                .iter()
                .position(|&x| x == idx)
                .expect("Index not found in current order")
        })
        .collect();

    // Check if identity permutation
    if permutation.iter().enumerate().all(|(i, &p)| i == p) {
        None // No permutation needed
    } else {
        Some(permutation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_einsum() {
        let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
        assert_eq!(expr.lhs_indices, vec![b'i', b'j', b'k']);
        assert_eq!(expr.rhs_indices, vec![b'j', b'k', b'l']);
        assert_eq!(expr.out_indices, vec![b'i', b'l']);
    }

    #[test]
    fn test_parse_einsum_matmul() {
        let expr = EinsumExpr::parse("ik,kj->ij").unwrap();
        assert_eq!(expr.lhs_indices, vec![b'i', b'k']);
        assert_eq!(expr.rhs_indices, vec![b'k', b'j']);
        assert_eq!(expr.out_indices, vec![b'i', b'j']);
    }

    #[test]
    fn test_parse_einsum_trace() {
        let expr = EinsumExpr::parse("ii->").unwrap();
        assert_eq!(expr.lhs_indices, vec![b'i', b'i']);
        assert_eq!(expr.rhs_indices, Vec::<u8>::new());
        assert_eq!(expr.out_indices, Vec::<u8>::new());
    }

    #[test]
    fn test_parse_einsum_whitespace() {
        let expr = EinsumExpr::parse(" i j k , j k l -> i l ").unwrap();
        assert_eq!(expr.lhs_indices, vec![b'i', b'j', b'k']);
        assert_eq!(expr.rhs_indices, vec![b'j', b'k', b'l']);
        assert_eq!(expr.out_indices, vec![b'i', b'l']);
    }

    #[test]
    fn test_parse_einsum_invalid_missing_arrow() {
        let result = EinsumExpr::parse("ijk,jkl");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expected 'inputs->output'"));
    }

    #[test]
    fn test_parse_einsum_invalid_too_many_inputs() {
        let result = EinsumExpr::parse("i,j,k->i");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unary or binary operations"));
    }

    #[test]
    fn test_parse_einsum_invalid_character() {
        let result = EinsumExpr::parse("i1j,jk->ik");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid index character"));
    }

    #[test]
    fn test_contraction_plan() {
        let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
        let plan = ContractionPlan::from_expr(&expr);

        assert_eq!(plan.contracted, vec![b'j', b'k']);
        assert_eq!(plan.free_lhs, vec![b'i']);
        assert_eq!(plan.free_rhs, vec![b'l']);
    }

    #[test]
    fn test_contraction_plan_matmul() {
        let expr = EinsumExpr::parse("ik,kj->ij").unwrap();
        let plan = ContractionPlan::from_expr(&expr);

        assert_eq!(plan.contracted, vec![b'k']);
        assert_eq!(plan.free_lhs, vec![b'i']);
        assert_eq!(plan.free_rhs, vec![b'j']);
    }

    #[test]
    fn test_lhs_permutation_identity() {
        let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
        let plan = ContractionPlan::from_expr(&expr);

        let perm = plan.lhs_permutation(&expr.lhs_indices, &expr.rhs_indices);
        assert_eq!(perm, None); // Already in correct order: [i, jk]
    }

    #[test]
    fn test_lhs_permutation_needed() {
        let expr = EinsumExpr::parse("ikj,jkl->il").unwrap();
        let plan = ContractionPlan::from_expr(&expr);

        let perm = plan.lhs_permutation(&expr.lhs_indices, &expr.rhs_indices);
        assert_eq!(perm, Some(vec![0, 2, 1])); // [i,k,j] → [i,j,k]
    }

    #[test]
    fn test_rhs_permutation_identity() {
        let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
        let plan = ContractionPlan::from_expr(&expr);

        let perm = plan.rhs_permutation(&expr.rhs_indices);
        assert_eq!(perm, None); // Already in correct order: [jk, l]
    }

    #[test]
    fn test_rhs_permutation_needed() {
        let expr = EinsumExpr::parse("ijk,ljk->il").unwrap();
        let plan = ContractionPlan::from_expr(&expr);

        let perm = plan.rhs_permutation(&expr.rhs_indices);
        assert_eq!(perm, Some(vec![1, 2, 0])); // [l,j,k] → [j,k,l]
    }

    #[test]
    fn test_compute_permutation_identity() {
        let current = vec![b'i', b'j', b'k'];
        let target = vec![b'i', b'j', b'k'];
        assert_eq!(compute_permutation(&current, &target), None);
    }

    #[test]
    fn test_compute_permutation() {
        let current = vec![b'i', b'k', b'j'];
        let target = vec![b'i', b'j', b'k'];
        let perm = compute_permutation(&current, &target).unwrap();
        assert_eq!(perm, vec![0, 2, 1]);
    }
}
